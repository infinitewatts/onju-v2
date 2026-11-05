import json
import os
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from pipeline.services import board_mcp, voice_agent

FAKE_CLAUDE = """#!{python}
import json, os, sys
args = sys.argv[1:]
log = os.environ["FAKE_CLAUDE_LOG"]
with open(log, "a") as f:
    f.write(json.dumps(args) + "\\n")
mode = os.environ.get("FAKE_CLAUDE_MODE", "ok")
if mode == "fail_resume" and "--resume" in args:
    sys.exit(1)
if mode == "error":
    print(json.dumps({{"is_error": True, "result": "boom"}}))
    sys.exit(1)
print(json.dumps({{"type": "result", "is_error": False, "result": "Hello from fake."}}))
"""


def settings(tmp: Path, claude_bin: str, **over) -> voice_agent.Settings:
    args = voice_agent.parse_args(["--workdir", str(tmp / "state"), "--claude-bin", claude_bin,
                                   "--mcp-config", str(tmp / "mcp.json"),
                                   "--allowed-tools", "WebSearch mcp__board__add_task"])
    for k, v in over.items():
        setattr(args, k, v)
    return voice_agent.Settings(args)


class CommandTests(unittest.TestCase):
    def test_locked_down_flags(self):
        with tempfile.TemporaryDirectory() as d:
            s = settings(Path(d), "claude")
            cmd = voice_agent.build_command(s, "hi", "sid-1", new=True)
        self.assertIn("--bare", cmd)
        self.assertIn("--strict-mcp-config", cmd)
        self.assertEqual(cmd[cmd.index("--permission-mode") + 1], "dontAsk")
        self.assertEqual(cmd[cmd.index("--tools") + 1], "WebSearch,WebFetch")
        self.assertEqual(cmd[cmd.index("--session-id") + 1], "sid-1")
        self.assertNotIn("--resume", cmd)
        self.assertNotIn("--dangerously-skip-permissions", cmd)
        self.assertIn("--max-budget-usd", cmd)

    def test_resume_on_existing_session(self):
        with tempfile.TemporaryDirectory() as d:
            cmd = voice_agent.build_command(settings(Path(d), "claude"), "hi", "sid-1", new=False)
        self.assertEqual(cmd[cmd.index("--resume") + 1], "sid-1")
        self.assertNotIn("--session-id", cmd)

    def test_parse_result(self):
        self.assertEqual(voice_agent.parse_result(b'{"result": " Hi. ", "is_error": false}'), "Hi.")
        self.assertEqual(voice_agent.parse_result(b'{"result": "x", "is_error": true}'), "")
        self.assertEqual(voice_agent.parse_result(b"not json"), "")

    def test_extract_last_user_message(self):
        body = {"messages": [{"role": "user", "content": "old"},
                             {"role": "assistant", "content": "a"},
                             {"role": "user", "content": [{"type": "text", "text": "new"}]}]}
        self.assertEqual(voice_agent.extract_user_text(body), "new")

    def test_session_store_expires(self):
        with tempfile.TemporaryDirectory() as d:
            store = voice_agent.SessionStore(Path(d) / "s.json", idle_s=60)
            sid, new = store.get("dev", 1000)
            self.assertTrue(new)
            store.touch("dev", sid, 1000)
            self.assertEqual(store.get("dev", 1030), (sid, False))
            self.assertTrue(store.get("dev", 1100)[1])
            reloaded = voice_agent.SessionStore(Path(d) / "s.json", idle_s=60)
            self.assertEqual(reloaded.get("dev", 1030), (sid, False))


class AppTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        fake = self.d / "claude"
        fake.write_text(FAKE_CLAUDE.format(python=sys.executable))
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        self.log = self.d / "calls.jsonl"
        os.environ["FAKE_CLAUDE_LOG"] = str(self.log)
        os.environ.pop("FAKE_CLAUDE_MODE", None)
        os.environ.pop("VOICE_AGENT_TOKEN", None)
        self.fake = str(fake)

    def tearDown(self):
        self.tmp.cleanup()

    def calls(self):
        return [json.loads(l) for l in self.log.read_text().splitlines()]

    def client(self):
        return TestClient(voice_agent.create_app(settings(self.d, self.fake)))

    def post(self, c, text="hi", user="onju-8DAB80", stream=False):
        return c.post("/v1/chat/completions", json={
            "model": "m", "user": user, "stream": stream,
            "messages": [{"role": "user", "content": text}]})

    def test_two_turns_share_a_session(self):
        c = self.client()
        r = self.post(c)
        self.assertEqual(r.json()["choices"][0]["message"]["content"], "Hello from fake.")
        self.post(c, "again")
        first, second = self.calls()
        sid = first[first.index("--session-id") + 1]
        self.assertEqual(second[second.index("--resume") + 1], sid)

    def test_stream_returns_sse(self):
        r = self.post(self.client(), stream=True)
        self.assertIn("Hello from fake.", r.text)
        self.assertTrue(r.text.rstrip().endswith("data: [DONE]"))

    def test_failed_resume_starts_fresh(self):
        c = self.client()
        self.post(c)
        os.environ["FAKE_CLAUDE_MODE"] = "fail_resume"
        r = self.post(c)
        self.assertEqual(r.json()["choices"][0]["message"]["content"], "Hello from fake.")
        self.assertIn("--session-id", self.calls()[-1])

    def test_error_gives_spoken_fallback(self):
        os.environ["FAKE_CLAUDE_MODE"] = "error"
        r = self.post(self.client())
        self.assertEqual(r.json()["choices"][0]["message"]["content"], voice_agent.FALLBACK_REPLY)

    def test_rejects_bad_device_id(self):
        self.assertEqual(self.post(self.client(), user="../x").status_code, 400)

    def test_token_required_when_set(self):
        os.environ["VOICE_AGENT_TOKEN"] = "t0k"
        c = self.client()
        self.assertEqual(self.post(c).status_code, 401)
        r = c.post("/v1/chat/completions", headers={"Authorization": "Bearer t0k"},
                   json={"user": "onju-8DAB80", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(r.status_code, 200)


class BoardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.board = board_mcp.Board(Path(self.tmp.name), "onju-8DAB80")

    def tearDown(self):
        self.tmp.cleanup()

    def entries(self):
        return [json.loads(l) for l in self.board.inbox.read_text().splitlines()]

    def test_add_task_is_tagged_and_cleaned(self):
        out = self.board.add_task("Call\x07 the\n  roofer", "about Monday")
        self.assertTrue(out.startswith("Queued"))
        (e,) = self.entries()
        self.assertEqual(e["title"], "Call the roofer")
        self.assertEqual(e["source"], "voice (unverified speaker)")
        self.assertEqual(e["device"], "onju-8DAB80")
        self.assertEqual(os.stat(self.board.inbox).st_mode & 0o777, 0o600)

    def test_rejects_empty_title(self):
        self.assertTrue(self.board.add_task("  ").startswith("Error"))

    def test_rate_limit(self):
        for i in range(board_mcp.MAX_TASKS_PER_HOUR):
            self.assertTrue(self.board.add_task(f"task {i}").startswith("Queued"))
        self.assertTrue(self.board.add_task("one more").startswith("Error"))

    def test_read_tasks(self):
        self.assertIn("isn't available", self.board.read_tasks())
        self.board.snapshot.write_text("Needs Eric: X")
        self.assertEqual(self.board.read_tasks(), "Needs Eric: X")
        old = time.time() - 8 * 3600
        os.utime(self.board.snapshot, (old, old))
        self.assertIn("8 hours old", self.board.read_tasks())

    def test_claim_is_at_least_once_until_ack(self):
        self.board.add_task("First task")
        first = self.board.claim()
        self.assertEqual([i["entry"]["title"] for i in first], ["First task"])
        self.board.add_task("Second task")
        again = self.board.claim()
        self.assertEqual([i["entry"]["title"] for i in again], ["First task", "Second task"])
        self.board.ack(sorted({i["file"] for i in again}) + ["../etc/passwd"])
        self.assertEqual(self.board.claim(), [])

    def test_rate_limit_survives_claim(self):
        for i in range(board_mcp.MAX_TASKS_PER_HOUR):
            self.board.add_task(f"task {i}")
        self.board.ack([i["file"] for i in self.board.claim()])
        self.assertTrue(self.board.add_task("one more").startswith("Error"))

    def test_mcp_protocol(self):
        init = board_mcp.handle(self.board, {"id": 1, "method": "initialize",
                                             "params": {"protocolVersion": "2025-06-18"}})
        self.assertEqual(init["result"]["protocolVersion"], "2025-06-18")
        self.assertIsNone(board_mcp.handle(self.board, {"method": "notifications/initialized"}))
        names = [t["name"] for t in board_mcp.handle(self.board, {"id": 2, "method": "tools/list"})
                 ["result"]["tools"]]
        self.assertEqual(names, ["read_tasks", "add_task"])
        r = board_mcp.handle(self.board, {"id": 3, "method": "tools/call",
                                          "params": {"name": "add_task", "arguments": {"title": "Buy breakers"}}})
        self.assertFalse(r["result"]["isError"])
        bad = board_mcp.handle(self.board, {"id": 4, "method": "tools/call",
                                            "params": {"name": "send_text", "arguments": {}}})
        self.assertIn("error", bad)


class BoardSyncTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import board_sync
        self.bs = board_sync
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def item(self, eid, title="Order breakers | now", f="claimed-1.jsonl"):
        return {"file": f, "entry": {"id": eid, "title": title, "detail": "two 40A",
                                     "source": "voice (unverified speaker)", "device": "onju-8DAB80"}}

    def test_deliver_dedups_and_formats(self):
        log, seen = self.d / "board.log", self.d / "seen"
        items = [self.item("voice-0123456789ab"), self.item("voice-0123456789ab"),
                 self.item("bad id; rm -rf"), self.item("voice-aaaaaaaaaaaa", f="claimed-2.jsonl")]
        n, files = self.bs.deliver(items, log, seen)
        self.assertEqual(n, 2)
        self.assertEqual(files, ["claimed-1.jsonl", "claimed-2.jsonl"])
        lines = log.read_text().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].split(" | ")[1:4], ["onju-voice", "Voice task", "new-task"])
        self.assertIn("Order breakers / now — two 40A", lines[0])
        self.assertEqual(self.bs.deliver(items, log, seen)[0], 0)  # redelivery is a no-op
        self.assertEqual(len(log.read_text().splitlines()), 2)

    def test_snapshot_keeps_today_and_needs_eric(self):
        md = self.d / "board.md"
        md.write_text("# Task board\nintro\n\n## Today — Thu\n\n- item one\n\n## Earlier\n- old\n\n"
                      "## Needs Eric\n\n| Task / tab (pane) | State | Exact next action | Evidence |\n"
                      "| --- | --- | --- | --- |\n| Roof job (`w1`) | needs-eric | Call Bob | x |\n")
        snap = self.bs.snapshot(md)
        self.assertIn("item one", snap)
        self.assertNotIn("old", snap)
        self.assertIn("- Roof job (`w1`): needs-eric. Call Bob", snap)
        self.assertNotIn("---", snap)


if __name__ == "__main__":
    unittest.main()
