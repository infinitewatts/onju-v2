"""Local-only adapter checks; no speaker, SSH tunnel, or Claude session."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import unittest
from unittest.mock import patch

from pipeline.services.orchestrator_bridge import Adapter, BridgeError, Client, handle

HOST = "local_test-orchestrator"
SESSION = "session-one"
QUESTION = {"question": "Which format?", "header": "Format", "options": [{"label": "Brief", "description": "Short report"}, {"label": "Detailed", "description": "Long report"}], "multiSelect": False}
EVENT = {"hook_event_name": "PreToolUse", "tool_name": "AskUserQuestion", "session_id": SESSION, "tool_use_id": "toolu_one", "tool_input": {"questions": [QUESTION], "metadata": {"preserve": True}}}


class Fake:
    def __init__(self, capture=True):
        self.calls, self.records = [], {}
        self.capture = capture
    def call(self, p):
        self.calls.append(copy.deepcopy(p))
        key = p["question_id"]
        if p["op"] == "submit":
            q = {**p, "status": "queued"}
            self.records[key] = q
        else:
            if key not in self.records:
                raise BridgeError("Question not found")
            q = self.records[key]
            if p["op"] == "status" and self.capture and q["status"] == "queued":
                q.update(status="captured", answer_id="answer-" + key, answer={"choice": 1, "text": "Brief"})
            elif p["op"] == "emitted":
                q["status"] = "emitted"
            elif p["op"] == "ack":
                q["status"] = "acknowledged"
            elif p["op"] in ("cancel", "screen"):
                q["status"] = "cancelled" if p["op"] == "cancel" else "needs_desktop"
        return {"question": copy.deepcopy(q)}


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.now = 100
        self.fake = Fake()
        self.config = {"mode": "live", "host_session_id": HOST, "session_id": SESSION, "device": "onju-8DAB80", "state_dir": self.temp.name}
        self.adapter = Adapter(self.config, client=self.fake, env={"CLAUDE_CODE_HOST_SESSION_ID": HOST}, clock=lambda: self.now, sleep=self.advance)
    def tearDown(self):
        self.temp.cleanup()
    def advance(self, seconds):
        self.now += seconds
    def test_preserves_full_input_and_ack_only_matching_post_result(self):
        before = copy.deepcopy(EVENT)
        result = self.adapter.hook(EVENT)["hookSpecificOutput"]
        self.assertEqual(EVENT, before)
        self.assertEqual(result["updatedInput"]["metadata"], {"preserve": True})
        self.assertEqual(result["updatedInput"]["questions"], [QUESTION])
        self.assertEqual(result["updatedInput"]["answers"], {"Which format?": "Brief"})
        self.assertEqual(self.fake.calls[-1]["op"], "emitted")
        post = {**EVENT, "hook_event_name": "PostToolUse", "tool_response": {"answers": {"Which format?": "Detailed"}}}
        self.adapter.hook(post)
        self.assertFalse(any(p["op"] == "ack" for p in self.fake.calls))
        post["tool_response"]["answers"]["Which format?"] = "Brief"
        self.adapter.hook({**post, "tool_use_id": "toolu_other"})
        self.assertFalse(any(p["op"] == "ack" for p in self.fake.calls))
        self.adapter.hook(post)
        self.assertEqual(self.fake.calls[-1]["op"], "ack")
    def test_wrong_host_cannot_initialize(self):
        with self.assertRaises(BridgeError):
            Adapter(self.config, client=self.fake, env={"CLAUDE_CODE_HOST_SESSION_ID": "other"})
        self.assertEqual(self.fake.calls, [])
    def test_observe_never_calls_transport_or_stores_question_text(self):
        self.adapter.mode = "observe"
        self.assertEqual(self.adapter.hook(EVENT), {})
        result = self.adapter.tool("aiden_question", {"question_id": "plain", "version": 1, "text": "Which format?"})
        self.assertEqual(result["status"], "observed")
        self.assertEqual(self.fake.calls, [])
        for f in Path(self.temp.name).glob("observe-*.json"):
            self.assertNotIn("Which format", f.read_text())
    def test_timeout_renews_then_cancels_before_native_ui(self):
        self.fake.capture = False
        self.assertEqual(self.adapter.hook(EVENT), {})
        self.assertTrue(any(p["op"] == "renew" for p in self.fake.calls))
        self.assertEqual(self.fake.calls[-1]["op"], "cancel")
        self.assertAlmostEqual(self.now, 155)
    def test_stale_answer_or_desktop_handoff_does_not_emit(self):
        original = self.fake.call
        def stale(p):
            result = original(p)
            if p["op"] == "status":
                result["question"]["version"] = 2
            return result
        self.fake.call = stale
        self.assertEqual(self.adapter.hook(EVENT), {})
        self.assertFalse(any(p["op"] == "emitted" for p in self.fake.calls))
    def test_sensitive_multiselect_visual_and_duplicate_batch_stay_native(self):
        for changes in ({"question": "Approve payment?"}, {"multiSelect": True}, {"preview": "image"}, {"requires_desktop": True}, {"options": [{"label": "one", "preview": "visual"}]}):
            event = copy.deepcopy(EVENT)
            event["tool_input"]["questions"][0].update(changes)
            self.assertEqual(self.adapter.hook(event), {})
        event = copy.deepcopy(EVENT)
        event["tool_input"]["questions"] *= 2
        self.assertEqual(self.adapter.hook(event), {})
        self.assertEqual(self.fake.calls, [])
    def test_explicit_needs_separate_ack_and_caller_cannot_override_session(self):
        result = self.adapter.tool("aiden_question", {"question_id": "plain", "version": 1, "text": "Which format?"})
        self.assertEqual(result["status"], "emitted")
        self.assertNotIn("ack", [p["op"] for p in self.fake.calls])
        self.adapter.tool("aiden_acknowledge", {"question_id": result["question_id"], "version": 1, "answer_id": result["answer_id"]})
        self.assertEqual(self.fake.calls[-1]["op"], "ack")
        self.assertEqual(self.fake.calls[-1]["session_id"], SESSION)
        with self.assertRaises(BridgeError):
            self.adapter.tool("aiden_cancel", {"question_id": "plain", "version": 1, "session_id": "injected"})
    def test_enrollment_is_private_exact_host_and_mcp_fails_without_it(self):
        self.adapter.config.pop("session_id")
        with self.assertRaises(BridgeError):
            self.adapter.binding()
        self.adapter.hook({"hook_event_name": "SessionStart", "session_id": SESSION})
        self.assertEqual(self.adapter.binding(), SESSION)
        self.assertEqual((Path(self.temp.name) / "binding.json").stat().st_mode & 0o777, 0o600)
    def test_enrollment_retains_validated_optional_directory_metadata(self):
        self.adapter.env["CLAUDE_PROJECT_DIR"] = self.temp.name
        self.adapter.hook({"hook_event_name": "SessionStart", "session_id": SESSION, "cwd": str(Path(self.temp.name).parent)})
        binding = json.loads((Path(self.temp.name) / "binding.json").read_text())
        self.assertEqual(binding["cwd"], str(Path(self.temp.name).parent))
        self.assertEqual(binding["project_dir"], self.temp.name)
        self.assertEqual(binding["host_session_id"], HOST)
        self.adapter.env.pop("CLAUDE_PROJECT_DIR")
        self.adapter.hook({"hook_event_name": "SessionStart", "session_id": SESSION, "cwd": self.temp.name})
        binding = json.loads((Path(self.temp.name) / "binding.json").read_text())
        self.assertIsNone(binding["project_dir"])  # Never infer root from cwd.
        for invalid in ("relative/path", "/does-not-exist/aiden-fixture", 123):
            with self.assertRaises(BridgeError):
                self.adapter.hook({"hook_event_name": "SessionStart", "session_id": SESSION, "cwd": invalid})
        self.adapter.env["CLAUDE_PROJECT_DIR"] = "relative/root"
        with self.assertRaises(BridgeError):
            self.adapter.hook({"hook_event_name": "SessionStart", "session_id": SESSION, "cwd": self.temp.name})
    def test_transport_error_cancels_pending_and_never_emits_partial_answers(self):
        event = copy.deepcopy(EVENT)
        event["tool_input"]["questions"].append({**QUESTION, "question": "Which length?"})
        original = self.fake.call
        def fail_second(p):
            if p["op"] == "submit" and len(self.fake.records) == 1:
                raise BridgeError("down")
            return original(p)
        self.fake.call = fail_second
        with self.assertRaises(BridgeError):
            self.adapter.hook(event)
        self.assertFalse(any(p["op"] == "emitted" for p in self.fake.calls))
        self.assertTrue(any(p["op"] == "cancel" for p in self.fake.calls))
    def test_two_questions_complete_before_single_hook_output(self):
        event = copy.deepcopy(EVENT)
        event["tool_input"]["questions"].append({**QUESTION, "question": "Which length?"})
        output = self.adapter.hook(event)["hookSpecificOutput"]["updatedInput"]
        self.assertEqual(len(output["answers"]), 2)
        self.assertEqual([p["op"] for p in self.fake.calls][-2:], ["emitted", "emitted"])
    def test_real_store_multiquestion_capture_emit_and_ack(self):
        from pipeline.orchestrator_bridge import BridgeStore
        store = BridgeStore({"mode": "live", "host_session_id": HOST, "device": "onju-8DAB80", "state_path": str(Path(self.temp.name) / "core" / "state.db")}, clock=lambda: self.now)
        class StoreClient:
            def call(inner, payload):
                if payload["op"] == "status":
                    q = store.dispatch(payload)["question"]
                    if q["status"] == "queued":
                        ident = {key: q[key] for key in ("host_session_id", "session_id", "question_id", "version")}
                        store.dispatch({**ident, "op": "announced"})
                        store.dispatch({**ident, "op": "propose", "answer_id": "answer-" + q["question_id"], "answer": {"choice": 1, "text": "Brief"}})
                return store.dispatch(payload)
        self.adapter.client = StoreClient()
        event = copy.deepcopy(EVENT)
        event["tool_input"]["questions"].append({**QUESTION, "question": "Which length?"})
        try:
            out = self.adapter.hook(event)["hookSpecificOutput"]["updatedInput"]
            self.assertEqual(len(out["answers"]), 2)
            self.adapter.hook({**event, "hook_event_name": "PostToolUse", "tool_response": {"answers": out["answers"]}})
            rows = store._rows()
            self.assertEqual([q["status"] for _, q in rows], ["acknowledged", "acknowledged"])
        finally:
            store.close()
    def test_mcp_protocol_and_actionable_tool_errors(self):
        self.assertIsNone(handle(self.adapter, {"jsonrpc": "2.0", "method": "notifications/initialized"}))
        response = handle(self.adapter, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25"}})
        self.assertEqual(response["result"]["protocolVersion"], "2025-11-25")
        tools = handle(self.adapter, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
        self.assertEqual(len(tools), 3)
        error = handle(self.adapter, {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "aiden_cancel", "arguments": {}}})
        self.assertTrue(error["result"]["isError"])
    def test_uncertain_cancel_waits_for_last_possible_lease_before_ui(self):
        original = self.fake.call
        self.fake.capture = False
        def fail_cleanup(p):
            if p["op"] == "cancel":
                raise BridgeError("transport uncertain")
            return original(p)
        self.fake.call = fail_cleanup
        self.assertEqual(self.adapter.hook(EVENT), {})
        self.assertGreaterEqual(self.now, 166)  # last renewal153 + 10s lease + 2s margin
    def test_storage_cap_fails_before_any_live_submit(self):
        for index in range(1024):
            (Path(self.temp.name) / f"retained-{index}").touch()
        with self.assertRaises(BridgeError):
            self.adapter.hook(EVENT)
        self.assertEqual(self.fake.calls, [])
    def test_explicit_sensitive_question_immediately_hands_to_desktop(self):
        result = self.adapter.tool("aiden_question", {"question_id": "sensitive", "version": 1, "text": "Approve payment?"})
        self.assertEqual(result["status"], "needs_desktop")
        self.assertEqual([p["op"] for p in self.fake.calls], ["submit", "screen", "cancel"])
    def test_actual_stdio_has_only_json_protocol_output(self):
        config_path = Path(self.temp.name) / "config.json"
        config_path.write_text(json.dumps(self.config))
        messages = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}}, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, {"jsonrpc": "2.0", "id": 3, "method": "ping"}]
        proc = subprocess.run([sys.executable, "-m", "pipeline.services.orchestrator_bridge", "--config", str(config_path), "mcp"], input="".join(json.dumps(m) + "\n" for m in messages), text=True, capture_output=True, env={**os.environ, "CLAUDE_CODE_HOST_SESSION_ID": HOST}, timeout=5)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual([json.loads(line)["id"] for line in proc.stdout.splitlines()], [1, 2, 3])


class TransportTests(unittest.TestCase):
    def test_no_remote_url_credentials_redirects_or_proxy(self):
        for url in ("https://127.0.0.1:13012", "http://localhost:13012", "http://example.com", "http://a@127.0.0.1", "http://127.0.0.1/path"):
            with self.assertRaises(BridgeError):
                Client(url)
        class Redirect(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(302)
                self.send_header("Location", "http://example.com")
                self.end_headers()
            def log_message(self, *args):
                pass
        server = HTTPServer(("127.0.0.1", 0), Redirect)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            with patch.dict(os.environ, {"http_proxy": "http://invalid:1"}), self.assertRaises(BridgeError):
                Client(f"http://127.0.0.1:{server.server_port}").call({"op": "status"})
        finally:
            server.shutdown()
            thread.join()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
