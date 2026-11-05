"""OpenAI-compatible gateway that answers voice turns with headless Claude Code.

The pipeline's ``agentic`` backend POSTs one user message per turn to
``/v1/chat/completions``. This service runs ``claude -p`` with a fixed, small
tool set and resumes one session per device, so the conversation keeps context.

Security model: anyone in the room can speak, and the speaker is not
authenticated. The agent therefore only gets web lookups plus the tools in the
MCP config passed with ``--mcp-config`` (e.g. board read/add). It never gets
Bash, file edits, or the operator's own MCP servers, hooks or CLAUDE.md
(``--bare --strict-mcp-config``). Unlisted tools are denied, not prompted
(``--permission-mode dontAsk``).

Run (loopback only):
    ANTHROPIC_API_KEY=... python -m pipeline.services.voice_agent --port 18789
"""
import argparse
import asyncio
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

log = logging.getLogger("voice_agent")

DEVICE_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_USER_CHARS = 4000
FALLBACK_REPLY = "Sorry, I couldn't finish that one. Try asking again."

DEFAULT_POLICY = """\
You are Eric's home voice assistant, speaking through a small speaker.
Your words are read aloud by text-to-speech: answer in short, plain spoken
sentences. No markdown, lists, code, links or emoji.

Anyone in the house may be speaking, and you cannot verify who. So:
- You may chat, answer questions, and look things up on the web.
- You may read the task board and add a task to it when asked. Adding a task
  is a request for Eric's agents, not an approval. Say it was added to the
  board, never that the work is done or approved.
- Never claim to send messages or emails, move money, change customer
  records, or touch production. If asked, offer to add a board task instead.
- Never read secrets, passwords or codes aloud.
If a tool fails, say so briefly instead of guessing."""


class Settings:
    def __init__(self, args: argparse.Namespace):
        self.claude_bin = args.claude_bin
        self.model = args.model
        self.workdir = Path(args.workdir).expanduser()
        self.mcp_config = args.mcp_config
        self.allowed_tools = args.allowed_tools
        self.builtin_tools = args.tools
        self.max_budget_usd = args.max_budget_usd
        self.timeout = args.timeout
        self.session_idle_s = args.session_idle_s
        self.policy = Path(args.policy_file).read_text() if args.policy_file else DEFAULT_POLICY
        self.state_file = self.workdir / "sessions.json"


class SessionStore:
    """device_id -> (session uuid, last-used time), persisted across restarts."""

    def __init__(self, path: Path, idle_s: float):
        self.path = path
        self.idle_s = idle_s
        try:
            self.data = json.loads(path.read_text())
        except (OSError, ValueError):
            self.data = {}

    def get(self, device: str, now: float) -> tuple[str, bool]:
        """Return (session_id, is_new). Sessions expire after idle_s."""
        entry = self.data.get(device)
        if entry and now - entry["t"] < self.idle_s:
            return entry["id"], False
        return str(uuid.uuid4()), True

    def touch(self, device: str, sid: str, now: float) -> None:
        self.data[device] = {"id": sid, "t": now}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data))
        os.chmod(tmp, 0o600)
        tmp.replace(self.path)

    def drop(self, device: str) -> None:
        self.data.pop(device, None)


def build_command(s: Settings, sid: str, new: bool) -> list[str]:
    """The spoken text goes on stdin, never argv: no option parsing, not in ps."""
    cmd = [
        s.claude_bin, "-p",
        "--bare",
        "--disable-slash-commands",
        "--output-format", "json",
        "--model", s.model,
        "--tools", s.builtin_tools,
        "--permission-mode", "dontAsk",
        "--strict-mcp-config",
        "--system-prompt", s.policy,
        "--max-budget-usd", str(s.max_budget_usd),
    ]
    if s.allowed_tools:
        cmd += ["--allowedTools", s.allowed_tools]
    if s.mcp_config:
        cmd += ["--mcp-config", s.mcp_config]
    cmd += ["--session-id", sid] if new else ["--resume", sid]
    return cmd


def parse_result(stdout: bytes) -> str:
    """Extract the reply text from `claude -p --output-format json`."""
    try:
        data = json.loads(stdout)
    except ValueError:
        return ""
    if isinstance(data, dict) and not data.get("is_error"):
        return str(data.get("result") or "").strip()
    return ""


def extract_user_text(body: dict) -> str:
    msgs = body.get("messages") or []
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            return str(c or "")[:MAX_USER_CHARS]
    return ""


def create_app(s: Settings) -> FastAPI:
    s.workdir.mkdir(parents=True, exist_ok=True, mode=0o700)
    store = SessionStore(s.state_file, s.session_idle_s)
    locks: dict[str, asyncio.Lock] = {}
    expected_key = os.environ.get("VOICE_AGENT_TOKEN", "")
    app = FastAPI()

    async def run_turn(device: str, text: str) -> str:
        lock = locks.setdefault(device, asyncio.Lock())
        async with lock:
            sid, new = store.get(device, time.time())
            proc = await asyncio.create_subprocess_exec(
                *build_command(s, sid, new), cwd=s.workdir,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                out, err = await asyncio.wait_for(proc.communicate(text.encode()), s.timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                log.warning(f"[{device}] claude timed out after {s.timeout}s")
                return FALLBACK_REPLY
            reply = parse_result(out)
            if proc.returncode == 0 and reply:
                store.touch(device, sid, time.time())
                return reply
            # Never retry: the failed run may already have queued a board task.
            # Drop the session so the next turn starts clean.
            log.warning(f"[{device}] claude exit {proc.returncode}: "
                        f"{err.decode(errors='replace')[-200:]}")
            store.drop(device)
            return FALLBACK_REPLY

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        if expected_key:
            auth = request.headers.get("authorization", "")
            if auth != f"Bearer {expected_key}":
                raise HTTPException(401)
        body = await request.json()
        device = str(body.get("user") or "")
        if not DEVICE_RE.match(device):
            raise HTTPException(400, "user must be a device id")
        text = extract_user_text(body).strip()
        reply = await run_turn(device, text) if text else "Sorry, I didn't catch that."
        model = body.get("model", "voice-agent")
        if body.get("stream"):
            async def chunks():
                chunk = {"id": "va", "object": "chat.completion.chunk", "model": model,
                         "choices": [{"index": 0, "delta": {"content": reply},
                                      "finish_reason": None}]}
                yield "data: " + json.dumps(chunk) + "\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(chunks(), media_type="text/event-stream")
        return JSONResponse({
            "id": "va", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": reply}}],
        })

    @app.get("/health")
    async def health():
        return {"ok": True}

    return app


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18789)
    p.add_argument("--claude-bin", default="claude")
    p.add_argument("--model", default="sonnet")
    p.add_argument("--workdir", default="~/.local/state/onju-voice-agent")
    p.add_argument("--policy-file", default="")
    p.add_argument("--mcp-config", default="", help="JSON file with the allowed MCP servers")
    p.add_argument("--tools", default="WebSearch,WebFetch", help="built-in tools")
    p.add_argument("--allowed-tools", default="WebSearch WebFetch",
                   help="pre-approved tools; everything else is denied")
    p.add_argument("--max-budget-usd", type=float, default=0.50, help="per-turn cap")
    p.add_argument("--timeout", type=float, default=90)
    p.add_argument("--session-idle-s", type=float, default=1800)
    return p.parse_args(argv)


def main():
    args = parse_args()
    if args.host not in ("127.0.0.1", "::1", "localhost"):
        raise SystemExit("voice_agent must listen on loopback only")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    import uvicorn
    uvicorn.run(create_app(Settings(args)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
