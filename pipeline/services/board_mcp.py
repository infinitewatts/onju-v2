"""Minimal stdio MCP server: read the task board snapshot, queue a new task.

The voice agent gets exactly two tools:
- read_tasks: returns the board snapshot file (pushed here by the board sync job).
- add_task:   appends one validated request to an inbox JSONL. The sync job on
              the board's host moves inbox lines into the board log, where the
              orchestrator picks them up like any agent's status line.

Nothing here dispatches work, sends messages, or runs commands. Speakers are
unverified, so every queued task is tagged ``source: voice (unverified speaker)``.

Usage (from an MCP config):
    python -m pipeline.services.board_mcp --state-dir ~/.local/state/onju-voice-agent
"""
import argparse
import fcntl
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

TITLE_MAX = 120
DETAIL_MAX = 1000
MAX_TASKS_PER_HOUR = 10
SNAPSHOT_MAX = 6000
SNAPSHOT_STALE_S = 6 * 3600
CLAIMED_RE = re.compile(r"^claimed-\d+\.jsonl$")
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

TOOLS = [
    {
        "name": "read_tasks",
        "description": "Read the current task board summary (what is open and what needs Eric).",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "add_task",
        "description": (
            "Add a task to the board for Eric's agents. It is a request, not an approval. "
            "Use a short title and a plain description of what the speaker asked for."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "maxLength": TITLE_MAX},
                "detail": {"type": "string", "maxLength": DETAIL_MAX},
            },
            "required": ["title"],
            "additionalProperties": False,
        },
    },
]


def clean(text: str, limit: int) -> str:
    text = CONTROL_CHARS.sub(" ", str(text or ""))
    return " ".join(text.split())[:limit]


class Board:
    def __init__(self, state_dir: Path, device: str):
        self.dir = state_dir
        self.dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.inbox = self.dir / "board-inbox.jsonl"
        self.snapshot = self.dir / "board-snapshot.md"
        self.device = device

    def read_tasks(self) -> str:
        try:
            st = self.snapshot.stat()
            text = self.snapshot.read_text()[:SNAPSHOT_MAX]
        except OSError:
            return "The board isn't available right now."
        age_h = (time.time() - st.st_mtime) / 3600
        note = f"(Board snapshot is {age_h:.0f} hours old.)\n" if age_h * 3600 > SNAPSHOT_STALE_S else ""
        return note + text

    def _rate_ok(self, now: float) -> bool:
        """Sliding one-hour window, kept apart from the inbox so claiming
        the inbox doesn't reset it. Caller holds the inbox lock."""
        path = self.dir / "rate.json"
        try:
            stamps = [t for t in json.loads(path.read_text()) if now - t < 3600]
        except (OSError, ValueError, TypeError):
            stamps = []
        if len(stamps) >= MAX_TASKS_PER_HOUR:
            return False
        path.write_text(json.dumps(stamps + [now]))
        return True

    def claim(self) -> list[dict]:
        """Move the inbox aside and return every unacked claimed entry.
        Re-returns earlier claims until acked, so delivery is at-least-once."""
        with open(self.dir / ".inbox.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.inbox.exists() and self.inbox.stat().st_size:
                self.inbox.rename(self.dir / f"claimed-{time.time_ns()}.jsonl")
        out = []
        for f in sorted(self.dir.glob("claimed-*.jsonl")):
            for line in f.read_text().splitlines():
                try:
                    out.append({"file": f.name, "entry": json.loads(line)})
                except ValueError:
                    continue
        return out

    def ack(self, names: list[str]) -> None:
        for name in names:
            if CLAIMED_RE.match(name):
                (self.dir / name).unlink(missing_ok=True)

    def add_task(self, title: str, detail: str = "") -> str:
        title = clean(title, TITLE_MAX)
        detail = clean(detail, DETAIL_MAX)
        if len(title) < 3:
            return "Error: the task needs a short title."
        now = time.time()
        with open(self.dir / ".inbox.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not self._rate_ok(now):
                return "Error: too many voice tasks this hour. Ask Eric to add it directly."
            entry = {
                "id": f"voice-{uuid.uuid4().hex[:12]}",
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "epoch": now,
                "device": self.device,
                "source": "voice (unverified speaker)",
                "title": title,
                "detail": detail,
            }
            fd = os.open(self.inbox, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.write(fd, (json.dumps(entry) + "\n").encode())
                os.fsync(fd)
            finally:
                os.close(fd)
        return f"Queued for the board as {entry['id']}: {title}"


def handle(board: Board, msg: dict) -> dict | None:
    method = msg.get("method")
    mid = msg.get("id")
    if mid is None:
        return None  # notification
    if method == "initialize":
        version = (msg.get("params") or {}).get("protocolVersion", "2025-06-18")
        result = {"protocolVersion": version, "capabilities": {"tools": {}},
                  "serverInfo": {"name": "onju-board", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = msg.get("params") or {}
        args = params.get("arguments") or {}
        name = params.get("name")
        if name == "read_tasks":
            text = board.read_tasks()
        elif name == "add_task":
            text = board.add_task(args.get("title", ""), args.get("detail", ""))
        else:
            return {"jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32602, "message": f"unknown tool {name}"}}
        result = {"content": [{"type": "text", "text": text}],
                  "isError": text.startswith("Error:")}
    elif method == "ping":
        result = {}
    else:
        return {"jsonrpc": "2.0", "id": mid,
                "error": {"code": -32601, "message": f"unknown method {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state-dir", default="~/.local/state/onju-voice-agent")
    p.add_argument("--device", default="onju")
    p.add_argument("--claim", action="store_true", help="print claimed inbox entries as JSON lines")
    p.add_argument("--ack", nargs="*", metavar="FILE", help="delete delivered claimed files")
    a = p.parse_args()
    board = Board(Path(a.state_dir).expanduser(), a.device)
    if a.claim:
        for item in board.claim():
            print(json.dumps(item))
        return
    if a.ack is not None:
        board.ack(a.ack)
        return
    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        reply = handle(board, msg)
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
