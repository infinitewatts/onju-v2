#!/usr/bin/env python3
"""Relay voice board requests between the voice server and the task board.

Runs on the board's host (the MacBook), on a timer:
1. Claims queued voice tasks from the voice server over SSH and appends one
   line per new task to the board log, where the orchestrator picks it up.
   Delivery is at-least-once; a local seen-ID file drops repeats.
2. Pushes a short board summary back so the voice agent can read the board.

It never dispatches agents or edits board.md. Stdlib only.

    python3 scripts/board_sync.py --host macmini
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REMOTE_STATE = "~/.local/state/onju-voice-agent"
REMOTE_BOARD = "~/onju-v2-live/.venv/bin/python ~/onju-v2-live/pipeline/services/board_mcp.py"
SNAPSHOT_MAX = 6000
ID_RE = re.compile(r"^voice-[0-9a-f]{12}$")
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]


def ssh(host: str, cmd: str, stdin: bytes | None = None) -> bytes:
    r = subprocess.run(["ssh", *SSH_OPTS, host, cmd], input=stdin,
                       capture_output=True, timeout=60)
    if r.returncode:
        raise RuntimeError(f"ssh exit {r.returncode}: {r.stderr.decode(errors='replace')[-200:]}")
    return r.stdout


def board_line(e: dict) -> str:
    """One board.log line in the board's ``ts | pane | task | state | text`` format."""
    def one(s, n):
        return " ".join(str(s or "").replace("|", "/").split())[:n]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = one(e.get("title"), 120)
    detail = one(e.get("detail"), 600)
    if detail:
        text += f" — {detail}"
    return (f"{ts} | onju-voice | Voice task | new-task | [{e['id']}, {one(e.get('source'), 40)}, "
            f"{one(e.get('device'), 40)}] {text}\n")


def deliver(items: list[dict], board_log: Path, seen_path: Path) -> tuple[int, list[str]]:
    """Append unseen entries; return (appended count, claimed files safe to ack)."""
    try:
        seen = set(seen_path.read_text().split())
    except OSError:
        seen = set()
    new_lines, new_ids, files = [], [], set()
    for item in items:
        e, f = item.get("entry") or {}, item.get("file", "")
        files.add(f)
        eid = e.get("id", "")
        if not ID_RE.match(eid) or eid in seen or eid in new_ids:
            continue
        new_lines.append(board_line(e))
        new_ids.append(eid)
    if new_lines:
        with open(board_log, "a") as fh:
            fh.writelines(new_lines)
            fh.flush()
            os.fsync(fh.fileno())
        with open(seen_path, "a") as fh:
            fh.write("".join(i + "\n" for i in new_ids))
        os.chmod(seen_path, 0o600)
    return len(new_lines), sorted(f for f in files if f)


def snapshot(board_md: Path) -> str:
    """Latest 'Today' section plus the Needs Eric table's task and next-action columns."""
    text = board_md.read_text()
    sections = re.split(r"(?m)^## ", text)
    out = []
    today = next((s for s in sections if s.startswith("Today")), None)
    if today:
        out.append("## " + today.strip())
    needs = next((s for s in sections if s.startswith("Needs Eric")), None)
    if needs:
        rows = []
        for line in needs.splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 3 and not set(cells[0]) <= set("- ") and cells[0] != "Task / tab (pane)":
                rows.append(f"- {cells[0]}: {cells[1]}. {cells[2]}")
        out.append("## Needs Eric\n" + "\n".join(rows))
    return "\n\n".join(out)[:SNAPSHOT_MAX] + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="macmini")
    p.add_argument("--board-md", default="~/.codex/board.md")
    p.add_argument("--board-log", default="~/.codex/board.log")
    p.add_argument("--seen", default="~/.local/state/onju-board-sync/seen-ids")
    a = p.parse_args()
    seen = Path(a.seen).expanduser()
    seen.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    out = ssh(a.host, f"{REMOTE_BOARD} --state-dir {REMOTE_STATE} --claim")
    items = [json.loads(l) for l in out.decode().splitlines() if l.strip()]
    n, files = deliver(items, Path(a.board_log).expanduser(), seen)
    if files:
        safe = " ".join(f for f in files if re.match(r"^claimed-\d+\.jsonl$", f))
        ssh(a.host, f"{REMOTE_BOARD} --state-dir {REMOTE_STATE} --ack {safe}")

    snap = snapshot(Path(a.board_md).expanduser()).encode()
    ssh(a.host, f"umask 077; mkdir -p {REMOTE_STATE} && cat > {REMOTE_STATE}/board-snapshot.md.tmp "
                f"&& mv {REMOTE_STATE}/board-snapshot.md.tmp {REMOTE_STATE}/board-snapshot.md", stdin=snap)
    print(f"{datetime.now().isoformat(timespec='seconds')} delivered={n} snapshot={len(snap)}B")
    return 0


if __name__ == "__main__":
    sys.exit(main())
