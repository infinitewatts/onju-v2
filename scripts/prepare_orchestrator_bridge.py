"""Prepare a private observe-only bridge bundle. Never installs or starts it."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys

SOURCE = Path(__file__).resolve().parents[1] / "pipeline/services/orchestrator_bridge.py"


def prepare(output_dir, host_session_id, reviewed_sha=None, runtime_dir=None):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", host_session_id):
        raise ValueError("Invalid pinned host session ID")
    if reviewed_sha is not None and not re.fullmatch(r"[a-f0-9]{40}", reviewed_sha):
        raise ValueError("Reviewed SHA must be the full 40-character reviewed commit")
    output = Path(output_dir).expanduser().absolute()
    # No recursive creation: the caller must choose an existing trusted parent.
    if not output.parent.is_dir() or output.parent.is_symlink() or output.exists() or output.is_symlink():
        raise ValueError("Use a new output directory in an existing trusted parent")
    runtime = Path(runtime_dir).expanduser().absolute() if runtime_dir else output
    if runtime.is_symlink():
        raise ValueError("Runtime directory must not itself be a symlink")
    output.mkdir(mode=0o700)
    source = SOURCE.read_bytes()
    files = {"adapter.py": source}
    config = {"mode": "observe", "host_session_id": host_session_id, "device": "onju-8DAB80", "url": "http://127.0.0.1:13012", "state_dir": str(runtime / "state")}
    base = [sys.executable, str(runtime / "adapter.py"), "--config", str(runtime / "adapter.json")]
    command = shlex.join(base + ["hook"])
    files["adapter.json"] = (json.dumps(config, indent=2) + "\n").encode()
    hooks = {"hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": command, "timeout": 90}]}], "PreToolUse": [{"matcher": "AskUserQuestion", "hooks": [{"type": "command", "command": command, "timeout": 90}]}], "PostToolUse": [{"matcher": "AskUserQuestion", "hooks": [{"type": "command", "command": command, "timeout": 10}]}]}}
    files["claude-hooks.fragment.json"] = (json.dumps(hooks, indent=2) + "\n").encode()
    files["claude-mcp.fragment.json"] = (json.dumps({"mcpServers": {"aiden-orchestrator": {"command": base[0], "args": base[1:] + ["mcp"]}}}, indent=2) + "\n").encode()
    forward = ["ssh", "-N", "-T", "-oExitOnForwardFailure=yes", "-oServerAliveInterval=15", "-oServerAliveCountMax=3", "-L127.0.0.1:13012:127.0.0.1:3012", "infinitewatts@macmini"]
    files["ssh-forward.argv.json"] = (json.dumps(forward, indent=2) + "\n").encode()
    for name, content in files.items():
        fd = os.open(output / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
    (output / "state").mkdir(mode=0o700)
    receipt = {"mode": "observe", "prepared_only": True, "runtime_dir": str(runtime), "host_session_id": host_session_id, "source_path": str(SOURCE), "reviewed_sha": reviewed_sha, "source_sha256": hashlib.sha256(source).hexdigest(), "files": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}, "enrollment": "Run SessionStart hook in the intended conversation, then start/restart the MCP server. Actual session identity is unverified until enrollment."}
    fd = os.open(output / "receipt.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host-session-id", required=True)
    parser.add_argument("--runtime-dir", help="Render eventual private runtime paths without creating or installing there")
    parser.add_argument("--reviewed-sha")
    args = parser.parse_args()
    try:
        receipt = prepare(args.output_dir, args.host_session_id, args.reviewed_sha, args.runtime_dir)
    except (OSError, ValueError) as exc:
        parser.exit(1, "Preparation failed: " + str(exc) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
