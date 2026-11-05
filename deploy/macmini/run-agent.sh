#!/bin/zsh -f
# Starts the voice agent with its Anthropic key injected from BWS (profile onju-voice).
set -euo pipefail
REPO="${0:A:h:h:h}"
STATE="$HOME/.local/state/onju-voice-agent"
mkdir -p -m 700 "$STATE"
cat > "$STATE/mcp-board.json" <<JSON
{"mcpServers": {"board": {"command": "$REPO/.venv/bin/python",
  "args": ["$REPO/pipeline/services/board_mcp.py", "--state-dir", "$STATE", "--device", "onju-8DAB80"]}}}
JSON
chmod 600 "$STATE/mcp-board.json"
cd "$REPO"
exec "$HOME/secrets-ops/bin/bws-run-profile" onju-voice -- \
  "$REPO/.venv/bin/python" -m pipeline.services.voice_agent \
  --claude-bin "$HOME/.local/bin/claude" --model sonnet \
  --workdir "$STATE" --mcp-config "$STATE/mcp-board.json" \
  --tools "WebSearch,WebFetch" \
  --allowed-tools "WebSearch WebFetch mcp__board__read_tasks mcp__board__add_task"
