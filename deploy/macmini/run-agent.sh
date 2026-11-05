#!/bin/zsh -f
# Starts the voice agent with its Anthropic key injected from BWS (profile onju-voice).
# bws-run-profile re-runs this script with ONJU_VOICE_ANTHROPIC_API_KEY set; that
# second pass hands the key to Claude Code under the name it expects.
set -euo pipefail
if [[ -n "${ONJU_VOICE_ANTHROPIC_API_KEY:-}" ]]; then
  export ANTHROPIC_API_KEY="$ONJU_VOICE_ANTHROPIC_API_KEY"
  unset ONJU_VOICE_ANTHROPIC_API_KEY
  shift  # drop the --run marker
  exec "$@"
fi
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
  /bin/zsh -f "$0" --run "$REPO/.venv/bin/python" -m pipeline.services.voice_agent \
  --claude-bin "$HOME/.local/bin/claude" --model sonnet \
  --workdir "$STATE" --mcp-config "$STATE/mcp-board.json" \
  --tools "WebSearch,WebFetch" \
  --allowed-tools "WebSearch WebFetch mcp__board__read_tasks mcp__board__add_task"
