#!/bin/zsh -f
# MacBook (board host): relay voice board requests from the Mac mini every 2 min.
# Appends voice tasks to ~/.codex/board.log and pushes a board summary to the Mini.
# It never dispatches agents or edits board.md. Uninstall:
#   launchctl bootout gui/$(id -u)/com.eric.onju-board-sync
set -euo pipefail
REPO="${0:A:h:h:h}"
LABEL=com.eric.onju-board-sync
PL="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$HOME/Library/Logs/onju-board-sync.log"
BIN="$HOME/.local/libexec/onju-board-sync.py"
# Run a copy, so branch switches in the dev clone never change the live job.
# Install from a reviewed master checkout.
mkdir -p "${BIN:h}"
install -m 0700 "$REPO/scripts/board_sync.py" "$BIN"
cat > "$PL" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>/usr/bin/python3</string>
    <string>$BIN</string>
    <string>--host</string><string>macmini</string>
  </array>
  <key>StartInterval</key><integer>120</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$LOG</string>
</dict></plist>
PLIST
plutil -lint -s "$PL"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PL"
echo "installed $LABEL; log $LOG"
