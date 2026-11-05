#!/bin/zsh -f
# Installs/refreshes the four onju LaunchAgents on the Mac mini. Run on the Mini:
#   ~/onju-v2-live/deploy/macmini/install.sh
# Re-running is safe: it re-renders the plists and restarts the services.
set -euo pipefail
REPO="${0:A:h:h:h}"
LA="$HOME/Library/LaunchAgents"
PY="$REPO/.venv/bin/python"
mkdir -p "$REPO/logs" "$REPO/data" "$LA"
cp "$REPO/deploy/macmini/config.yaml" "$REPO/pipeline/config.yaml"

plist() {  # label, then program arguments
  local label=$1; shift
  local args="" a
  for a in "$@"; do args+="    <string>${a}</string>"$'\n'; done
  cat > "$LA/$label.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key><array>
$args  </array>
  <key>WorkingDirectory</key><string>$REPO</string>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>HF_HUB_OFFLINE</key><string>1</string>
    <key>HF_HUB_DISABLE_IMPLICIT_TOKEN</key><string>1</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>15</integer>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>$REPO/logs/$label.log</string>
  <key>StandardErrorPath</key><string>$REPO/logs/$label.log</string>
</dict></plist>
PLIST
  plutil -lint -s "$LA/$label.plist"
  launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$LA/$label.plist"
}

plist com.infinitewatts.onju-asr "$PY" -m pipeline.services.asr_server --host 127.0.0.1 --port 8100
plist com.infinitewatts.onju-tts "$PY" -m mlx_audio.server --host 127.0.0.1 --port 8880
plist com.infinitewatts.onju-agent /bin/zsh -f "$REPO/deploy/macmini/run-agent.sh"
plist com.infinitewatts.onju-pipeline "$PY" -m pipeline.main
launchctl list | grep com.infinitewatts.onju- || true
