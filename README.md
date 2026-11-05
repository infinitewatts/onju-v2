# Onju Voice v2 (OnjuClaw) 🍐🦞

## Personal fork: safe local bring-up

This fork tracks upstream and keeps conversation, voice and tool policy on the
server. Firmware handles audio, touch, transport and maintenance. No server address
is compiled into it: provision a reserved **private IPv4** pin over serial (`c`,
`server 192.168.1.10`, `exit`). The first boot requests this pin; other TCP peers
are rejected, and legacy learned `server_ip` values are ignored. Move to a different
server with the same serial setting; changing LLM/voice/prompts needs no flash.

Before compilation, read the physical flash size with the installed esptool's
`flash-id` command. Set `ONJU_FLASH_SIZE=4M`, `8M` or `16M` accordingly. For example,
`ONJU_FLASH_SIZE=16M ./flash.sh compile` selects two 3 MB OTA app slots on a 16 MB
chip. Changing the partition scheme requires USB. Confirm the actual PCB pin
mapping (`BOARD_V3` is upstream's default), OPI PSRAM and power before upload.

Provision a per-device OTA credential with
`python3 scripts/ota_keychain.py --device AA:BB:CC:DD:EE:FF` on macOS. It creates
or reuses a password in Keychain, verifies readback and writes the setup password
plus OTA SHA-256 verifier to ignored `onjuino/ota_credentials.h` (0600). Both are
sensitive. Never commit them, Wi-Fi settings or build output. With the optional
`setup-qr` extra installed, add `--setup-qr` to write a private join QR SVG in ignored
`data/`; no credential is printed. For a placeholder
compile, copy both header templates; upload rejects placeholders. Wi-Fi generation
still uses the owner-approved Keychain prompt through `flash.sh --regen compile`.

ArduinoOTA is password-gated and available for five minutes after a power cycle,
when idle on private-address station Wi-Fi; mDNS OTA advertisement is off. Use the
matching core 3.3.12 uploader, target UDP 3232 and an application image only. The
router must have **no WAN forwarding** for this service; a private local address
is not a network firewall. Keep the updater on the trusted LAN. No HTTP firmware
updater or cloud update service is installed. Audio is suspended
during an accepted update; PSRAM/allocation/decoder failure leaves a maintenance
path available. Failed authentication must not change the firmware. Password
authentication does not provide signed firmware or transport encryption. Do not
claim automatic health-based rollback: upstream Arduino startup may mark a new
application valid before sketch-level diagnostics. Keep a known USB recovery image.

Home plus four travel networks are stored in bounded NVS profiles. Home is tried
first; unsuccessful saved-network attempts have a combined approximately 30-second
budget. Then `onju-XXXXXX-setup` starts a WPA2-protected AP using the same OTA/setup
password. Join from a phone, open `http://192.168.4.1` if the captive page does not
appear, select/type a 2.4 GHz network and enter its password. The AP closes after
a verified successful save or five minutes; timeouts retain existing profiles.
Serial `P` reboots into a setup window for testing. Runtime disconnect also gates
microphone transport and retries saved networks before setup. A full list replaces
the oldest travel entry and retains home. Updated home passwords survive OTA builds.
HTTP is bound to the AP address; canonical Host/Origin checks and a per-session
token protect saves. No firmware-update route exists in this portal. Network
passwords and setup credentials remain recoverable from unencrypted device flash;
physical possession of the device is a trust boundary.

Joining travel Wi-Fi does **not** make the home Mac mini reachable. The home server
pin is inactive on travel profiles, preventing an unrelated machine at the same
private IP from receiving microphone audio. Later options require explicit work:
a paired portable server on the travel LAN, a VPN travel router plus authenticated
per-profile server selection, or an encrypted authenticated relay. A phone hotspot
alone does not extend the home LAN. Do not forward UDP/TCP audio or OTA to the WAN.
The setup page handles personal Wi-Fi credentials, not hotel captive-login pages
or enterprise authentication.

Reserve device/server IPs in DHCP. The device re-announces after a minute without
an accepted TCP connection. Add exact hostname/IP pairs to the server config:

```yaml
network:
  allowed_devices:
    onju-A1B2C3: "192.168.1.50"
```

An empty allowlist rejects all devices; manual/control registration uses the same
policy. Names must match `onju-[0-9A-F]{6}`, including conversation filenames.
These IP controls do not prevent LAN ARP/address spoofing; use an isolated trusted
network before attaching sensitive tools. Playback volume is capped at 16 in the
server protocol. The physical mute switch is enabled for the assembled V3 unit;
verify its state and microphone suppression on the actual device. The watchdog
gap documented in the pre-flash review remains a desk-test limitation.

Use `.venv` with `.[asr,tts-local,mic]` and Homebrew opus, portaudio and ffmpeg.
Control HTTP is bound to 127.0.0.1; ASR defaults to loopback. Start mlx-audio with
`--host 127.0.0.1 --port 8880`, and keep the LLM/gateway on loopback. For keyless
transport tests, run `.venv/bin/python -m uvicorn local_llm_stub:app --app-dir scripts
--host 127.0.0.1 --port 8080`. It always reports that no task was created. Disable
the cloud stall classifier. Qwen3-TTS CustomVoice needs local `voice: ryan` and
`lang_code: English`; configure ASR 8100, TTS 8880 and the stub 8080 locally.

The emulator `tests/test_client.py` now advertises `onju-000001` and listens only
on loopback. For that test, explicitly allow `onju-000001: "127.0.0.1"`; do not
use that mapping for real hardware. Run security regressions with
`.venv/bin/python -m unittest discover -s tests -p test_security.py -v`.
Additional regressions: `test_flash.py`, `test_wifi_profiles.py` (native C++ policy
tests with address/undefined-behavior sanitizers) and `test_mic_receiver.py`.
Register the emulator explicitly with the pipeline's
`--device onju-000001=127.0.0.1` flag (add `:ptt` for a file-fed PTT test).
Its upstream multicast sender uses the LAN interface, so multicast alone does not
register a loopback-only emulator under the strict IP policy.

Enable multiple "Google Home" speakers to connect to your Mac Mini for talking to your agent(s).

This repo consists of:
* An async server pipeline handling ASR -> TTS from multiple devices using any LLM or agent platforms like OpenClaw 🦞
* Hardware designs for a drop-in replacement PCB to the original Google Nest Mini (2nd gen), using the ESP32-S3 for audio processing and WiFi connectivity ([order link](https://www.pcbway.com/project/shareproject/Onju_Voice_d33625a1.html))

> This is an upgraded version of [onju-voice](https://github.com/justLV/onju-voice) as DEMO'd [here](https://x.com/justLV/status/1681377298308820992?s=20)!.

<img src="images/header_white.jpg" width="960">

## What's new in v2

* **Agentic backend** 🦞 -- delegate conversation history, session management, and tool execution to an [OpenClaw](https://github.com/openclaw) gateway for centralized, multi-device orchestration
* **Opus compression** -- 14-16x downstream compression (server to speaker) for better audio quality over WiFi
* **Streaming-ready architecture** -- designed for sentence-level TTS streaming and agentic tool-calling loops
* **Modular async pipeline** -- replaced the monolithic server with a pluggable architecture for ASR, LLM, and TTS backends etc.
* **Any LLM** -- works with any OpenAI-compatible API (Ollama, mlx_lm, Gemini, OpenRouter, Claude, etc.)
* **Pluggable TTS** -- ElevenLabs (recommended) or local via [mlx-audio](https://github.com/lucasnewman/mlx-audio) for fully offline operation
* **Silero VAD** -- server-side voice activity detection with configurable thresholds, replacing webrtcvad
* **VAD-aware interruption** -- tap to interrupt playback and start speaking immediately
* **M5 Echo support** -- get started with a [$13 dev kit](https://shop.m5stack.com/products/atom-echo-smart-speaker-dev-kit) instead of ordering a custom PCB ([battery base](https://shop.m5stack.com/products/atomic-battery-base-200mah))
* **One-command flashing** -- `./flash.sh` handles compilation, WiFi credential generation (from macOS Keychain), and upload. No Arduino IDE or manual configuration required

## Supported devices

| | Onjuino (custom PCB) | M5Stack ATOM Echo |
|---|---|---|
| **Board** | ESP32-S3 | ESP32-PICO-D4 |
| **Interaction** | Capacitive touch: tap to start (uses VAD to end) | Physical button: hold to talk |
| **Mic** | I2S (INMP441) | PDM (SPM1423) |
| **Speaker** | MAX98357A, 6 NeoPixel LEDs | NS4168, 1 SK6812 LED |
| **PSRAM** | Yes (2MB playback buffer) | No (smaller buffers) |
| **Audio upstream** | mu-law 16kHz UDP (16 KB/s) | mu-law 16kHz UDP (16 KB/s) |
| **Audio downstream** | Opus 16kHz TCP (~1.5 KB/s) | Opus 16kHz TCP (~1.5 KB/s) |

Both targets use the same network protocol and connect to the same server. See the [M5 Echo README](m5_echo/README.md) for hardware-specific details.

## Architecture

```
                ESP32 Device                              Server Pipeline
  ┌──────────────────────────────┐       ┌──────────────────────────────────────┐
  │  Mic > I2S RX > mu-law =======UDP 3000===> mu-law decode > VAD > ASR        │
  │                              │       │                                      │
  │  Speaker < I2S TX < Opus <===TCP 3001<=== Opus encode < TTS < LLM           │
  └──────────────────────────────┘       └──────────────────────────────────────┘
```

**Why mu-law upstream:** Stateless sample-by-sample encoding (~1% CPU), zero buffering latency. ASR models handle the quality fine.

**Why Opus downstream:** Human ears need better quality than ASR, and Opus decoding is easier for an ESP32. Opus gives 14-16x compression vs mu-law's 2x, and TCP ensures reliable ordered delivery for the stateful codec.

### Device discovery

1. ESP32 boots and joins WiFi
2. Sends multicast announcement to `239.0.0.1:12345` with hostname, git hash, and PTT flag
3. Server discovers device and connects to its TCP server on port 3001
4. ESP32 learns server IP from the TCP connection and starts sending mic audio via UDP

### TCP command protocol

All commands use a 6-byte header. The server initiates TCP connections to the ESP32.

| Byte 0 | Command | Payload |
|---|---|---|
| `0xAA` | Audio playback | mic_timeout(2B), volume, LED fade, compression type, then length-prefixed Opus frames |
| `0xBB` | Set LEDs | LED bitmask, RGB color |
| `0xCC` | LED blink (VAD) | intensity, RGB color, fade rate |
| `0xDD` | Mic timeout | timeout in seconds (2B) |

A zero-length Opus frame (`0x00 0x00`) signals end of speech.

### FreeRTOS task layout

| Core | Task | Purpose |
|---|---|---|
| Core 0 | Arduino loop | TCP server, touch/mute input, UART debug |
| Core 1 | `micTask` | I2S read, mu-law encode, UDP send |
| Core 1 | `opusDecodeTask` | TCP read, Opus decode, I2S write (created per playback) |
| Core 1 | `updateLedTask` | 40Hz LED refresh with gamma-corrected fade |

### Conversation backends

The pipeline supports two conversation backends, selectable via `config.yaml`:

**Conversational** (`conversation.backend: "conversational"`): Plain chat-completions backend. Manages conversation history client-side with per-device JSON persistence and sends the full message history on each LLM request. Works with any OpenAI-compatible endpoint (OpenRouter, Gemini, Ollama, mlx_lm.server, etc.). Good for simple voice chat with no tool use.

**Agentic** (`conversation.backend: "agentic"`): Delegates session management and tool execution to a remote agent gateway like [OpenClaw](https://github.com/openclaw). Only sends the latest user message — the gateway tracks history server-side using the device ID as the session key, and runs its own tool loop (web search, file access, multi-step work). Set `OPENCLAW_GATEWAY_TOKEN` in your environment and point `base_url` at your gateway.

### Setting up OpenClaw

If you have [OpenClaw](https://github.com/openclaw) installed, a setup script is included:

```bash
./setup_openclaw.sh
```

This will:
1. Enable the chat completions HTTP endpoint on the gateway
2. Append a voice mode prompt to `~/.openclaw/workspace/AGENTS.md` (tells the agent to respond in concise, speech-friendly prose when the message channel is `onju-voice`)
3. Restart the gateway

Then set `conversation.backend: "agentic"` in `pipeline/config.yaml` and ensure `OPENCLAW_GATEWAY_TOKEN` is set in your environment.

### Streaming and stalls w/ OpenClaw

Agentic requests can take 5-60+ seconds while the gateway runs tools, so the pipeline is built to make that wait feel responsive.

**Sentence-level streaming.** The agent's SSE response is consumed delta by delta. A splitter (`pipeline/conversation/__init__.py:sentence_chunks`) buffers text until it hits a sentence boundary, then hands the sentence to TTS and pushes the audio to the device. Any narration the agent emits between tool calls gets spoken aloud as it arrives, while the next tool runs in the background. Intermediate sends use `mic_timeout=0` so the mic only reopens after the final chunk.

**Contextual stall phrases.** Before the main agent call, the pipeline fires a fast classifier that decides whether the question needs a brief spoken acknowledgment. Conversational questions return `NONE` and get no stall. Tool-needing questions get a short personality-matched phrase that plays within about a second while the agent works. The stall text is then injected back into the agent's user message as a parenthetical continuity note so it doesn't repeat itself. Configure in `conversation.stall`.

**The first-turn caveat with OpenClaw.** OpenClaw's OpenAI-compatible endpoint buffers all content from the first agent turn until the first round of tool execution completes. If the model generates an opening sentence and then calls a tool, that sentence is held server-side until the tool finishes. Narration between *subsequent* tool rounds streams fine. This is why the stall classifier exists: it gives the user a fast spoken acknowledgment that bypasses the gateway's first-turn buffering. See `pipeline/conversation/stall.py`.

## Installation

### Server

```bash
# Clone and set up Python environment
git clone https://github.com/justLV/onju-v2.git
cd onju-v2
uv venv && source .venv/bin/activate
uv pip install -e .

# macOS: install system libraries for Opus encoding
brew install opus portaudio

# Configure
cp pipeline/config.yaml.example pipeline/config.yaml
# Edit config.yaml with your API keys and preferences
```

**ASR** -- an embedded [parakeet-mlx](https://github.com/senstella/parakeet-mlx) server is included (Apple Silicon):
```bash
uv pip install -e ".[asr]"
python -m pipeline.services.asr_server  # runs on port 8100
```
Or point `asr.url` in config.yaml at any Whisper-compatible endpoint.

**LLM** -- any OpenAI-compatible server:
```bash
# Local (mlx_lm on Apple Silicon)
mlx_lm.server --model unsloth/gemma-4-E4B-it-UD-MLX-4bit --port 8080

# Local (Ollama)
ollama run gemma4:e4b

# Cloud -- just set base_url and api_key in config.yaml (default: Haiku via OpenRouter)
```

**TTS** -- [ElevenLabs](https://elevenlabs.io) is the default (set your API key in config.yaml). For fully offline TTS, you can use [mlx-audio](https://github.com/lucasnewman/mlx-audio) (`uv pip install -e ".[tts-local]"`, then set `tts.backend: "local"` in config.yaml - I don't think this is the best quality, just including as reference for a local TTS!).

**Run:**
```bash
source .venv/bin/activate
python -m pipeline.main
```

### Firmware

Both targets can be compiled and flashed from the command line:

```bash
# Flash onjuino (default)
./flash.sh

# Flash M5 Echo
./flash.sh m5_echo

# Compile only (no device needed)
./flash.sh compile

# Regenerate WiFi credentials from macOS Keychain, defaults to manual entry
./flash.sh --regen
```

Requires `arduino-cli`:
```bash
# macOS
brew install arduino-cli
arduino-cli core install esp32:esp32
```
The flash script auto-installs required libraries (Adafruit NeoPixel, esp32_opus).

WiFi credentials are generated from your macOS Keychain on first flash, or you can edit the `credentials.h.template` files manually.

For Arduino IDE users: select **ESP32S3 Dev Module** (onjuino) or **ESP32 Dev Module** (M5 Echo), enable **USB CDC on Boot** and **OPI PSRAM** (onjuino only), then build and upload.

### Hardware

<p float="left">
  <img src="images/copper.png" width="48%" />
  <img src="images/render.png" width="48%" />
</p>

[Preview schematics & PCB](https://365.altium.com/files/77C755F4-7195-4B29-93AA-0C10A2471AC3) | [Order from PCBWay](https://www.pcbway.com/project/shareproject/Onju_Voice_d33625a1.html) | Altium source files and schematics in `hardware/`.

If you don't have a custom PCB, you can use the [M5Stack ATOM Echo](https://shop.m5stack.com/products/atom-echo-smart-speaker-dev-kit). I'd recommend adding a Battery ([Biscuit](https://www.youtube.com/watch?v=OMg3epr53Ns)) Base ([link](https://shop.m5stack.com/products/atomic-battery-base-200mah))

## Configuration reference

See [`pipeline/config.yaml.example`](pipeline/config.yaml.example) for all options. Key sections:

| Section | What it controls |
|---|---|
| `asr` | Speech-to-text service URL |
| `conversation.backend` | `"conversational"` (plain chat) or `"agentic"` (OpenClaw, tools) |
| `conversation.conversational` | LLM endpoint, model, system prompt, message history |
| `conversation.agentic` | OpenClaw gateway URL, auth token, message channel |
| `conversation.stall` | Fast classifier that decides if the agentic backend needs a brief spoken stall |
| `tts` | TTS backend (`"elevenlabs"` or `"local"`), voice settings |
| `vad` | Voice activity detection thresholds and timing |
| `network` | UDP/TCP/multicast ports |
| `device` | Volume, mic timeout, LED settings, greeting audio |

### Environment variables

| Variable | Used by |
|---|---|
| `OPENROUTER_API_KEY` | Conversational backend via OpenRouter |
| `GEMINI_API_KEY` | Conversational backend via Gemini, and the stall classifier |
| `ANTHROPIC_API_KEY` | Conversational backend via Anthropic API directly |
| `OPENCLAW_GATEWAY_TOKEN` | Agentic (OpenClaw) backend |

## Testing

All scripts are in `tests/` and should be run from the project root.

```bash
# Emulate an ESP32 device (no hardware needed)
python tests/test_client.py                  # localhost
python tests/test_client.py 192.168.1.50     # remote server

# Test speaker output (send audio file to device w/ TCP and Opus encoding)
python tests/test_speaker.py <device-ip>

# Test mic input (receive and record UDP audio)
python tests/test_mic.py --duration 10

# Benchmark the stall classifier against the configured Gemini endpoint
python tests/test_stall.py

# Inspect raw SSE chunk shapes from the configured agentic gateway
python tests/test_stream.py
python tests/test_stream.py "your prompt here"

# Serial monitor (auto-detects USB port)
python serial_monitor.py
```

## UART debug commands

Both firmware targets support serial commands at 115200 baud:

| Key | Action |
|---|---|
| `r` | Reboot |
| `M` | Enable mic for 10 minutes |
| `m` | Disable mic |
| `A` | Re-send multicast announcement |
| `c` | Enter config mode (WiFi, server, volume) |
| `W`/`w` | LED test fast/slow (onjuino) |
| `P` | Play 440Hz test tone (M5 Echo) |

## License

MIT
