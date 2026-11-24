import asyncio
import logging
import time

import numpy as np

from pipeline.conversation import ConversationBackend, create_backend
from pipeline.mic_stream import MicStreamMonitor
from pipeline.vad import VAD
from pipeline.security import validate_device

log = logging.getLogger(__name__)


class Device:
    def __init__(self, hostname: str, ip: str, config: dict, conversation: ConversationBackend,
                 voice: str | None = None, ptt: bool = False):
        self.hostname = hostname
        self.ip = ip
        self.config = config
        self.conversation = conversation
        el_cfg = config["tts"].get("elevenlabs", {})
        if voice:
            self.voice = voice
        elif ptt and el_cfg.get("default_voice_ptt"):
            self.voice = el_cfg["default_voice_ptt"]
        else:
            self.voice = el_cfg.get("default_voice", "Emma")
        self.ptt = ptt
        self.vad = None if ptt else VAD(config)
        self.last_user_text: str | None = None
        self.last_response: str | None = None
        self.led_power = 0
        self.led_update_time = 0.0
        self.vad_writer: asyncio.StreamWriter | None = None  # persistent TCP for LED blinks

        # PTT state
        self.ptt_buffer: list = []  # raw PCM frames during PTT
        self.processing = False     # True while ASR/LLM/TTS pipeline is running
        self.interrupted = asyncio.Event()
        self.mic_stream = MicStreamMonitor()
        self.audio_frames = 0
        self.last_audio_at = None
        self.audio_rms = 0.0
        self.peak_audio_rms = 0.0
        self._rate_slot, self._rate_count, self._rate_last = None, 0, (None, 0)
        self.wake_gate = None
        if config.get("wake_word", {}).get("enabled") and not ptt:
            backend = config['wake_word'].get('backend', 'keyword')
            if backend == 'local_asr':
                from pipeline.local_wake import LocalSpeechWakeGate
                self.wake_gate = LocalSpeechWakeGate(config)
            elif backend == 'keyword':
                from pipeline.wakeword import WakeWordGate
                self.wake_gate = WakeWordGate(config['wake_word'])
            else:
                raise ValueError('Unknown wake backend')

    def observe_audio(self, pcm):
        """Keep level/counter metadata only, never microphone samples or text."""
        self.audio_frames += 1
        self.last_audio_at = time.monotonic()
        # Frames per ten-second slot: 312 expected while the mic streams, so a
        # shortfall while idle shows microphone audio lost before it arrived.
        slot = int(self.last_audio_at // 10)
        if slot != self._rate_slot:
            self._rate_last = (self._rate_slot, self._rate_count)
            self._rate_slot, self._rate_count = slot, 0
        self._rate_count += 1
        self.audio_rms = float(np.sqrt(np.mean(np.square(pcm.astype(np.float32)))))
        self.peak_audio_rms = max(self.peak_audio_rms, self.audio_rms)

    def _frames_prior_slot(self, slot):
        # The newest slot stays open until the next frame arrives, so check it too.
        if self._rate_slot == slot:
            return self._rate_count
        return self._rate_last[1] if self._rate_last[0] == slot else 0

    def to_dict(self) -> dict:
        return {
            "hostname": self.hostname,
            "ip": self.ip,
            "voice": self.voice,
            "ptt": self.ptt,
            "listening": {
                "processing": self.processing,
                "queued_utterances": getattr(self, "queued_utterances", 0),
                "wake_enabled": self.wake_gate is not None,
                "wake_active": bool(self.wake_gate and self.wake_gate.active),
                "audio_frames": self.audio_frames,
                "last_audio_age_s": (round(max(0, time.monotonic() - self.last_audio_at), 2)
                                     if self.last_audio_at is not None else None),
                "audio_rms": round(self.audio_rms, 1),
                "peak_audio_rms": round(self.peak_audio_rms, 1),
                "audio_frames_prior_10s": self._frames_prior_slot(int(time.monotonic() // 10) - 1),
                "wake_frames": getattr(self.wake_gate, "frames_seen", None),
                "wake_model_profile": getattr(self.wake_gate, "model_profile", None),
                "wake_detections": getattr(self.wake_gate, "detections", None),
                "wake_checks": getattr(self.wake_gate, "checks", None),
                "wake_failures": getattr(self.wake_gate, "failures", None),
                "wake_max_feed_gap_s": getattr(self.wake_gate, "max_feed_gap_s", None),
                "wake_resets": getattr(self.wake_gate, "reset_counts", None),
                "wake_misses": getattr(self.wake_gate, "miss_counts", None),
                "mic_stream": self.mic_stream.to_dict(),
            },
        }

    def __repr__(self):
        mode = "PTT" if self.ptt else "VOX"
        return f"<Device {self.hostname} {self.ip} {mode}>"


class DeviceManager:
    def __init__(self, config: dict):
        self.config = config
        self.devices: dict[str, Device] = {}

    def create_device(self, hostname: str, ip: str, ptt: bool = False) -> Device:
        validate_device(self.config, hostname, ip)
        device = self.devices.get(hostname)
        if device is None:
            conv = create_backend(self.config, hostname)
            device = Device(hostname, ip, self.config, conversation=conv, ptt=ptt)
            self.devices[hostname] = device
            log.debug(f"New device: {device}")
        elif device.ip != ip:
            device.ip = ip
            log.debug(f"Updated {hostname} IP to {ip}")
        else:
            log.debug(f"Device {hostname} reconnected ({ip})")
        return device

    def get_by_ip(self, ip: str) -> Device | None:
        for d in self.devices.values():
            if d.ip == ip:
                return d
        return None

    def get_most_recent(self) -> Device | None:
        if self.devices:
            return next(reversed(self.devices.values()))
        return None
