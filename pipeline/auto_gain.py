"""Automatic level control for far-field microphone audio.

From across a room speech reaches the board clean but quiet (about -40 dBFS).
The speech detector that guards the wake check is level sensitive: at that
level its probability never reaches the onset threshold, so the wake phrase is
never even examined. This raises quiet audio toward a fixed peak level, one
32 ms frame at a time, and never clips loud audio.

The gain for a frame comes from a peak envelope that jumps up at once on a loud
frame and falls back slowly, so gain drops immediately when someone speaks up
close and recovers gradually afterwards. Gain is bounded: silence is never
amplified by more than max_gain.
"""
import math

import numpy as np


class AutoGain:
    def __init__(self, target_peak: float = 24000.0, max_gain: float = 16.0,
                 release_s: float = 1.0, frame_s: float = 0.032):
        if not (0 < target_peak <= 32767 and max_gain >= 1 and release_s > 0 and frame_s > 0):
            raise ValueError("Invalid auto gain settings")
        self.target_peak = float(target_peak)
        self.max_gain = float(max_gain)
        self._decay = math.exp(-frame_s / release_s)
        self._envelope = 0.0
        self.gain = 1.0  # last applied gain, for diagnostics only

    @classmethod
    def from_config(cls, config: dict):
        """Build from audio.auto_gain, or return None when it is off."""
        cfg = (config.get("audio") or {}).get("auto_gain") or {}
        if not cfg.get("enabled"):
            return None
        audio = config["audio"]
        return cls(target_peak=cfg.get("target_peak", 24000.0), max_gain=cfg.get("max_gain", 16.0),
                   release_s=cfg.get("release_s", 1.0),
                   frame_s=audio["chunk_size"] / audio["sample_rate"])

    def process(self, pcm: np.ndarray) -> np.ndarray:
        """Return the frame with gain applied (int16 in, int16 out).

        Gain moves from the previous frame's value to this frame's across the
        frame, so a change never lands as a step in the waveform.
        """
        peak = float(np.max(np.abs(pcm.astype(np.int32)))) if pcm.size else 0.0
        self._envelope = max(peak, self._envelope * self._decay)
        previous = self.gain
        self.gain = max(1.0, min(self.max_gain, self.target_peak / max(self._envelope, 1.0)))
        if previous == 1.0 and self.gain == 1.0:
            return pcm
        ramp = np.linspace(previous, self.gain, pcm.size, endpoint=True, dtype=np.float32)
        return np.clip(pcm.astype(np.float32) * ramp, -32768, 32767).astype(np.int16)
