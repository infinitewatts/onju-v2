"""Automatic level control for far-field microphone audio.

From across a room speech reaches the board clean but quiet (about -40 dBFS).
The speech detector that guards the wake check is level sensitive: at that
level its probability never reaches the onset threshold, so the wake phrase is
never even examined. This raises quiet audio toward a fixed peak level, one
32 ms frame at a time, without clipping.

Gain follows a peak envelope that rises at once and falls back slowly. Output
runs one frame (32 ms) behind the input so the gain can ease down across the
frame *before* a loud one, instead of stepping or clipping at the first
syllable after silence. Gain is bounded: silence is never amplified by more
than max_gain, and audio already at the target level passes through unchanged.
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
        self._held = None       # the frame waiting to go out
        self._held_peak = 0.0
        self.gain = 1.0         # gain at the end of the last frame sent out

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
        """Take one frame in, return the previous frame with gain applied.

        int16 in, int16 out, same length. The first call returns silence.
        """
        peak_next = float(np.max(np.abs(pcm.astype(np.int32)))) if pcm.size else 0.0
        frame, peak = self._held, self._held_peak
        self._held, self._held_peak = pcm, peak_next
        if frame is None:
            return np.zeros_like(pcm)
        self._envelope = max(peak, self._envelope * self._decay)
        # End this frame at a gain the next frame can also take.
        end = max(1.0, min(self.max_gain, self.target_peak / max(self._envelope, peak_next, 1.0)))
        start = min(self.gain, 32767.0 / peak) if peak else self.gain
        self.gain = end
        if start == 1.0 and end == 1.0:
            return frame
        ramp = np.linspace(start, end, frame.size, endpoint=True, dtype=np.float32)
        return np.clip(frame.astype(np.float32) * ramp, -32768, 32767).astype(np.int16)
