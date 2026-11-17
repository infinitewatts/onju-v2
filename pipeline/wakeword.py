"""Local wake-phrase gate for 16 kHz microphone frames.

Idle audio goes only to the keyword detector and a bounded memory prebuffer.
Only a detected Hey Aiden phrase (or an explicit follow-up arm) can open a
short command window. The caller resets the gate after accepting one utterance.
"""
from collections import deque
import math
from pathlib import Path
import tempfile

import numpy as np

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512
KEYWORD = "HEY_AIDEN"
KEYWORD_PHONES = "HH EY1 EY1 D AH0 N @HEY_AIDEN\n"
_IDLE_STREAM_S = 60
_IDLE_GAP_S = 1.0


def _positive(value: float, name: str, *, zero: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0 or (not zero and number == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return number


def _load_detector(config: dict):
    root = Path(config["model_dir"]).expanduser()
    files = {
        "tokens": root / "tokens.txt",
        "encoder": root / "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx",
        "decoder": root / "decoder-epoch-13-avg-2-chunk-8-left-64.onnx",
        "joiner": root / "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx",
    }
    for model in files.values():
        if not model.is_file():
            raise FileNotFoundError(f"Wake-word model file is missing: {model}")
    # Optional dependency: importing the pipeline never requires it when wake
    # detection is disabled. Missing support must fail startup, never bypass it.
    import sherpa_onnx

    with tempfile.TemporaryDirectory(prefix="onju-wake-keyword-") as directory:
        keywords = Path(directory) / "keywords.txt"
        keywords.write_text(KEYWORD_PHONES)
        keywords.chmod(0o600)
        return sherpa_onnx.KeywordSpotter(
            **{key: str(value) for key, value in files.items()},
            keywords_file=str(keywords),
            keywords_score=2.0,
            keywords_threshold=_positive(config.get("threshold", 0.25), "threshold"),
            provider="cpu",
            num_threads=1,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
        )


class WakeWordGate:
    """Use a Sherpa KeywordSpotter-compatible detector, injectable for tests.

    ``feed`` refreshes ``active`` and expires windows against the supplied
    monotonic time. ``just_activated`` is true only on the detecting feed.
    Explicit ``arm`` opens a follow-up window without adding synthetic audio.
    """

    def __init__(self, config: dict, detector=None):
        self.command_window_s = _positive(config.get("command_window_s", 30),
                                          "command_window_s")
        self.pre_buffer_s = _positive(config.get("pre_buffer_s", 1.5),
                                     "pre_buffer_s", zero=True)
        # VAD consumes complete 512-sample frames. Round down rather than
        # trimming an older frame into a shape the downstream VAD cannot use.
        self._pre_buffer_samples = (int(self.pre_buffer_s * SAMPLE_RATE)
                                    // FRAME_SAMPLES * FRAME_SAMPLES)
        self._detector = detector if detector is not None else _load_detector(config)
        self._pre_buffer: deque[np.ndarray] = deque()
        self._buffered_samples = 0
        self.reset()

    def _new_stream(self):
        self._stream = self._detector.create_stream()
        self._idle_samples = 0
        # Initial silence lets the keyword model recognize a phrase arriving
        # immediately after arming. It never enters the command prebuffer.
        self._stream.accept_waveform(SAMPLE_RATE, np.zeros(SAMPLE_RATE // 2,
                                                          dtype=np.float32))
        while self._detector.is_ready(self._stream):
            self._detector.decode_stream(self._stream)

    def reset(self):
        self.active = False
        self.just_activated = False
        self._deadline = None
        self._last_feed_at = None
        self._pre_buffer.clear()
        self._buffered_samples = 0
        self._new_stream()

    def arm(self, now: float, window_s: float = 10):
        now = self._time(now)
        window_s = _positive(window_s, "window_s")
        self.reset()
        self.active = True
        self._deadline = now + window_s
        self._last_feed_at = now

    @staticmethod
    def _time(now: float) -> float:
        now = float(now)
        if not math.isfinite(now):
            raise ValueError("Wake-word timestamps must be finite")
        return now

    def _remember(self, frame: np.ndarray):
        if not self._pre_buffer_samples:
            return
        # Copy because a caller may reuse its microphone buffer on the next feed.
        self._pre_buffer.append(frame.copy())
        self._buffered_samples += frame.size
        while self._buffered_samples > self._pre_buffer_samples:
            self._buffered_samples -= self._pre_buffer.popleft().size

    def feed(self, pcm_int16: np.ndarray, now: float) -> list[np.ndarray]:
        self.just_activated = False
        now = self._time(now)
        frame = np.asarray(pcm_int16)
        if frame.shape != (FRAME_SAMPLES,) or frame.dtype != np.int16:
            raise ValueError("Wake-word audio must be 512 samples of one-dimensional int16 PCM")
        if self._last_feed_at is not None and now < self._last_feed_at:
            self.reset()  # a backward clock cannot extend an authorization
        if self.active and now >= self._deadline:
            self.reset()
        if self.active:
            self._last_feed_at = now
            return [frame]
        if (self._last_feed_at is not None and now - self._last_feed_at > _IDLE_GAP_S
                or self._idle_samples >= _IDLE_STREAM_S * SAMPLE_RATE):
            self.reset()
        self._last_feed_at = now
        self._remember(frame)
        self._stream.accept_waveform(SAMPLE_RATE, frame.astype(np.float32) / 32768.0)
        self._idle_samples += frame.size
        detected = False
        while self._detector.is_ready(self._stream):
            self._detector.decode_stream(self._stream)
            if self._detector.get_result(self._stream) == KEYWORD:
                detected = True
                break
        if not detected:
            return []
        self.active = True
        self.just_activated = True
        self._deadline = now + self.command_window_s
        buffered = list(self._pre_buffer)
        self._pre_buffer.clear()
        self._buffered_samples = 0
        # Even a zero/smaller-than-frame prebuffer must pass the detecting frame.
        if self._pre_buffer_samples < frame.size:
            return [frame.copy()]
        return buffered
