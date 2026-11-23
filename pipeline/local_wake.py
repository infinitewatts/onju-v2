"""Bounded, local-only speech recognition for wake admission.

Idle transcripts are examined in memory and never logged or sent to an agent.
The normal command pipeline receives audio only after an exact wake prefix.
"""
import asyncio
import copy
import re
from urllib.parse import urlsplit

import httpx
import numpy as np

from pipeline.audio import pcm_to_wav
from pipeline.vad import VAD

# Spellings the recognizer uses for the same two sounds; no other names.
_GREETING = r"(?:hey|hay)"
_NAME = r"(?:aiden|aidan|ayden|aydan|aidyn|aydin|aden)"
_PREFIX = re.compile(rf"^\s*{_GREETING}[\s,.!?:;-]+{_NAME}\b[\s,.!?:;-]*", re.IGNORECASE)
_GREETING_FIRST = re.compile(rf"^\s*{_GREETING}\b[\s,.!?:;-]*(.*)$", re.IGNORECASE | re.DOTALL)
_NAME_ANYWHERE = re.compile(rf"\b{_NAME}\b", re.IGNORECASE)
MISS_REASONS = ('no_text', 'greeting_only', 'greeting_other_word', 'name_not_first', 'other_speech')
_RATE = 16000
_FRAME = 512
_MAX_FRAMES = 125  # Four seconds, including prebuffer; never an unbounded queue.


def wake_match(text):
    if not isinstance(text, str):
        return False, False
    match = _PREFIX.match(text)
    return (True, bool(text[match.end():].strip())) if match else (False, False)


def miss_reason(text):
    """Why a rejected segment missed, as a fixed label. Never returns the words."""
    if not isinstance(text, str) or not text.strip():
        return 'no_text'
    greeting = _GREETING_FIRST.match(text)
    if greeting:
        return 'greeting_other_word' if greeting.group(1).strip() else 'greeting_only'
    return 'name_not_first' if _NAME_ANYWHERE.search(text) else 'other_speech'


class LocalSpeechWakeGate:
    model_profile = "local_asr"

    def __init__(self, config, *, vad=None, recognize=None):
        url = config['asr']['url'].rstrip('/')
        parsed = urlsplit(url)
        if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', '::1')
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path):
            raise ValueError('Local wake ASR requires a literal loopback HTTP origin')
        self.url = url + '/transcribe'
        idle_config = copy.deepcopy(config)
        idle_config['vad']['silence_time'] = .35
        idle_config['vad'].pop('short_speech_pause_s', None)
        self.vad = vad if vad is not None else VAD(idle_config)
        self._recognize = recognize or self._request
        self._task = None
        self._retired = set()
        self._candidate = None
        self._pending = []
        self._last_feed = None
        self.active = False
        self.just_activated = False
        self._follow_up_window = 0
        self._deadline = 0
        self._command_started = False
        self.frames_seen = self.detections = self.checks = self.failures = 0
        self.max_feed_gap_s = 0.0
        self.reset_counts = {}
        self.miss_counts = dict.fromkeys(MISS_REASONS, 0)

    async def _request(self, audio):
        # No environment proxy, redirects, credentials, or transcript logging.
        async with httpx.AsyncClient(timeout=4, trust_env=False, follow_redirects=False) as client:
            response = await client.post(self.url, files={
                'audio': ('wake.wav', pcm_to_wav(audio), 'audio/wav')})
            response.raise_for_status()
            text = response.json().get('text')
            result = wake_match(text)
            if not result[0]:
                # Label only: the rejected words are dropped here, uncounted and unlogged.
                self.miss_counts[miss_reason(text)] += 1
            return result

    async def _check(self, audio):
        try:
            return await asyncio.wait_for(self._recognize(audio), timeout=4)
        except Exception:
            self.failures += 1
            return False, False

    def _reaped(self, task):
        self._retired.discard(task)
        if not task.cancelled():
            task.exception()

    def reset(self, reason='explicit'):
        self.reset_counts[reason] = self.reset_counts.get(reason, 0) + 1
        if self._task is not None:
            self._task.cancel()
            self._retired.add(self._task)
            self._task.add_done_callback(self._reaped)
            self._task = None
        self._candidate = None
        self._pending.clear()
        self.vad.reset()
        self.vad.pre_buffer.clear()
        self._follow_up_window = 0
        self.active = self.just_activated = False
        self._command_started = False
        self._last_feed = None

    async def aclose(self):
        self.reset()
        if self._retired:
            await asyncio.gather(*self._retired, return_exceptions=True)

    def arm(self, now, *, window_s=10):
        self.reset()
        self.active = True
        self._deadline = now + min(window_s, 10)
        self._last_feed = now

    def await_follow_up(self, now, window_s, playback_remaining_s):
        """Start the answer timer on resumed mic audio, with bounded playback grace."""
        self.reset()
        self.active = True
        self._follow_up_window = min(window_s, 30)
        self._deadline = now + min(playback_remaining_s, 60) + self._follow_up_window

    def begin_command(self, now):
        if self.active and not self._command_started:
            self._command_started = True
            self._deadline = now + 30

    def feed(self, pcm, now):
        if pcm.shape != (_FRAME,) or pcm.dtype != np.int16 or not np.isfinite(now):
            raise ValueError('Expected finite time and 512 int16 samples')
        self.just_activated = False
        self.frames_seen += 1
        if self._last_feed is not None:
            gap = now - self._last_feed
            self.max_feed_gap_s = max(self.max_feed_gap_s, gap)
            if gap < 0 or gap > 1:
                self.reset('gap')
        if self.active and now >= self._deadline:
            self.reset('expiry')
        if self._follow_up_window:
            self._deadline = min(self._deadline, now + self._follow_up_window)
            self._follow_up_window = 0
        self._last_feed = now
        if self.active:
            return [pcm]
        if self._retired:
            # Do not start another request until cancellation has been reaped.
            return []
        if self._task is not None:
            self._pending.append(pcm.copy())
            if len(self._pending) > _MAX_FRAMES:
                self.reset('overflow')
                return []
            if not self._task.done():
                return []
            matched, has_command = self._task.result()
            self._task = None
            frames = self._pending
            self._pending = []
            if matched:
                if has_command:
                    frames = list(self._candidate.reshape(-1, _FRAME)) + frames
                self.arm(now)
                self.detections += 1
                self.just_activated = True
                return frames
            self._candidate = None
            self.vad.reset()
            self.vad.pre_buffer.clear()
            return []
        utterance = self.vad.process_frame(pcm)
        if utterance is None and len(self.vad.buffer) >= _MAX_FRAMES:
            utterance = np.concatenate(self.vad.buffer[:_MAX_FRAMES])
            self.vad.reset()
            self.vad.pre_buffer.clear()
        if utterance is not None:
            self._candidate = utterance.copy()
            self.checks += 1
            self._task = asyncio.create_task(self._check(self._candidate))
        return []
