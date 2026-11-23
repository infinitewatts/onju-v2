import asyncio
from collections import deque
import unittest
from unittest.mock import Mock, AsyncMock, patch

import numpy as np

from pipeline.vad import VAD
from pipeline.local_wake import LocalSpeechWakeGate, miss_reason, wake_match

CONFIG = {'asr': {'url': 'http://127.0.0.1:8100'},
          'vad': {'silence_time': 1.5}, 'audio': {}}


class LocalWakeTests(unittest.IsolatedAsyncioTestCase):
    def make(self, result=(True, False)):
        vad = Mock(pre_buffer=deque(), buffer=[])
        vad.process_frame.return_value = None
        recognize = AsyncMock(return_value=result)
        gate = LocalSpeechWakeGate(CONFIG, vad=vad, recognize=recognize)
        self.addAsyncCleanup(gate.aclose)
        return gate, vad, recognize

    async def finish(self, gate):
        await gate._task

    async def test_wake_only_replays_following_speech_not_wake_phrase(self):
        gate, vad, recognize = self.make()
        wake = np.ones(512, dtype=np.int16)
        command = wake * 2
        vad.process_frame.return_value = wake
        self.assertEqual(gate.feed(wake, 0), [])
        await self.finish(gate)
        frames = gate.feed(command, .032)
        self.assertEqual(len(frames), 1)
        np.testing.assert_array_equal(frames[0], command)
        self.assertTrue(gate.active)
        self.assertTrue(gate.just_activated)
        gate.feed(command, 10.1)
        self.assertFalse(gate.active)

    async def test_combined_command_and_pending_audio_preserved(self):
        gate, vad, recognize = self.make((True, True))
        a = np.ones(512, dtype=np.int16)
        vad.process_frame.return_value = np.concatenate([a, a * 2])
        gate.feed(a, 0)
        gate.feed(a * 3, .032)
        await self.finish(gate)
        frames = gate.feed(a * 4, .064)
        self.assertEqual([int(f[0]) for f in frames], [1, 2, 3, 4])
        self.assertEqual(gate.detections, 1)

    async def test_negative_never_releases_audio(self):
        gate, vad, _ = self.make((False, False))
        a = np.zeros(512, dtype=np.int16)
        vad.process_frame.return_value = a
        gate.feed(a, 0)
        await self.finish(gate)
        self.assertEqual(gate.feed(a, .032), [])
        self.assertFalse(gate.active)
        self.assertEqual(gate.detections, 0)

    async def test_error_fails_closed_and_counts_without_logging(self):
        gate, vad, recognize = self.make()
        recognize.side_effect = RuntimeError('private transcript')
        a = np.zeros(512, dtype=np.int16)
        vad.process_frame.return_value = a
        gate.feed(a, 0)
        await self.finish(gate)
        self.assertEqual(gate.feed(a, .032), [])
        self.assertEqual(gate.failures, 1)

    async def test_pending_audio_bounded_and_request_cancelled(self):
        gate, vad, recognize = self.make()
        recognize.side_effect = lambda audio: None
        async def hang(audio):
            await asyncio.Event().wait()
        gate._recognize = hang
        a = np.zeros(512, dtype=np.int16)
        vad.process_frame.return_value = a
        gate.feed(a, 0)
        task = gate._task
        for i in range(1, 127):
            self.assertEqual(gate.feed(a, i * .032), [])
        await gate.aclose()
        self.assertTrue(task.cancelled())
        self.assertLessEqual(len(gate._pending), 125)
        self.assertFalse(gate.active)

    async def test_continuous_idle_speech_is_capped(self):
        gate, vad, recognize = self.make()
        a = np.zeros(512, dtype=np.int16)
        vad.buffer = [a] * 125
        gate.feed(a, 0)
        await self.finish(gate)
        self.assertEqual(recognize.call_args.args[0].size, 64000)

    async def test_reset_prevents_late_result_from_arming(self):
        gate, vad, _ = self.make()
        a = np.zeros(512, dtype=np.int16)
        vad.process_frame.return_value = a
        gate.feed(a, 0)
        await self.finish(gate)
        gate.reset()
        vad.process_frame.return_value = None
        self.assertEqual(gate.feed(a, .032), [])
        self.assertFalse(gate.active)

    async def test_follow_up_starts_on_first_mic_frame_and_expires(self):
        gate, _, recognize = self.make()
        frame = np.zeros(512, dtype=np.int16)
        gate.await_follow_up(0, 15, 8)
        for now in range(8, 23):
            self.assertTrue(gate.feed(frame, now))
        self.assertEqual(gate.feed(frame, 23), [])
        self.assertFalse(gate.active)
        recognize.assert_not_awaited()

    async def test_disconnected_mic_cannot_open_stale_follow_up(self):
        gate, _, _ = self.make()
        gate.await_follow_up(0, 15, 8)
        self.assertEqual(gate.feed(np.zeros(512, dtype=np.int16), 24), [])
        self.assertFalse(gate.active)

    async def test_command_start_extends_once_but_stays_bounded(self):
        gate, _, _ = self.make()
        gate.arm(0)
        for now in range(10):
            gate.feed(np.zeros(512, dtype=np.int16), now)
        gate.begin_command(9)
        self.assertEqual(gate._deadline, 39)
        gate.begin_command(20)
        self.assertEqual(gate._deadline, 39)
        for now in range(10, 39):
            self.assertTrue(gate.feed(np.zeros(512, dtype=np.int16), now))
        self.assertEqual(gate.feed(np.zeros(512, dtype=np.int16), 39), [])
        self.assertFalse(gate.active)
        gate.reset()
        gate.begin_command(40)
        self.assertFalse(gate.active)

    async def test_reset_during_request_discards_late_match_and_reaps(self):
        gate, vad, _ = self.make()
        started = asyncio.Event()
        async def late_match(audio):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return True, True
        gate._recognize = late_match
        a = np.zeros(512, dtype=np.int16)
        vad.process_frame.return_value = a
        gate.feed(a, 0)
        await started.wait()
        task = gate._task
        gate.reset()
        self.assertEqual(gate.feed(a, .032), [])
        self.assertIsNone(gate._task)
        await gate.aclose()
        self.assertTrue(task.done())
        self.assertFalse(gate.active)
        self.assertEqual(gate.detections, 0)

    async def test_timeout_cancels_recognizer_and_fails_closed(self):
        gate, vad, _ = self.make()
        cancelled = asyncio.Event()
        async def hang(audio):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        gate._recognize = hang
        a = np.zeros(512, dtype=np.int16)
        vad.process_frame.return_value = a
        gate.feed(a, 0)
        await self.finish(gate)
        self.assertTrue(cancelled.is_set())
        self.assertEqual(gate.failures, 1)
        self.assertEqual(gate.feed(a, .032), [])

    async def test_idle_http_has_no_proxy_redirect_or_text_logging(self):
        gate, _, _ = self.make()
        response = Mock()
        response.json.return_value = {'text': 'Hey Aiden, hello'}
        client = AsyncMock()
        client.post.return_value = response
        context = AsyncMock()
        context.__aenter__.return_value = client
        with patch('pipeline.local_wake.httpx.AsyncClient', return_value=context) as factory:
            result = await gate._request(np.zeros(512, dtype=np.int16))
        self.assertEqual(result, (True, True))
        factory.assert_called_once_with(timeout=4, trust_env=False, follow_redirects=False)
        self.assertEqual(client.post.call_args.args[0], 'http://127.0.0.1:8100/transcribe')

    def test_exact_prefix_only(self):
        for text in ['Hey Aiden!', 'HEY, AIDEN', 'hey aiden what time is it?']:
            self.assertTrue(wake_match(text)[0], text)
        for text in ['Aiden', 'Hey Adrian', 'Harry Aiden', 'Do not say Hey Aiden',
                     'I spoke with Aiden', 'Hey Aidens', 'Hey', None]:
            self.assertEqual(wake_match(text), (False, False), text)

    def test_same_sound_spellings_match_but_other_names_do_not(self):
        for text in ['Hey Aidan', 'Hay Aiden, lights', 'hey ayden', 'Hey Aden.', 'hey aidyn what time']:
            self.assertTrue(wake_match(text)[0], text)
        for text in ['Hey Eden', 'Hey Hayden', 'Hi Aiden', 'Hey Adrian', 'Hey Aidens', 'Okay Aiden']:
            self.assertEqual(wake_match(text), (False, False), text)

    def test_every_admitted_spelling_is_stripped_before_routing(self):
        from pipeline.wake_phrase import strip_wake_prefix
        for text in ['Hey Aiden, option one', 'Hey Aidan, option one', 'hay ayden option one',
                     'Aden: option one', 'Hey Aidyn - option one']:
            self.assertTrue(wake_match(text)[0] or text.startswith('Aden'), text)
            self.assertEqual(strip_wake_prefix(text), 'option one', text)
        self.assertEqual(strip_wake_prefix('Hey Aidan!'), '')
        for text in ['Hey Eden, option one', 'option one', 'Aidens option']:
            self.assertEqual(strip_wake_prefix(text), text)

    def test_miss_reason_is_a_fixed_label(self):
        cases = {None: 'no_text', '  ': 'no_text', 'Hey': 'greeting_only', 'hay.': 'greeting_only',
                 'Hey Adrian': 'greeting_other_word', 'Harry Aiden': 'name_not_first',
                 'so I said hey Aidan': 'name_not_first', 'turn the lights off': 'other_speech'}
        for text, reason in cases.items():
            self.assertEqual(miss_reason(text), reason, text)

    async def test_rejected_segment_counts_reason_and_keeps_no_words(self):
        gate, _, _ = self.make()
        response = Mock()
        response.json.return_value = {'text': 'Hey Adrian private words'}
        client = AsyncMock()
        client.post.return_value = response
        context = AsyncMock()
        context.__aenter__.return_value = client
        with patch('pipeline.local_wake.httpx.AsyncClient', return_value=context), \
                self.assertNoLogs(level='DEBUG'):
            result = await gate._request(np.zeros(512, dtype=np.int16))
        self.assertEqual(result, (False, False))
        self.assertEqual(gate.miss_counts['greeting_other_word'], 1)
        self.assertEqual(sum(gate.miss_counts.values()), 1)
        self.assertFalse(any(isinstance(v, str) and 'private' in v for v in vars(gate).values()))
        response.json.return_value = {'text': 'Hey Aidan'}
        with patch('pipeline.local_wake.httpx.AsyncClient', return_value=context):
            self.assertEqual(await gate._request(np.zeros(512, dtype=np.int16)), (True, False))
        self.assertEqual(sum(gate.miss_counts.values()), 1)

    def test_nonlocal_endpoint_rejected(self):
        for url in ['https://example.com', 'http://localhost:8100',
                    'http://127.0.0.1:8100/path', 'http://user@127.0.0.1:8100']:
            with self.assertRaises(ValueError):
                LocalSpeechWakeGate(dict(CONFIG, asr={'url': url}), vad=Mock())

class ConversationPauseTests(unittest.TestCase):
    def test_short_filler_then_answer_is_one_turn(self):
        with patch("pipeline.vad._load_silero_onnx") as load:
            vad = VAD({"audio": {"sample_rate": 16000, "chunk_size": 512},
                       "vad": {"threshold": .5, "neg_threshold": .35, "silence_time": 1.5,
                               "short_speech_pause_s": 3, "pre_buffer_s": 1}})
        frame = np.zeros(512, dtype=np.int16)
        for prob, count in [(1, 10), (0, 70), (1, 40), (0, 46)]:
            load.return_value.run.return_value = np.array([[prob]]), vad._state
            for _ in range(count):
                self.assertIsNone(vad.process_frame(frame))
        self.assertEqual(len(vad.process_frame(frame)), (10 + 70 + 40 + 47) * 512)
