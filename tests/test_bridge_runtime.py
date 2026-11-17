"""Local transport tests, with no speaker, paid agent, or network calls."""
import asyncio
import copy
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from pipeline.bridge_runtime import BridgeRuntime, identity
from pipeline.main import _queue_utterance, process_utterances
from tests.test_voice_pilot import CONFIG, DEVICE, device


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.config = copy.deepcopy(CONFIG)
        self.config['orchestrator_bridge'] = {'mode': 'live', 'host_session_id': 'host',
            'device': DEVICE, 'state_path': self.directory.name + '/bridge.sqlite'}
        self.d = device()
        self.manager = SimpleNamespace(devices={DEVICE: self.d})
        self.bridge = BridgeRuntime(self.config, self.manager)
        self.tts = patch('pipeline.bridge_runtime.tts.synthesize', new=AsyncMock(return_value=b'pcm')).start()
        self.send = patch('pipeline.bridge_runtime.send_audio', new=AsyncMock()).start()
        patch('pipeline.bridge_runtime.opus_encode', return_value=[]).start()
        patch('pipeline.bridge_runtime.opus_frames_to_tcp_payload', return_value=b'audio').start()

    async def asyncTearDown(self):
        await self.bridge.close()
        patch.stopall()
        self.directory.cleanup()

    def submit(self, version=1, question_id='question', **extra):
        payload = dict(op='submit', host_session_id='host', session_id='session', question_id=question_id,
                       version=version, text='Choose a color?', kind='structured', options=[{'label': 'blue'},
                       {'label': 'green', 'description': 'Green details'}], requires_desktop=False,
                       expires_at=time.time() + 300, **extra)
        return self.bridge.dispatch(payload)['question']

    async def test_observe_has_zero_playback_or_interception(self):
        await self.bridge.close()
        self.config['orchestrator_bridge']['mode'] = 'observe'
        self.bridge = BridgeRuntime(self.config, self.manager)
        q = self.submit()
        self.assertEqual(q['status'], 'observed')
        self.bridge.start()
        await self.bridge.tick()
        self.assertIsNone(self.bridge.capture_ref(self.d))
        self.assertFalse(await self.bridge.intercept(self.d, 'one', identity(q)))
        self.tts.assert_not_awaited()
        self.send.assert_not_awaited()
        self.d.wake_gate.reset.assert_not_called()
        self.assertIsNone(self.bridge.task)

    async def test_matching_capture_persists_before_confirmation(self):
        self.submit()
        await self.bridge.tick()
        ref = self.bridge.capture_ref(self.d)
        self.assertIsNotNone(ref)
        async def confirm(*args, **kwargs):
            q = self.bridge.dispatch(dict(ref, op='status'))['question']
            self.assertEqual(q['status'], 'captured')
            self.assertNotEqual(q['status'], 'acknowledged')
        self.send.side_effect = confirm
        self.assertTrue(await self.bridge.intercept(self.d, 'option two', ref))
        q = self.bridge.dispatch(dict(ref, op='status'))['question']
        self.assertEqual(q['answer']['choice'], 2)
        self.assertTrue(q['unverified'])
        self.assertIn('Answer captured', self.tts.await_args.args[0])
        self.assertNotIn('received your answer', self.tts.await_args.args[0])

    async def test_stale_capture_cannot_attach_to_new_version(self):
        self.submit()
        await self.bridge.tick()
        ref = self.bridge.capture_ref(self.d)
        new = self.submit(version=2)
        await self.bridge.tick()
        self.assertTrue(await self.bridge.intercept(self.d, 'one', ref))
        q = self.bridge.dispatch(dict(identity(new), op='status'))['question']
        self.assertEqual(q['status'], 'announced')
        self.assertNotIn('answer', q)

    async def test_cancel_during_tts_sends_no_audio(self):
        q = self.submit()
        async def cancel(*args):
            self.bridge.dispatch(dict(identity(q), op='cancel'))
            return b'pcm'
        self.tts.side_effect = cancel
        await self.bridge.tick()
        self.send.assert_not_awaited()
        self.assertFalse(self.d.processing)
        self.assertFalse(self.d.bridge_speaking)

    async def test_one_reservation_and_busy_capture_defers_prompt(self):
        self.submit()
        self.d.queued_utterances = 1
        await self.bridge.tick()
        self.tts.assert_not_awaited()
        self.d.queued_utterances = 0
        entered, release = asyncio.Event(), asyncio.Event()
        async def slow(*args):
            entered.set()
            await release.wait()
            return b'pcm'
        self.tts.side_effect = slow
        task = asyncio.create_task(self.bridge.tick())
        await entered.wait()
        await self.bridge.tick()
        self.assertEqual(self.tts.await_count, 1)
        release.set()
        await task
        await self.bridge.tick()
        self.assertEqual(self.send.await_count, 1)

    async def test_queue_preserves_capture_identity_through_state_change(self):
        self.submit()
        await self.bridge.tick()
        ref = self.bridge.capture_ref(self.d)
        self.d.bridge_capture_ref = ref
        queue = asyncio.Queue()
        await _queue_utterance(queue, self.d, np.zeros(320, dtype=np.int16))
        self.submit(version=2)
        item = queue.get_nowait()
        self.assertEqual(item[2], ref)
        self.assertIsNone(self.d.bridge_capture_ref)

    async def test_unrelated_continues_paid_agent_path(self):
        self.submit()
        await self.bridge.tick()
        queue = asyncio.Queue()
        await queue.put((self.d, np.zeros(320, dtype=np.int16), self.bridge.capture_ref(self.d)))
        called = asyncio.Event()
        async def agent(*args, **kwargs):
            called.set()
            if False:
                yield ''
        self.d.conversation.stream.side_effect = agent
        with patch('pipeline.main.asr.transcribe', new=AsyncMock(return_value={'text': 'Hey Aiden what time is it?'})), \
             patch('pipeline.main.send_audio', new=AsyncMock()):
            task = asyncio.create_task(process_utterances(self.config, self.manager, queue, bridge=self.bridge))
            await asyncio.wait_for(called.wait(), 1)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(self.d.conversation.stream.call_count, 1)

    async def test_close_cancels_tts_and_releases_reservation(self):
        self.submit()
        entered = asyncio.Event()
        async def slow(*args):
            entered.set()
            await asyncio.Future()
        self.tts.side_effect = slow
        self.bridge.start()
        await entered.wait()
        await self.bridge.close()
        self.assertFalse(self.d.processing)
        self.assertFalse(self.d.bridge_speaking)
        self.assertEqual(self.bridge.cache_bytes, 0)
        # Teardown is idempotent.
        self.bridge.store = None

    async def test_receipt_only_after_ack_and_once_across_restart(self):
        self.submit()
        await self.bridge.tick()
        ref = self.bridge.capture_ref(self.d)
        await self.bridge.intercept(self.d, 'one', ref)
        await self.bridge.tick()
        self.assertFalse(any('received your answer' in c.args[0] for c in self.tts.await_args_list))
        q = self.bridge.dispatch(dict(ref, op='emitted'))['question']
        self.bridge.dispatch(dict(ref, op='ack', answer_id=q['answer_id']))
        await self.bridge.tick()
        self.assertEqual(sum('received your answer' in c.args[0] for c in self.tts.await_args_list), 1)
        await self.bridge.close()
        self.bridge = BridgeRuntime(self.config, self.manager)
        await self.bridge.tick()
        self.assertEqual(sum('received your answer' in c.args[0] for c in self.tts.await_args_list), 1)

    async def test_replacement_while_asr_runs_rejects_original_answer(self):
        self.submit()
        await self.bridge.tick()
        ref = self.bridge.capture_ref(self.d)
        queue = asyncio.Queue()
        await queue.put((self.d, np.zeros(320, dtype=np.int16), ref))
        async def asr(*args):
            self.submit(version=2)
            return {'text': 'Hey Aiden option one'}
        with patch('pipeline.main.asr.transcribe', new=AsyncMock(side_effect=asr)), \
             patch('pipeline.main.send_audio', new=AsyncMock()):
            task = asyncio.create_task(process_utterances(self.config, self.manager, queue, bridge=self.bridge))
            await asyncio.wait_for(queue.join(), 1)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        current = self.bridge.dispatch({'op': 'next'})['question']
        self.assertEqual(current['version'], 2)
        self.assertNotIn('answer', current)
        self.d.conversation.stream.assert_not_called()

    async def test_control_bridge_only_larger_body_and_off_is_absent(self):
        from pipeline.main import control_server
        from tests.test_voice_pilot import FakeWriter
        started = asyncio.Event()
        handlers = []
        class Server:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def serve_forever(self):
                await asyncio.Future()
        async def start(handler, host, port):
            self.assertEqual(host, '127.0.0.1')
            handlers.append(handler)
            started.set()
            return Server()
        async def request(target, body):
            reader = asyncio.StreamReader()
            reader.feed_data(f'POST {target} HTTP/1.1\r\nContent-Length: {len(body)}\r\n\r\n'.encode() + body)
            reader.feed_eof()
            writer = FakeWriter()
            await handlers[-1](reader, writer)
            return bytes(writer.data)
        with patch('pipeline.main.asyncio.start_server', side_effect=start):
            task = asyncio.create_task(control_server(self.config, self.manager, bridge=self.bridge))
            await started.wait()
            self.assertIn(b'200 OK', await request('/bridge', b'{"op":"next"}' + b' ' * 5000))
            self.assertEqual(await request('/devices', b' ' * 5000), b'')
            self.assertEqual(await request('/bridge', b' ' * 32769), b'')
            await self.bridge.close()
            self.assertIn(b'404 Not Found', await request('/bridge', b'{"op":"next"}'))
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_cancellation_during_tcp_connect_prevents_first_write(self):
        from pipeline.protocol import send_audio as real_send_audio
        q = self.submit()
        writer = SimpleNamespace(write=Mock(), drain=AsyncMock(), close=Mock(), wait_closed=AsyncMock())
        async def connect(*args):
            self.bridge.dispatch(dict(identity(q), op='cancel'))
            return None, writer
        with patch('pipeline.protocol.asyncio.open_connection', new=AsyncMock(side_effect=connect)), \
             patch('pipeline.bridge_runtime.send_audio', new=real_send_audio):
            await self.bridge.tick()
        writer.write.assert_not_called()
        writer.close.assert_called_once()
        writer.wait_closed.assert_awaited_once()
        self.assertEqual(self.bridge.dispatch(dict(identity(q), op='status'))['question']['status'], 'cancelled')

    async def test_transport_failure_cannot_mark_announcement(self):
        self.submit()
        self.send.return_value = False
        await self.bridge.tick()
        self.assertEqual(self.bridge.dispatch({'op': 'next'})['question']['status'], 'queued')

    async def test_transport_cancel_closes_owned_connection(self):
        from pipeline.protocol import send_audio as real_send_audio
        draining = asyncio.Event()
        async def drain():
            draining.set()
            await asyncio.Future()
        writer = SimpleNamespace(write=Mock(), drain=drain, close=Mock(), wait_closed=AsyncMock())
        with patch('pipeline.protocol.asyncio.open_connection', new=AsyncMock(return_value=(None, writer))):
            task = asyncio.create_task(real_send_audio('127.0.0.1', 3001, b'audio'))
            await draining.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        writer.close.assert_called_once()
        writer.wait_closed.assert_awaited_once()

    async def test_health_is_scalar_and_observe_mode_has_no_question(self):
        await self.bridge.close()
        self.config['orchestrator_bridge']['mode'] = 'observe'
        self.bridge = BridgeRuntime(self.config, self.manager)
        self.assertEqual(self.bridge.dispatch({'op': 'health'}),
                         {'mode': 'observe', 'host_session_id': 'host', 'device': DEVICE, 'closed': False})
        self.assertIsNone(self.bridge.dispatch({'op': 'next'})['question'])
        self.tts.assert_not_awaited()
        self.send.assert_not_awaited()

    async def test_cancel_context_survives_restart_without_latest_guess(self):
        self.submit()
        await self.bridge.tick()
        ref = self.bridge.capture_ref(self.d)
        await self.bridge.intercept(self.d, 'one', ref)
        await self.bridge.close()
        self.bridge = BridgeRuntime(self.config, self.manager)
        restored = self.bridge.capture_ref(self.d)
        self.assertEqual(restored, ref)
        await self.bridge.intercept(self.d, 'cancel that answer', restored)
        self.assertEqual(self.bridge.dispatch(dict(ref, op='status'))['question']['status'], 'cancelled')

    async def test_multiple_pending_answers_clarify_and_pending_count_is_truthful(self):
        refs = []
        for qid in ('first', 'second'):
            self.submit(question_id=qid)
            await self.bridge.tick()
            ref = self.bridge.capture_ref(self.d)
            refs.append(ref)
            await self.bridge.intercept(self.d, 'one', ref)
        await self.bridge.close()
        self.bridge = BridgeRuntime(self.config, self.manager)
        ambiguous = self.bridge.capture_ref(self.d)
        self.assertEqual(ambiguous, {'ambiguous': True})
        await self.bridge.intercept(self.d, 'cancel that answer', ambiguous)
        self.assertIn('More than one answer', self.tts.await_args.args[0])
        for ref in refs:
            self.assertEqual(self.bridge.dispatch(dict(ref, op='status'))['question']['status'], 'captured')
        await self.bridge.intercept(self.d, 'what is pending', ambiguous)
        self.assertIn('2 pending', self.tts.await_args.args[0])
