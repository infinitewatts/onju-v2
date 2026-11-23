"""Deterministic streaming tests: no microphone, model, network or speaker."""
import asyncio
import unittest
from contextlib import aclosing
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from pipeline.local_wake import LocalSpeechWakeGate
from pipeline.main import interruptible_sentences, process_utterances
from pipeline.conversation.agentic import AgenticBackend
from pipeline.protocol import send_audio


CONFIG = {
    "network": {"tcp_port": 3001},
    "device": {"default_volume": 15, "default_mic_timeout": 60, "led_fade": 2},
    "audio": {"sample_rate": 16000, "opus_frame_size": 320},
    "asr": {"url": "http://unused"},
    "conversation": {"backend": "conversational"},
}


def device(stream):
    return SimpleNamespace(
        conversation=SimpleNamespace(stream=stream, commit=Mock()),
        interrupted=asyncio.Event(), processing=False, vad_writer=None,
        hostname="onju-000001", ip="127.0.0.1", voice="aiden", ptt=False,
        last_user_text=None, last_response=None,
    )


class PipelineStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_short_listening_window_follows_opus_on_same_connection(self):
        for timeout in (0, 10, 60):
            for audio in (b"", b"\x00\x03abc\x00\x00"):
                with self.subTest(timeout=timeout, audio=audio):
                    with patch("pipeline.protocol.send_tcp", AsyncMock()) as tcp:
                        await send_audio("127.0.0.1", 3001, audio, mic_timeout=timeout)
                    tcp.assert_awaited_once()
                    header, payload = tcp.call_args.args[2:]
                    self.assertEqual(header[:3], bytes([0xAA, 0, timeout]))
                    expected = audio or b"\x00\x00"
                    if timeout < 20:
                        expected += bytes([0xDD, 0, timeout, 0, 0, 0])
                    self.assertEqual(payload, expected)

    async def run_pipeline(self, stream, check, *, gate=None, config=CONFIG):
        dev = device(stream)
        dev.wake_gate = gate
        queue = asyncio.Queue()
        await queue.put((dev, np.zeros(512, dtype=np.int16)))
        sent = []
        first_audio = asyncio.Event()

        async def send(ip, port, payload, **kwargs):
            sent.append((payload, kwargs))
            if payload:
                first_audio.set()
            return True

        with patch("pipeline.main.asr.transcribe", AsyncMock(return_value={"text": "Hello"})), \
             patch("pipeline.main.tts.synthesize", AsyncMock(return_value=b"\x00\x00" * 320)) as tts, \
             patch("pipeline.main.send_audio", side_effect=send):
            task = asyncio.create_task(process_utterances(config, None, queue))
            try:
                await check(dev, queue, sent, first_audio, tts)
            finally:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
        self.assertFalse(dev.processing)
        return dev, sent

    async def test_acknowledgment_reopens_conversation(self):
        gate = Mock(spec=LocalSpeechWakeGate)
        async def stream(*args, **kwargs):
            yield "Okay."
        async def check(dev, queue, sent, first_audio, tts):
            await asyncio.wait_for(queue.join(), 2)
            gate.await_follow_up.assert_called_once()
            self.assertEqual(gate.await_follow_up.call_args.args[1], 15)
            gate.reset.assert_not_called()
        await self.run_pipeline(stream, check, gate=gate,
                                config={**CONFIG, "wake_word": {"follow_up_window_s": 15}})

    async def test_first_sentence_plays_before_second_exists_and_tail_finishes(self):
        release = asyncio.Event()

        async def stream(*args, **kwargs):
            yield "First sentence. "
            await release.wait()
            yield "Trailing fragment"

        async def check(dev, queue, sent, first_audio, tts):
            await asyncio.wait_for(first_audio.wait(), 2)
            self.assertFalse(release.is_set())
            self.assertEqual(tts.call_args.args[0], "First sentence.")
            self.assertEqual(sent[0][1]["mic_timeout"], 0)
            release.set()
            await asyncio.wait_for(queue.join(), 2)
            self.assertEqual([c.args[0] for c in tts.call_args_list],
                             ["First sentence.", "Trailing fragment"])
            self.assertEqual(sent[-1][0], b"")
            self.assertEqual(sent[-1][1]["mic_timeout"], 60)
            dev.conversation.commit.assert_called_once_with("First sentence. Trailing fragment")

        await self.run_pipeline(stream, check)

    async def test_interruption_closes_stalled_stream_without_committing(self):
        closed = asyncio.Event()

        async def stream(*args, **kwargs):
            try:
                yield "First. "
                await asyncio.Event().wait()
            finally:
                closed.set()

        async def check(dev, queue, sent, first_audio, tts):
            await asyncio.wait_for(first_audio.wait(), 2)
            dev.interrupted.set()
            await asyncio.wait_for(queue.join(), 2)
            self.assertTrue(closed.is_set())
            dev.conversation.commit.assert_not_called()
            self.assertEqual(sent[-1][1]["mic_timeout"], 60)

        await self.run_pipeline(stream, check)

    async def test_partial_failure_recovers_mic_without_committing(self):
        async def stream(*args, **kwargs):
            yield "First. "
            raise RuntimeError("synthetic upstream failure")

        async def check(dev, queue, sent, first_audio, tts):
            await asyncio.wait_for(queue.join(), 2)
            self.assertEqual(len(sent), 2)
            self.assertEqual(sent[-1][1]["mic_timeout"], 60)
            dev.conversation.commit.assert_not_called()

        await self.run_pipeline(stream, check)

    async def test_interrupt_cancels_blocked_tts_and_closes_upstream(self):
        started, tts_closed, stream_closed = (asyncio.Event() for _ in range(3))

        async def stream(*args, **kwargs):
            try:
                yield "First. "
                await asyncio.Event().wait()
            finally:
                stream_closed.set()

        async def blocked_tts(*args):
            try:
                started.set()
                await asyncio.Event().wait()
            finally:
                tts_closed.set()

        async def check(dev, queue, sent, first_audio, tts):
            tts.side_effect = blocked_tts
            await asyncio.wait_for(started.wait(), 2)
            dev.interrupted.set()
            await asyncio.wait_for(queue.join(), 2)
            self.assertTrue(tts_closed.is_set())
            self.assertTrue(stream_closed.is_set())
            self.assertEqual(sent, [])
            dev.conversation.commit.assert_not_called()

        await self.run_pipeline(stream, check)

    async def test_empty_response_reopens_mic(self):
        async def stream(*args, **kwargs):
            if False:
                yield ""

        async def check(dev, queue, sent, first_audio, tts):
            await asyncio.wait_for(queue.join(), 2)
            tts.assert_not_called()
            self.assertEqual(sent, [(b"", {"mic_timeout": 60, "volume": 0, "fade": 0})])

        await self.run_pipeline(stream, check)

    async def test_interrupt_before_first_sentence_closes_upstream(self):
        started, closed = asyncio.Event(), asyncio.Event()

        async def stream(*args, **kwargs):
            try:
                started.set()
                await asyncio.Event().wait()
                yield "Never"
            finally:
                closed.set()

        dev = device(stream)
        async with aclosing(interruptible_sentences(dev, "hi", None)) as sentences:
            task = asyncio.create_task(anext(sentences))
            await asyncio.wait_for(started.wait(), 2)
            dev.interrupted.set()
            with self.assertRaises(StopAsyncIteration):
                await asyncio.wait_for(task, 2)
        self.assertTrue(closed.is_set())

    async def test_agentic_backend_closes_http_stream_on_early_exit(self):
        async def events():
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello"))])
            await asyncio.Event().wait()

        class HTTPStream:
            close = AsyncMock()
            def __aiter__(self):
                return events()

        stream = HTTPStream()
        backend = AgenticBackend.__new__(AgenticBackend)
        backend._build_kwargs = Mock(return_value={})
        backend.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=AsyncMock(return_value=stream))))
        async with aclosing(backend.stream("hi")) as chunks:
            self.assertEqual(await anext(chunks), "Hello")
        stream.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
