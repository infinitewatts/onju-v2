"""send_sentence with tts.local.stream on: fake synthesis and fake socket."""
import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from pipeline.local_wake import LocalSpeechWakeGate
from pipeline.main import process_utterances

CONFIG = {
    "network": {"tcp_port": 3001},
    "device": {"default_volume": 15, "default_mic_timeout": 60, "led_fade": 2},
    "audio": {"sample_rate": 16000, "opus_frame_size": 320},
    "asr": {"url": "http://unused"},
    "conversation": {"backend": "conversational"},
    "tts": {"backend": "local", "local": {"stream": True}},
}
PIECE = b"\x10\x00" * 3200  # 0.2 s at 16 kHz = 10 frames


def device(stream):
    return SimpleNamespace(
        conversation=SimpleNamespace(stream=stream, commit=Mock()),
        interrupted=asyncio.Event(), processing=False, vad_writer=None,
        hostname="onju-000001", ip="127.0.0.1", voice="aiden", ptt=False,
        last_user_text=None, last_response=None, wake_gate=None,
    )


class StreamedSendTests(unittest.IsolatedAsyncioTestCase):
    async def run_pipeline(self, synth, check, *, text="Hello there.", config=CONFIG, gate=None,
                           gap=0.05, cap=3.0, margin=0.05):
        async def stream(*a, **k):
            yield text

        dev = device(stream)
        dev.wake_gate = gate
        queue = asyncio.Queue()
        await queue.put((dev, np.zeros(512, dtype=np.int16)))
        env = SimpleNamespace(opened=[], writes=[], closed=asyncio.Event(), plain=[],
                              upstream_closed=asyncio.Event(), emitted=0, kwargs=None)

        async def fake_stream(ip, port, frames, stats=None, **kwargs):
            env.opened.append(env.emitted)
            env.kwargs = kwargs
            try:
                async for batch in frames:
                    env.writes.append((time.monotonic(), len(batch)))
                    stats.frames_written += len(batch)
                return True
            finally:
                env.closed.set()

        async def plain(ip, port, payload, **kwargs):
            env.plain.append((payload, kwargs))
            return True

        async def synthesize_stream(*a):
            try:
                async for piece in synth(env):
                    env.emitted += 1
                    yield piece
            finally:
                env.upstream_closed.set()

        with patch("pipeline.main.asr.transcribe", AsyncMock(return_value={"text": "Hi"})), \
             patch("pipeline.main.tts.synthesize_stream", side_effect=synthesize_stream), \
             patch("pipeline.main.tts.synthesize", AsyncMock(side_effect=AssertionError)), \
             patch("pipeline.main.send_audio_stream", side_effect=fake_stream), \
             patch("pipeline.main.send_audio", side_effect=plain), \
             patch("pipeline.main.STREAM_GAP_S", gap), \
             patch("pipeline.main.STREAM_MARGIN_S", margin), \
             patch("pipeline.main.STREAM_SILENCE_CAP_S", cap):
            task = asyncio.create_task(process_utterances(config, None, queue))
            try:
                await check(dev, queue, env)
            finally:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
        return dev, env

    async def test_connects_only_after_prebuffer(self):
        async def synth(env):
            for _ in range(5):
                await asyncio.sleep(0.03)
                yield PIECE

        async def check(dev, queue, env):
            await asyncio.wait_for(queue.join(), 3)
            # 12 chars -> 0.4 + 0.15 * 0.696 = 0.504 s -> needs 3 pieces (0.6 s)
            self.assertEqual(env.opened, [3])
            self.assertEqual(sum(n for _, n in env.writes), 50)
            self.assertEqual(env.kwargs["mic_timeout"], 0)
            self.assertEqual(env.plain[-1][1]["mic_timeout"], 60)
            dev.conversation.commit.assert_called_once_with("Hello there.")

        await self.run_pipeline(synth, check)

    async def test_finished_short_stream_sends_immediately(self):
        async def synth(env):
            yield PIECE

        async def check(dev, queue, env):
            await asyncio.wait_for(queue.join(), 3)
            self.assertEqual(env.opened, [1])
            self.assertEqual(sum(n for _, n in env.writes), 10)

        await self.run_pipeline(synth, check)

    async def test_stalled_producer_gets_silence_before_the_speaker_runs_dry(self):
        async def synth(env):
            yield PIECE * 3          # 0.6 s of audio
            await asyncio.sleep(1.0)  # longer than the audio already sent
            yield PIECE

        async def check(dev, queue, env):
            await asyncio.wait_for(queue.join(), 4)
            total = sum(n for _, n in env.writes)
            self.assertGreater(total, 40)  # 40 real frames plus silence
            # Audio sent always stays ahead of real-time playback.
            start = env.writes[0][0]
            sent = 0
            for at, n in env.writes:
                self.assertGreater(sent * 0.02 - (at - start), -0.03) if sent else None
                sent += n
            dev.conversation.commit.assert_called_once()

        await self.run_pipeline(synth, check)

    async def test_bursts_slower_than_real_time_get_no_silence_while_buffered(self):
        async def synth(env):
            yield PIECE * 6           # 1.2 s buffered before playback starts
            for _ in range(4):
                await asyncio.sleep(0.25)  # each 0.2 s burst arrives late
                yield PIECE

        async def check(dev, queue, env):
            await asyncio.wait_for(queue.join(), 4)
            self.assertEqual(sum(n for _, n in env.writes), 100)  # no silence added
            dev.conversation.commit.assert_called_once()

        await self.run_pipeline(synth, check)

    async def test_silence_cap_ends_sentence_and_reopens_mic(self):
        async def synth(env):
            yield PIECE * 3
            await asyncio.Event().wait()

        async def check(dev, queue, env):
            await asyncio.wait_for(queue.join(), 3)
            self.assertTrue(env.upstream_closed.is_set())
            self.assertTrue(env.closed.is_set())
            silence = sum(n for _, n in env.writes) - 30
            self.assertGreater(silence, 0)
            self.assertLessEqual(silence * 0.02, 0.3 + 1e-6)
            dev.conversation.commit.assert_not_called()
            self.assertEqual(env.plain[-1], (b"", {"mic_timeout": 60, "volume": 0, "fade": 0}))

        await self.run_pipeline(synth, check, cap=0.3)

    async def test_upstream_failure_after_audio_is_failure(self):
        async def synth(env):
            yield PIECE * 3
            raise RuntimeError("boom")

        async def check(dev, queue, env):
            await asyncio.wait_for(queue.join(), 3)
            dev.conversation.commit.assert_not_called()
            self.assertEqual(env.plain[-1][1]["mic_timeout"], 60)

        await self.run_pipeline(synth, check)

    async def test_second_sentence_needs_no_prebuffer_while_first_still_plays(self):
        calls = []

        async def synth(env):
            calls.append(env.emitted)
            if len(calls) == 1:
                yield PIECE * 10      # 2 s of audio, delivered at once
            else:
                for _ in range(4):    # slower than real time, in small bursts
                    await asyncio.sleep(0.05)
                    yield PIECE

        async def check(dev, queue, env):
            await asyncio.wait_for(queue.join(), 4)
            # Sentence two connects on its first burst: the speaker still has
            # sentence one queued, so no extra lead is held back.
            self.assertEqual(env.opened, [1, 2])
            # In order, complete, and with no silence added.
            self.assertEqual(sum(n for _, n in env.writes), 140)
            dev.conversation.commit.assert_called_once_with("One two three. Four five six.")

        dev, env = await self.run_pipeline(synth, check, text="One two three. Four five six.")

    async def test_interruption_closes_upstream_and_socket(self):
        async def synth(env):
            yield PIECE * 3
            await asyncio.Event().wait()

        async def check(dev, queue, env):
            while not env.writes:
                await asyncio.sleep(0.01)
            dev.interrupted.set()
            await asyncio.wait_for(queue.join(), 3)
            self.assertTrue(env.upstream_closed.is_set())
            self.assertTrue(env.closed.is_set())
            dev.conversation.commit.assert_not_called()
            self.assertEqual(env.plain[-1][1]["mic_timeout"], 60)

        await self.run_pipeline(synth, check, gap=1.0)

    async def test_playback_end_counts_frames_sent(self):
        gate = Mock(spec=LocalSpeechWakeGate)

        async def synth(env):
            for _ in range(5):
                yield PIECE  # 1.0 s total

        async def check(dev, queue, env):
            await asyncio.wait_for(queue.join(), 3)
            gate.await_follow_up.assert_called_once()
            lead = gate.await_follow_up.call_args.args[2]
            self.assertGreater(lead, 0.85)
            self.assertLessEqual(lead, 1.0 + 0.12)  # audio sent plus the speaker's end-of-sentence silence

        cfg = {**CONFIG, "wake_word": {"follow_up_window_s": 15}}
        await self.run_pipeline(synth, check, gate=gate, config=cfg)

    async def test_flag_off_uses_plain_path(self):
        cfg = {**CONFIG, "tts": {"backend": "local", "local": {"stream": False}}}

        async def synth(env):
            yield PIECE

        async def check(dev, queue, env):
            await asyncio.wait_for(queue.join(), 3)
            self.assertEqual(env.opened, [])
            self.assertEqual(env.emitted, 0)

        # The helper makes one-shot synthesize raise, so the plain path fails
        # cleanly; the point is that the streamed sender and generator stay unused.
        await self.run_pipeline(synth, check, config=cfg)


if __name__ == "__main__":
    unittest.main()
