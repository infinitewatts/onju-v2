"""Wake admission tests never use a microphone, real model or paid agent."""
import asyncio
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from pipeline.main import udp_listener, process_utterances, maintain_wake_audio


CONFIG = {
    "network": {"tcp_port": 3001, "udp_port": 3000},
    "device": {"default_volume": 16, "default_mic_timeout": 60, "led_fade": 2,
               "led_power": 50, "led_update_period": 0.25},
    "audio": {"sample_rate": 16000, "chunk_size": 512, "opus_frame_size": 320},
    "vad": {"threshold": 0.5}, "asr": {"url": "unused"},
    "conversation": {"backend": "conversational"},
}


class WakeIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_active_window_without_frames_still_renews_audio_lease(self):
        dev = self.device()
        dev.wake_gate.active = True
        manager = SimpleNamespace(devices={dev.hostname: dev})
        with patch("pipeline.main.send_stop_listening", AsyncMock()) as renew, \
             patch("pipeline.main.asyncio.sleep", AsyncMock(side_effect=asyncio.CancelledError())):
            with self.assertRaises(asyncio.CancelledError):
                await maintain_wake_audio(CONFIG, manager)
        renew.assert_awaited_once_with(dev.ip, 3001, hold_s=60)

    def device(self):
        return SimpleNamespace(
            hostname="onju-000001", ip="127.0.0.1", ptt=False,
            processing=False, interrupted=asyncio.Event(), vad_writer=None,
            led_power=0, led_update_time=0,
            wake_gate=Mock(active=False, just_activated=False),
            vad=Mock(pre_buffer=deque(), speech_prob=0, recording=False),
            conversation=SimpleNamespace(stream=Mock(), commit=Mock()),
            last_user_text=None, last_response=None,
            observe_audio=Mock(),
        )

    async def test_ambient_never_enters_vad_or_utterance_queue(self):
        dev = self.device()
        dev.wake_gate.feed.return_value = []
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        packets = [(b"\xff" * 512, (dev.ip, 1234))] * 20
        with patch("pipeline.main.socket.socket"), patch.object(loop, "sock_recvfrom",
                AsyncMock(side_effect=packets + [asyncio.CancelledError()])):
            with self.assertRaises(asyncio.CancelledError):
                await udp_listener(CONFIG, SimpleNamespace(get_by_ip=lambda ip: dev), queue)
        dev.vad.process_frame.assert_not_called()
        self.assertTrue(queue.empty())

    async def test_one_activated_utterance_is_queued_then_gate_resets(self):
        dev = self.device()
        pcm = np.zeros(512, dtype=np.int16)
        dev.wake_gate.feed.return_value = [pcm]
        dev.vad.process_frame.return_value = pcm
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        with patch("pipeline.main.socket.socket"), patch.object(loop, "sock_recvfrom",
                AsyncMock(side_effect=[(b"\xff" * 512, (dev.ip, 1234)), asyncio.CancelledError()])):
            with self.assertRaises(asyncio.CancelledError):
                await udp_listener(CONFIG, SimpleNamespace(get_by_ip=lambda ip: dev), queue)
        self.assertEqual(queue.qsize(), 1)
        dev.wake_gate.reset.assert_called_once()

    async def test_wake_name_alone_never_calls_paid_agent(self):
        dev = self.device()
        queue = asyncio.Queue()
        await queue.put((dev, np.zeros(512, dtype=np.int16)))
        with patch("pipeline.main.asr.transcribe", AsyncMock(return_value={"text": "Hey Aiden!"})), \
             patch("pipeline.main.send_audio", AsyncMock()):
            task = asyncio.create_task(process_utterances(CONFIG, None, queue))
            try:
                await asyncio.wait_for(queue.join(), 2)
            finally:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
        dev.conversation.stream.assert_not_called()
        dev.wake_gate.arm.assert_called_once()
        self.assertEqual(dev.wake_gate.arm.call_args.kwargs, {"window_s": 10})
        dev.wake_gate.reset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
