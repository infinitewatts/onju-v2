"""Automatic level control: pure arithmetic, no microphone or model."""
import asyncio
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from pipeline.auto_gain import AutoGain
from pipeline.device import Device
from pipeline.main import udp_listener
from pipeline.audio import decode_ulaw

AUDIO = {"sample_rate": 16000, "chunk_size": 512, "opus_frame_size": 320}


def tone(peak, n=512):
    return (np.sin(np.arange(n) * 0.2) * peak).astype(np.int16)


def run(gain, frames):
    """Outputs lined up with their inputs (output runs one frame behind)."""
    out = [gain.process(f) for f in frames]
    out.append(gain.process(np.zeros_like(frames[-1])))
    return out[1:]


class AutoGainTests(unittest.TestCase):
    def test_output_is_the_previous_frame_and_the_first_is_silence(self):
        gain = AutoGain(target_peak=24000, max_gain=16)
        loud = [tone(30000), -tone(30000), tone(30000)]
        first = gain.process(loud[0])
        np.testing.assert_array_equal(first, np.zeros(512, dtype=np.int16))
        np.testing.assert_array_equal(gain.process(loud[1]), loud[0])
        np.testing.assert_array_equal(gain.process(loud[2]), loud[1])

    def test_quiet_speech_is_raised_to_the_target_and_never_past_max_gain(self):
        gain = AutoGain(target_peak=24000, max_gain=16)
        out = run(gain, [tone(2000)] * 6)
        self.assertAlmostEqual(int(np.max(np.abs(out[3]))), 24000, delta=300)
        whisper = AutoGain(target_peak=24000, max_gain=16)
        out = run(whisper, [tone(100)] * 6)
        self.assertEqual(whisper.gain, 16.0)
        self.assertAlmostEqual(int(np.max(np.abs(out[3]))), 1600, delta=20)

    def test_loud_audio_is_returned_untouched(self):
        gain = AutoGain(target_peak=24000, max_gain=16)
        loud = tone(30000)
        out = run(gain, [loud] * 4)
        self.assertEqual(gain.gain, 1.0)
        for frame in out[:3]:
            np.testing.assert_array_equal(frame, loud)

    def test_gain_eases_down_before_a_loud_frame_and_nothing_clips(self):
        for onset_peak in (32767, 16000, 8000, 3000, 2000, 500):
            with self.subTest(onset_peak=onset_peak):
                gain = AutoGain(target_peak=24000, max_gain=16)
                quiet, onset = tone(60), tone(onset_peak)
                out = run(gain, [quiet] * 60 + [onset, onset])
                before, first, second = (o.astype(np.int32) for o in out[59:62])
                wanted = max(1.0, min(16.0, 24000 / int(np.max(np.abs(onset)))))
                # The quiet frame just before the onset ends at the onset's gain...
                self.assertAlmostEqual(before[-1] / max(int(quiet[-1]), 1), wanted, delta=0.6)
                # ...so the onset itself is at one steady gain, unclipped.
                for frame in (first, second):
                    self.assertLessEqual(int(np.max(np.abs(frame))), max(onset_peak, 24000) + 1)
                    np.testing.assert_allclose(frame, onset.astype(np.float64) * wanted, atol=2)

    def test_gain_recovers_gradually_after_loud_sound(self):
        gain = AutoGain(target_peak=24000, max_gain=16, release_s=1.0)
        gain.process(tone(24000))
        seen = []
        for _ in range(125):  # four seconds of quiet
            gain.process(tone(100))
            seen.append(gain.gain)
        self.assertLess(seen[1], 1.1)
        self.assertLess(seen[15], 2.0)          # half a second later still low
        self.assertEqual(seen[-1], 16.0)        # fully recovered
        self.assertEqual(seen, sorted(seen))    # never jumps back down on quiet audio

    def test_gain_change_is_spread_across_the_frame(self):
        gain = AutoGain(target_peak=24000, max_gain=16)
        flat = lambda v: np.full(512, v, dtype=np.int16)
        out = run(gain, [flat(12000), flat(12000), flat(6000), flat(6000), flat(6000)])
        rising = out[3].astype(np.int32)        # gain recovering after the louder frames
        self.assertLessEqual(int(np.abs(np.diff(rising)).max()), 2)   # a ramp, not a step
        self.assertGreater(int(rising[-1]), int(rising[0]))

    def test_odd_input_stays_in_range(self):
        gain = AutoGain()
        zeros = np.zeros(512, dtype=np.int16)
        np.testing.assert_array_equal(run(gain, [zeros] * 3)[1], zeros)
        spike = zeros.copy(); spike[100] = -32768
        rng = np.random.default_rng(1)
        frames = [zeros, spike, np.full(512, -32768, dtype=np.int16), zeros] + \
                 [(rng.standard_normal(512) * s).astype(np.int16) for s in (5, 20000, 5, 3000, 50, 12000)]
        for frame in run(gain, frames):
            self.assertEqual((frame.dtype, frame.shape), (np.int16, (512,)))
            self.assertGreaterEqual(gain.gain, 1.0)
            self.assertLessEqual(gain.gain, 16.0)
        # Random onsets: gain never pushes a sample into the rail.
        gain = AutoGain()
        frames = [(rng.standard_normal(512) * s).clip(-30000, 30000).astype(np.int16)
                  for s in rng.choice([5, 60, 800, 3000, 9000], size=400)]
        for frame in run(gain, frames):
            self.assertLess(int(np.max(np.abs(frame.astype(np.int32)))), 32767)

    def test_bad_settings(self):
        for bad in ({"target_peak": 0}, {"target_peak": 40000}, {"max_gain": 0.5}, {"release_s": 0}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                AutoGain(**bad)

    def test_config_switch(self):
        self.assertIsNone(AutoGain.from_config({"audio": AUDIO}))
        self.assertIsNone(AutoGain.from_config({"audio": {**AUDIO, "auto_gain": None}}))
        self.assertIsNone(AutoGain.from_config({"audio": {**AUDIO, "auto_gain": {"enabled": False, "max_gain": 8}}}))
        on = AutoGain.from_config({"audio": {**AUDIO, "auto_gain": {"enabled": True, "max_gain": 8, "target_peak": 20000}}})
        self.assertEqual((on.max_gain, on.target_peak), (8.0, 20000.0))


class AutoGainWiringTests(unittest.IsolatedAsyncioTestCase):
    CONFIG = {
        "network": {"tcp_port": 3001, "udp_port": 3000},
        "device": {"default_volume": 16, "default_mic_timeout": 60, "led_fade": 2,
                   "led_power": 50, "led_update_period": 0.25},
        "audio": AUDIO, "vad": {"threshold": 0.5}, "asr": {"url": "unused"},
        "conversation": {"backend": "conversational"},
    }

    def device(self, auto_gain):
        dev = SimpleNamespace(
            hostname="onju-000001", ip="127.0.0.1", ptt=False,
            processing=False, interrupted=asyncio.Event(), vad_writer=None,
            led_power=0, led_update_time=0,
            wake_gate=Mock(active=False, just_activated=False),
            vad=Mock(pre_buffer=deque(), speech_prob=0, recording=False),
            observe_audio=Mock())
        if auto_gain is not None:
            dev.auto_gain = auto_gain
        dev.wake_gate.feed.return_value = []
        return dev

    async def run_listener(self, dev, packet):
        loop = asyncio.get_running_loop()
        with patch("pipeline.main.socket.socket"), patch.object(loop, "sock_recvfrom",
                AsyncMock(side_effect=[(packet, (dev.ip, 1234))] * 3 + [asyncio.CancelledError()])):
            with self.assertRaises(asyncio.CancelledError):
                await udp_listener(self.CONFIG, SimpleNamespace(get_by_ip=lambda ip: dev), asyncio.Queue())

    async def test_wake_gate_hears_raised_audio_and_telemetry_stays_raw(self):
        packet = bytes([0xCF] * 256 + [0x4F] * 256)   # u-law for a quiet +/-244 square
        raw = decode_ulaw(packet)
        dev = self.device(AutoGain(target_peak=24000, max_gain=16))
        await self.run_listener(dev, packet)
        np.testing.assert_array_equal(dev.observe_audio.call_args.args[0], raw)
        self.assertEqual(dev.wake_gate.feed.call_count, 3)   # one frame out per frame in
        fed = dev.wake_gate.feed.call_args.args[0]
        self.assertEqual((fed.shape, fed.dtype), ((512,), np.int16))
        self.assertEqual(int(np.max(np.abs(fed))), int(np.max(np.abs(raw))) * 16)

    async def test_without_auto_gain_the_gate_hears_the_raw_frame(self):
        packet = bytes([0xCF] * 256 + [0x4F] * 256)
        dev = self.device(None)
        await self.run_listener(dev, packet)
        np.testing.assert_array_equal(dev.wake_gate.feed.call_args.args[0], decode_ulaw(packet))

    async def test_push_to_talk_audio_is_buffered_raw(self):
        packet = bytes([0xCF] * 256 + [0x4F] * 256)
        dev = self.device(None)
        dev.ptt, dev.wake_gate, dev.ptt_buffer = True, None, []
        await self.run_listener(dev, packet)
        np.testing.assert_array_equal(dev.ptt_buffer[0], decode_ulaw(packet))


class AutoGainDeviceTests(unittest.TestCase):
    def make(self, ptt, enabled):
        config = {"tts": {}, "audio": {**AUDIO, "auto_gain": {"enabled": enabled}},
                  "vad": {"threshold": 0.5, "neg_threshold": 0.35, "silence_time": 1.5, "pre_buffer_s": 1.0}}
        with patch("pipeline.device.VAD", Mock()):
            return Device("onju-000001", "127.0.0.1", config, conversation=Mock(), ptt=ptt)

    def test_only_voice_activated_devices_get_it_and_the_gain_is_reported(self):
        self.assertIsNone(self.make(ptt=False, enabled=False).auto_gain)
        self.assertIsNone(self.make(ptt=True, enabled=True).auto_gain)
        self.assertIsNone(self.make(ptt=False, enabled=False).to_dict()["listening"]["auto_gain"])
        device = self.make(ptt=False, enabled=True)
        for _ in range(3):
            device.auto_gain.process(tone(3000))
        self.assertEqual(device.to_dict()["listening"]["auto_gain"], 8.0)


if __name__ == "__main__":
    unittest.main()
