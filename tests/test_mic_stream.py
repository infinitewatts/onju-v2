"""Mic stream counters: no microphone, model or network is used."""
import asyncio
import struct
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from pipeline.audio import decode_ulaw
from pipeline.main import udp_listener
from pipeline.mic_stream import MicStreamMonitor, parse_datagram, REPORT_SIZE

CONFIG = {
    "network": {"tcp_port": 3001, "udp_port": 3000},
    "device": {"default_volume": 16, "default_mic_timeout": 60, "led_fade": 2,
               "led_power": 50, "led_update_period": 0.25},
    "audio": {"sample_rate": 16000, "chunk_size": 512, "opus_frame_size": 320},
    "vad": {"threshold": 0.5}, "asr": {"url": "unused"},
    "conversation": {"backend": "conversational"},
}
AUDIO = bytes(range(256)) * 2


def packet(seq, audio=AUDIO):
    return audio + bytes([0xA5, 0x01, seq >> 8, seq & 0xFF])


def report(seq=7, rssi=-60):
    return struct.pack(">4sBHIIIHhbBB", b"OJST", 1, seq, 1, 2, 3, 4, -5, rssi, 6, 0xAB)


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class ParseTests(unittest.TestCase):
    def test_three_kinds_and_garbage(self):
        self.assertEqual(parse_datagram(AUDIO), ("audio", AUDIO, None))
        self.assertEqual(parse_datagram(packet(0x0102)), ("audio", AUDIO, 0x0102))
        kind, rep = parse_datagram(report())[:2]
        self.assertEqual(kind, "report")
        self.assertEqual(rep, {"seq": 7, "send_fail": 1, "send_retry": 2, "late_reads": 3,
                               "max_read_gap_ms": 4, "last_errno": -5, "rssi": -60,
                               "channel": 6, "bssid_last": 0xAB})
        self.assertEqual(REPORT_SIZE, 26)
        self.assertIsNone(parse_datagram(b""))
        self.assertIsNone(parse_datagram(b"x" * 100))
        self.assertIsNone(parse_datagram(AUDIO + b"\x00\x01\x00\x00"))  # bad magic
        self.assertIsNone(parse_datagram(AUDIO + b"\xA5\x02\x00\x00"))  # bad version
        self.assertIsNone(parse_datagram(b"XXXX" + report()[4:]))
        self.assertIsNone(parse_datagram(report()[:-1]))


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.m = MicStreamMonitor(clock=self.clock)

    def feed(self, seqs):
        for s in seqs:
            self.clock.t += 0.032
            self.m.observe(s)

    def test_no_loss(self):
        self.feed(range(10))
        self.assertEqual((self.m.seq_packets, self.m.seq_gaps, self.m.seq_lost), (10, 0, 0))

    def test_legacy_ignored(self):
        self.m.observe(None)
        self.assertEqual(self.m.seq_packets, 0)

    def test_gaps_and_buckets(self):
        self.feed([0, 2, 5, 10, 20, 40])  # lost 1, 2, 4, 9, 19
        self.assertEqual((self.m.seq_gaps, self.m.seq_lost), (5, 35))
        self.assertEqual(self.m.gap_histogram, {"1": 1, "2-3": 1, "4-7": 1, "8-15": 1, "16+": 1})

    def test_wraparound(self):
        self.feed([65534, 65535, 0, 1])
        self.assertEqual(self.m.seq_gaps, 0)
        self.feed([4])
        self.assertEqual((self.m.seq_gaps, self.m.seq_lost), (1, 2))
        self.feed([3])             # one packet arriving late
        self.assertEqual((self.m.seq_reordered, self.m.seq_lost), (1, 2))
        self.feed([5])             # the stream carries on where it was
        self.assertEqual((self.m.seq_gaps, self.m.seq_lost), (1, 2))

    def test_gap_across_wrap(self):
        self.feed([65534, 1])
        self.assertEqual(self.m.seq_lost, 2)

    def test_silence_without_a_gap_is_not_loss(self):
        # Playback, mute and reconnects stop capture; numbering resumes where it stopped.
        self.feed([0, 1])
        self.clock.t += 30
        self.feed([2, 3])
        self.assertEqual((self.m.seq_gaps, self.m.seq_lost, self.m.seq_resets), (0, 0, 0))

    def test_outage_while_capturing_is_counted_after_silence(self):
        self.feed([0, 1])
        self.clock.t += 5
        self.feed([150, 151])
        self.assertEqual((self.m.seq_gaps, self.m.seq_lost), (1, 148))
        self.assertEqual(self.m.gap_histogram["16+"], 1)

    def test_device_restart_resets_without_loss_and_keeps_tracking(self):
        self.feed([500, 501])
        self.clock.t += 8            # reboot takes seconds
        self.feed([0, 1, 2, 4])      # numbering began again; then one real loss
        self.assertEqual((self.m.seq_resets, self.m.seq_reordered), (1, 0))
        self.assertEqual((self.m.seq_gaps, self.m.seq_lost), (1, 1))

    def test_huge_gap_resets(self):
        self.feed([0, 3001 + 1])
        self.assertEqual((self.m.seq_gaps, self.m.seq_lost), (0, 0))
        self.feed([3003])
        self.assertEqual(self.m.seq_lost, 0)

    def test_gap_of_3000_counts(self):
        self.feed([0, 3001])
        self.assertEqual(self.m.seq_lost, 3000)

    def test_reorder_and_duplicate(self):
        self.feed([0, 1, 2, 4, 3, 4, 5])
        self.assertEqual((self.m.seq_gaps, self.m.seq_lost), (1, 1))
        self.assertEqual(self.m.seq_reordered, 2)  # 3 is late, 4 is a duplicate

    def test_report_and_age(self):
        self.m.observe_report({"rssi": -50})
        self.clock.t += 3
        d = self.m.to_dict()
        self.assertEqual(d["device_report"], {"rssi": -50})
        self.assertEqual(d["device_report_age_s"], 3.0)
        self.assertIsNone(MicStreamMonitor().to_dict()["device_report"])


class ListenerTests(unittest.IsolatedAsyncioTestCase):
    def device(self):
        return SimpleNamespace(
            hostname="onju-000001", ip="127.0.0.1", ptt=False,
            processing=False, interrupted=asyncio.Event(), vad_writer=None,
            led_power=0, led_update_time=0,
            wake_gate=Mock(active=False, just_activated=False),
            vad=Mock(pre_buffer=deque(), speech_prob=0, recording=False),
            conversation=SimpleNamespace(stream=Mock(), commit=Mock()),
            last_user_text=None, last_response=None,
            observe_audio=Mock(), mic_stream=MicStreamMonitor(),
        )

    async def run_listener(self, dev, datagrams):
        dev.wake_gate.feed.return_value = []
        loop = asyncio.get_running_loop()
        packets = [(d, (dev.ip, 1234)) for d in datagrams]
        with patch("pipeline.main.socket.socket"), patch.object(loop, "sock_recvfrom",
                AsyncMock(side_effect=packets + [asyncio.CancelledError()])):
            with self.assertRaises(asyncio.CancelledError):
                await udp_listener(CONFIG, SimpleNamespace(get_by_ip=lambda ip: dev), asyncio.Queue())

    async def test_trailer_packet_reaches_wake_gate_as_same_samples(self):
        legacy, new = self.device(), self.device()
        await self.run_listener(legacy, [AUDIO])
        await self.run_listener(new, [packet(9)])
        np.testing.assert_array_equal(legacy.wake_gate.feed.call_args[0][0],
                                      new.wake_gate.feed.call_args[0][0])
        np.testing.assert_array_equal(new.wake_gate.feed.call_args[0][0], decode_ulaw(AUDIO))
        self.assertEqual(new.mic_stream.seq_packets, 1)
        self.assertEqual(legacy.mic_stream.seq_packets, 0)

    async def test_gap_is_counted_by_listener(self):
        dev = self.device()
        await self.run_listener(dev, [packet(1), packet(2), packet(6)])
        self.assertEqual((dev.mic_stream.seq_gaps, dev.mic_stream.seq_lost), (1, 3))
        self.assertEqual(dev.wake_gate.feed.call_count, 3)

    async def test_stats_packet_reaches_no_audio_path(self):
        dev = self.device()
        await self.run_listener(dev, [report()])
        dev.observe_audio.assert_not_called()
        dev.wake_gate.feed.assert_not_called()
        self.assertEqual(dev.mic_stream.report["rssi"], -60)

    async def test_garbage_dropped_and_device_without_monitor_works(self):
        dev = self.device()
        del dev.mic_stream
        await self.run_listener(dev, [b"x" * 100, packet(1), report()])
        self.assertEqual(dev.wake_gate.feed.call_count, 1)


if __name__ == "__main__":
    unittest.main()
