"""Synthetic PCM only: no microphone, network, or downloaded model required."""
import builtins
from collections import deque
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from pipeline.wakeword import (ENGLISH_KEYWORD, KEYWORD, KEYWORD_PHONES,
                               SAMPLE_RATE, WakeWordGate, _PhaseDetector)

WAKE = 20000


class FakeStream:
    def __init__(self):
        self.pending = deque()
        self.accepted = []
        self.result = ""

    def accept_waveform(self, rate, audio):
        assert rate == SAMPLE_RATE
        assert audio.dtype == np.float32
        self.pending.append(audio)
        self.accepted.append(audio)


class FakeDetector:
    def __init__(self, result=KEYWORD):
        self.streams = []
        self.result = result

    def create_stream(self):
        stream = FakeStream()
        self.streams.append(stream)
        return stream

    def is_ready(self, stream):
        return bool(stream.pending)

    def decode_stream(self, stream):
        frame = stream.pending.popleft()
        stream.result = self.result if frame[0] == np.float32(WAKE / 32768.0) else ""

    def get_result(self, stream):
        return stream.result


def frame(value=10, samples=512):
    return np.full(samples, value, dtype=np.int16)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.detector = FakeDetector()
        self.gate = WakeWordGate({}, detector=self.detector)

    def test_diagnostics_distinguish_feed_gaps_from_explicit_resets(self):
        self.gate.feed(frame(), 0)
        self.gate.feed(frame(), 2)
        self.gate.feed(frame(WAKE), 2.032)
        self.gate.reset()
        self.assertEqual(self.gate.frames_seen, 3)
        self.assertEqual(self.gate.detections, 1)
        self.assertEqual(self.gate.max_feed_gap_s, 2)
        self.assertEqual(self.gate.reset_counts['gap'], 1)
        self.assertEqual(self.gate.reset_counts['explicit'], 2)

    def test_idle_ambient_audio_never_escapes(self):
        for i in range(400):
            self.assertEqual(self.gate.feed(frame(), i * .032), [])
            self.assertFalse(self.gate.active)
            self.assertFalse(self.gate.just_activated)
        self.assertLessEqual(self.gate._buffered_samples, int(1.5 * SAMPLE_RATE))
        self.assertLessEqual(sum(x.size for x in self.gate._pre_buffer), 24000)

    def test_detection_returns_bounded_recent_audio_including_current_frame(self):
        gate = WakeWordGate({"pre_buffer_s": .064}, detector=self.detector)
        source = frame(123)
        gate.feed(source, 1)
        source[:] = 999  # buffer must own the earlier samples
        captured = gate.feed(frame(WAKE), 1.032)
        audio = np.concatenate(captured)
        self.assertEqual(audio.size, 1024)
        np.testing.assert_array_equal(audio[:512], frame(123))
        np.testing.assert_array_equal(audio[-512:], frame(WAKE))
        self.assertTrue(gate.active)
        self.assertTrue(gate.just_activated)
        self.assertEqual(gate._buffered_samples, 0)
        current = frame(456)
        self.assertIs(gate.feed(current, 1.064)[0], current)
        self.assertFalse(gate.just_activated)

    def test_seed_silence_never_enters_command_audio(self):
        initial = self.detector.streams[-1].accepted[0]
        self.assertEqual(initial.size, SAMPLE_RATE // 2)
        self.assertTrue(np.all(initial == 0))
        captured = self.gate.feed(frame(WAKE), 0)
        np.testing.assert_array_equal(np.concatenate(captured), frame(WAKE))

    def test_expiration_is_closed_at_exact_deadline_and_after_gap(self):
        self.gate.feed(frame(WAKE), 10)
        self.assertTrue(self.gate.feed(frame(), 39.99))
        self.assertEqual(self.gate.feed(frame(), 40), [])
        self.assertFalse(self.gate.active)
        self.assertFalse(self.gate.just_activated)
        self.assertEqual(self.gate.feed(frame(), 50), [])

    def test_default_prebuffer_returns_only_complete_vad_frames(self):
        for i in range(100):
            self.gate.feed(frame(), i * .032)
        captured = self.gate.feed(frame(WAKE), 3.2)
        self.assertEqual(len(captured), 46)
        self.assertTrue(all(chunk.shape == (512,) for chunk in captured))
        self.assertLessEqual(sum(chunk.size for chunk in captured), 24000)

    def test_reset_disarms_and_discards_stale_audio_and_detector_state(self):
        self.gate.feed(frame(222), 0)
        self.gate.feed(frame(WAKE), .032)
        previous = self.detector.streams[-1]
        self.gate.reset()
        self.assertFalse(self.gate.active)
        self.assertFalse(self.gate.just_activated)
        self.assertIsNot(self.detector.streams[-1], previous)
        self.assertEqual(self.gate.feed(frame(333), 1), [])
        captured = self.gate.feed(frame(WAKE), 1.032)
        np.testing.assert_array_equal(np.concatenate(captured),
                                      np.concatenate([frame(333), frame(WAKE)]))

    def test_arm_opens_only_bounded_followup_window_with_empty_prebuffer(self):
        self.gate.feed(frame(111), 0)
        self.gate.arm(1)
        self.assertTrue(self.gate.active)
        self.assertFalse(self.gate.just_activated)
        np.testing.assert_array_equal(self.gate.feed(frame(222), 1.1)[0], frame(222))
        self.assertEqual(self.gate.feed(frame(333), 11), [])
        self.assertFalse(self.gate.active)
        self.gate.arm(12, window_s=2)
        self.assertTrue(self.gate.feed(frame(), 13.99))
        self.assertEqual(self.gate.feed(frame(), 14), [])

    def test_idle_gap_recreates_stream_and_discards_old_prebuffer(self):
        self.gate.feed(frame(111), 0)
        count = len(self.detector.streams)
        captured = self.gate.feed(frame(WAKE), 1.01)
        self.assertEqual(len(self.detector.streams), count + 1)
        np.testing.assert_array_equal(np.concatenate(captured), frame(WAKE))

    def test_active_gap_does_not_disarm_valid_command(self):
        self.gate.feed(frame(WAKE), 0)
        count = len(self.detector.streams)
        np.testing.assert_array_equal(self.gate.feed(frame(), 5)[0], frame())
        self.assertEqual(len(self.detector.streams), count)
        self.assertTrue(self.gate.active)

    def test_long_idle_recycles_detector_features_and_retains_bounded_prebuffer(self):
        for i in range(3751):
            self.assertEqual(self.gate.feed(frame(), i * .032), [])
        self.assertEqual(len(self.detector.streams), 3)
        self.assertLessEqual(self.gate._idle_samples, 60 * SAMPLE_RATE)
        self.assertLessEqual(self.gate._buffered_samples, 24000)

    def test_unknown_keyword_never_authorizes(self):
        gate = WakeWordGate({}, detector=FakeDetector("SOMETHING_ELSE"))
        self.assertEqual(gate.feed(frame(WAKE), 0), [])
        self.assertFalse(gate.active)

    def test_zero_prebuffer_still_includes_detecting_frame(self):
        gate = WakeWordGate({"pre_buffer_s": 0}, detector=self.detector)
        self.assertEqual(gate.feed(frame(), 0), [])
        captured = gate.feed(frame(WAKE), .032)
        np.testing.assert_array_equal(np.concatenate(captured), frame(WAKE))

    def test_backward_time_disarms_instead_of_extending_window(self):
        self.gate.arm(10)
        self.assertEqual(self.gate.feed(frame(), 9), [])
        self.assertFalse(self.gate.active)

    def test_invalid_config_and_audio_are_rejected(self):
        for config in ({"pre_buffer_s": -1}, {"command_window_s": 0},
                       {"command_window_s": float("inf")}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                WakeWordGate(config, detector=self.detector)
        with self.assertRaises(ValueError):
            self.gate.feed(np.zeros(512, dtype=np.float32), 0)
        with self.assertRaises(ValueError):
            self.gate.feed(frame().reshape(1, -1), 0)
        with self.assertRaises(ValueError):
            self.gate.feed(frame(samples=448), 0)
        with self.assertRaises(ValueError):
            self.gate.feed(frame(samples=0), 0)
        with self.assertRaises(ValueError):
            self.gate.feed(frame(), float("nan"))


class ModelLoadingTests(unittest.TestCase):
    NAMES = ("tokens.txt", "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx",
             "decoder-epoch-13-avg-2-chunk-8-left-64.onnx",
             "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx")

    def test_missing_model_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                WakeWordGate({"model_dir": directory})

    def test_english_profile_uses_fixed_bpe_and_two_shared_model_states(self):
        seen = {}
        detector = FakeDetector()
        def create(**kwargs):
            seen.update(kwargs)
            seen['keyword_contents'] = Path(kwargs['keywords_file']).read_text()
            return detector
        with tempfile.TemporaryDirectory() as directory:
            for name in ('tokens.txt', 'encoder-epoch-12-avg-2-chunk-16-left-64.onnx',
                         'decoder-epoch-12-avg-2-chunk-16-left-64.onnx',
                         'joiner-epoch-12-avg-2-chunk-16-left-64.onnx'):
                (Path(directory) / name).touch()
            with patch.dict(sys.modules, {'sherpa_onnx': types.SimpleNamespace(KeywordSpotter=create)}):
                gate = WakeWordGate({'model_dir': directory, 'model_profile': 'english'})
        self.assertEqual(gate.model_profile, 'english')
        self.assertEqual(seen['keywords_score'], 1)
        self.assertEqual(seen['keyword_contents'], ENGLISH_KEYWORD)
        self.assertEqual(len(detector.streams), 2)
        self.assertEqual(detector.streams[0].accepted[0].size, 8000)
        self.assertEqual(detector.streams[1].accepted[0].size, 5120)
        self.assertEqual(detector.streams[1].accepted[1].size, 8000)

    def test_unknown_profile_fails_closed(self):
        with self.assertRaises(ValueError):
            WakeWordGate({'model_dir': '.', 'model_profile': 'typo'})

    def test_model_constructor_uses_local_cpu_and_fixed_keyword(self):
        seen = {}
        def create(**kwargs):
            seen.update(kwargs)
            seen["keyword_contents"] = Path(kwargs["keywords_file"]).read_text()
            return FakeDetector()
        with tempfile.TemporaryDirectory() as directory:
            for name in self.NAMES:
                (Path(directory) / name).touch()
            with patch.dict(sys.modules, {"sherpa_onnx": types.SimpleNamespace(KeywordSpotter=create)}):
                WakeWordGate({"model_dir": directory})
        self.assertEqual(seen["num_threads"], 1)
        self.assertEqual(seen["provider"], "cpu")
        self.assertEqual(seen["keywords_score"], 2.0)
        self.assertEqual(seen["keywords_threshold"], .25)
        self.assertEqual(seen["keyword_contents"], KEYWORD_PHONES)
        self.assertFalse(Path(seen["keywords_file"]).exists())

    def test_missing_optional_dependency_fails_closed(self):
        real_import = builtins.__import__
        def importing(name, *args, **kwargs):
            if name == "sherpa_onnx":
                raise ImportError("sherpa unavailable")
            return real_import(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            for name in self.NAMES:
                (Path(directory) / name).touch()
            with patch("builtins.__import__", side_effect=importing):
                with self.assertRaises(ImportError):
                    WakeWordGate({"model_dir": directory})


class PhaseStreamTests(unittest.TestCase):
    def test_consuming_results_from_second_phase_open_one_window(self):
        class ConsumingDetector(FakeDetector):
            def get_result(self, stream):
                result, stream.result = stream.result, ''
                return result if self.streams.index(stream) % 2 else ''
        detector = ConsumingDetector()
        gate = WakeWordGate({'model_profile': 'english'}, detector=_PhaseDetector(detector))
        output = gate.feed(frame(WAKE), 0)
        self.assertTrue(gate.just_activated)
        self.assertEqual(gate.detections, 1)
        # Only actual microphone samples enter ASR, never either seed silence.
        np.testing.assert_array_equal(np.concatenate(output), frame(WAKE))
        gate.feed(frame(WAKE), .032)
        self.assertEqual(gate.detections, 1)
        self.assertFalse(gate.just_activated)
        gate.reset()
        self.assertEqual(len(detector.streams), 4)
        self.assertEqual(detector.streams[3].accepted[0].size, 5120)
        gate.feed(frame(WAKE), 1)
        self.assertEqual(gate.detections, 2)

    def test_ambient_gap_and_arm_recreate_both_phases_without_admitting_noise(self):
        detector = FakeDetector()
        gate = WakeWordGate({'model_profile': 'english'}, detector=_PhaseDetector(detector))
        self.assertEqual(gate.feed(frame(), 0), [])
        self.assertEqual(gate.feed(frame(), 2), [])
        self.assertEqual(len(detector.streams), 4)
        self.assertEqual(gate.reset_counts['gap'], 1)
        gate.arm(3)
        self.assertEqual(len(detector.streams), 6)
        self.assertEqual(detector.streams[-1].accepted[0].size, 5120)
        self.assertEqual(gate.detections, 0)


if __name__ == "__main__":
    unittest.main()
