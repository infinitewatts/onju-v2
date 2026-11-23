import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pipeline.device import Device


class DeviceDiagnosticsTests(unittest.TestCase):
    def device(self):
        with patch('pipeline.device.VAD'):
            return Device('onju-A1B2C3', '192.168.1.50', {'tts': {}}, None)

    def test_silence_audio_levels_and_freshness_without_samples(self):
        device = self.device()
        self.assertIsNone(device.to_dict()['listening']['last_audio_age_s'])
        with patch('pipeline.device.time.monotonic', return_value=10):
            device.observe_audio(np.zeros(512, dtype=np.int16))
            device.observe_audio(np.full(512, -32768, dtype=np.int16))
            device.observe_audio(np.full(512, 100, dtype=np.int16))
        with patch('pipeline.device.time.monotonic', return_value=12):
            state = device.to_dict()['listening']
        self.assertEqual(state['audio_frames'], 3)
        self.assertEqual(state['audio_rms'], 100)
        self.assertEqual(state['peak_audio_rms'], 32768)
        self.assertEqual(state['last_audio_age_s'], 2)
        self.assertFalse(any(isinstance(v, np.ndarray) for v in vars(device).values()))

    def test_prior_ten_second_frame_count_exposes_lost_audio(self):
        device = self.device()
        self.assertEqual(device.to_dict()['listening']['audio_frames_prior_10s'], 0)
        frame = np.zeros(512, dtype=np.int16)
        for moment in [100.0] * 280 + [110.5] * 5:
            with patch('pipeline.device.time.monotonic', return_value=moment):
                device.observe_audio(frame)
        with patch('pipeline.device.time.monotonic', return_value=111):
            self.assertEqual(device.to_dict()['listening']['audio_frames_prior_10s'], 280)
        with patch('pipeline.device.time.monotonic', return_value=125):
            self.assertEqual(device.to_dict()['listening']['audio_frames_prior_10s'], 0)

    def test_miss_labels_are_reported(self):
        device = self.device()
        device.wake_gate = SimpleNamespace(active=False, miss_counts={'no_text': 2})
        self.assertEqual(device.to_dict()['listening']['wake_misses'], {'no_text': 2})

    def test_processing_and_gate_state_are_distinct(self):
        device = self.device()
        device.processing = True
        device.queued_utterances = 2
        device.wake_gate = SimpleNamespace(active=False)
        state = device.to_dict()['listening']
        self.assertTrue(state['processing'])
        self.assertEqual(state['queued_utterances'], 2)
        self.assertTrue(state['wake_enabled'])
        self.assertFalse(state['wake_active'])


if __name__ == '__main__':
    unittest.main()
