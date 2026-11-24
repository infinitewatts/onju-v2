import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class FirmwareAudioTests(unittest.TestCase):
    def test_mulaw_encoder_keeps_full_range_under_sanitizers(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = str(Path(directory) / "ulaw-test")
            subprocess.run(["clang++", "-std=c++11", "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                            "-I", str(ROOT / "tests/arduino_stub"),
                            str(ROOT / "tests/ulaw_test.cpp"), "-o", binary],
                           check=True, capture_output=True, text=True)
            result = subprocess.run([binary], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS:", result.stdout)

    def test_encoder_matches_pipeline_decoder(self):
        import numpy as np
        from pipeline.audio import decode_ulaw
        header = (ROOT / "onjuino/audio_compression.h").read_text()
        table = header[header.index("exp_lut[256] = {") + 16:header.index("};")]
        firmware = np.array([int(v) for v in table.replace("\n", "").split(",") if v.strip()], dtype=np.int16)
        decoded = decode_ulaw(bytes(range(256)))
        self.assertTrue(np.array_equal(firmware, decoded))

    def test_radio_stays_awake_and_mic_saturates(self):
        wifi = (ROOT / "onjuino/wifi_setup.h").read_text()
        connect = wifi[wifi.index("void connectDeviceWiFi()"):]
        self.assertEqual(connect.count("WiFi.setSleep(false);"), 2)
        sketch = (ROOT / "onjuino/onjuino.ino").read_text()
        self.assertNotIn("static_cast<int16_t>(micBuffer[i] >> 14)", sketch)
        self.assertIn("mic_combine(&micCombiner, micBuffer", sketch)

    def test_two_microphone_combiner_under_sanitizers(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = str(Path(directory) / "mic-combine-test")
            subprocess.run(["clang++", "-std=c++11", "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                            str(ROOT / "tests/mic_combine_test.cpp"), "-o", binary],
                           check=True, capture_output=True, text=True)
            result = subprocess.run([binary], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS:", result.stdout)

    def test_mic_stream_stats_under_sanitizers(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = str(Path(directory) / "mic-stream-stats-test")
            subprocess.run(["clang++", "-std=c++11", "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                            str(ROOT / "tests/mic_stream_stats_test.cpp"), "-o", binary],
                           check=True, capture_output=True, text=True)
            result = subprocess.run([binary], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS:", result.stdout)

    def test_speaker_writes_fill_both_bus_slots(self):
        sketch = (ROOT / "onjuino/onjuino.ino").read_text()
        self.assertIn("I2S_CHANNEL_FMT_RIGHT_LEFT", sketch)
        # Every speaker sample write goes through the mono-to-stereo helper.
        self.assertEqual(sketch.count("i2s_write(I2S_NUM, (uint8_t *)wavData"), 0)
        self.assertEqual(sketch.count("i2sWriteMono(wavData, totalSamplesRead);"), 2)


if __name__ == "__main__":
    unittest.main()
