"""Synthetic loopback packets only; no device or real microphone recording."""
import socket
import stat
import subprocess
import sys
import tempfile
import time
import unittest
import wave
from pathlib import Path


class MicrophoneReceiverTests(unittest.TestCase):
    def run_receiver(self, allowed, payload):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "synthetic.wav"
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as chooser:
                chooser.bind(("127.0.0.1", 0))
                port = chooser.getsockname()[1]
            script = Path(__file__).with_name("test_mic.py")
            child = subprocess.Popen([sys.executable, "-u", str(script), "--port", str(port),
                "--device-ip", allowed, "--duration", "1", "--wait-timeout", "1",
                "--output", str(output)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                self.assertIn("Listening", child.stdout.readline())
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                    for _ in range(40):
                        sender.sendto(payload, ("127.0.0.1", port))
                        time.sleep(0.032)
                stdout, stderr = child.communicate(timeout=5)
                if output.exists():
                    self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
                    with wave.open(str(output)) as f:
                        self.assertGreater(f.getnframes(), 0)
                        self.assertEqual((f.getnchannels(), f.getframerate()), (1, 16000))
                return child.returncode, output.exists(), stdout + stderr
            finally:
                if child.poll() is None:
                    child.kill()
                    child.communicate()

    def test_unknown_source_is_ignored_and_wait_is_bounded(self):
        code, exists, output = self.run_receiver("192.168.1.250", b"\xff" * 512)
        self.assertEqual(code, 1)
        self.assertFalse(exists)
        self.assertIn("No audio received", output)

    def test_invalid_packets_do_not_start_recording(self):
        code, exists, output = self.run_receiver("127.0.0.1", b"bad")
        self.assertEqual(code, 1)
        self.assertFalse(exists)

    def test_expected_device_creates_private_wav(self):
        code, exists, output = self.run_receiver("127.0.0.1", b"\xff" * 512)
        self.assertEqual(code, 0, output)
        self.assertTrue(exists)

    def test_short_packet_gap_preserves_requested_capture_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "gap.wav"
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as chooser:
                chooser.bind(("127.0.0.1", 0))
                port = chooser.getsockname()[1]
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).with_name("test_mic.py")),
                "--port", str(port), "--device-ip", "127.0.0.1", "--duration", "3",
                "--wait-timeout", "2", "--output", str(output)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                self.assertIn("Listening", child.stdout.readline())
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                    sender.sendto(b"\xff" * 512, ("127.0.0.1", port))
                    time.sleep(1.3)
                    sender.sendto(b"\xff" * 512, ("127.0.0.1", port))
                child.communicate(timeout=6)
                self.assertEqual(child.returncode, 0)
                with wave.open(str(output)) as f:
                    self.assertEqual(f.getnframes(), 1024)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.communicate()


if __name__ == "__main__":
    unittest.main()
