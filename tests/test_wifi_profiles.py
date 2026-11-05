import subprocess
import tempfile
import unittest
from pathlib import Path


class FirmwarePolicyTests(unittest.TestCase):
    def test_native_profiles_and_portal_policy_under_sanitizers(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            binary = str(Path(directory) / "wifi-profiles-test")
            subprocess.run(["clang++", "-std=c++11", "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=address,undefined", str(root / "tests/wifi_profiles_test.cpp"),
                            "-o", binary], check=True, capture_output=True, text=True)
            result = subprocess.run([binary], check=True, capture_output=True, text=True)
            self.assertIn("PASS:", result.stdout)


if __name__ == "__main__":
    unittest.main()
