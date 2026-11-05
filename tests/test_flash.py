"""Exercise flash.sh guards with a fake CLI; never open a physical serial port."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class FlashGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copy(Path(__file__).resolve().parents[1] / "flash.sh", self.root)
        sketch = self.root / "onjuino"
        sketch.mkdir()
        (sketch / "onjuino.ino").write_text("void setup() {} void loop() {}")
        (sketch / "credentials.h").write_text('#define WIFI_SSID "ONJU_COMPILE_ONLY"\n')
        (sketch / "ota_credentials.h").write_text('#define OTA_PASSWORD_HASH "' + 'a' * 64 + '"\n')
        commands = self.root / "bin"
        commands.mkdir()
        cli = commands / "arduino-cli"
        cli.write_text('''#!/bin/sh
printf '%s\\n' "$*" >> "$MOCK_CALLS"
if [ "$1 $2" = 'lib list' ]; then
    printf 'Adafruit NeoPixel\\nesp32_opus\\n'
elif [ "$1" = compile ]; then
    while [ "$#" -gt 0 ]; do
        if [ "$1" = --build-path ]; then
            shift
            mkdir -p "$1"
            printf synthetic > "$1/onjuino.ino.bin"
        fi
        shift
    done
elif [ "$1" = upload ]; then
    echo 'A guarded test must never reach upload' >&2
    exit 99
fi
''')
        cli.chmod(0o700)
        self.env = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ["PATH"],
                        MOCK_CALLS=str(self.root / "calls"), ONJU_FLASH_SIZE="16M")

    def invoke(self, *args):
        return subprocess.run(["/bin/bash", str(self.root / "flash.sh"), *args],
                              env=self.env, capture_output=True, text=True)

    def calls(self):
        return (self.root / "calls").read_text()

    def test_compile_only_and_partition_cache_change(self):
        result = self.invoke("compile")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("FlashSize=16M,PartitionScheme=app3M_fat9M_16MB", self.calls())
        self.assertNotIn("upload", self.calls())
        self.env["ONJU_FLASH_SIZE"] = "4M"
        result = self.invoke("compile")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("FlashSize=4M,PartitionScheme=default", self.calls())

    def test_placeholder_prevents_upload(self):
        result = self.invoke("/dev/cu.synthetic")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("compile-only WiFi settings", result.stdout)
        self.assertNotIn("upload", self.calls())

    def test_invalid_ota_verifier_prevents_upload(self):
        (self.root / "onjuino/credentials.h").write_text('#define WIFI_SSID "synthetic"\n')
        (self.root / "onjuino/ota_credentials.h").write_text('#define OTA_PASSWORD_HASH "' + 'z' * 64 + '"\n')
        result = self.invoke("/dev/cu.synthetic")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Invalid OTA verifier", result.stderr)
        self.assertNotIn("upload", self.calls())


if __name__ == "__main__":
    unittest.main()
