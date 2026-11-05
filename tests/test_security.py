import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from pipeline.conversation.conversational import ConversationalBackend
from pipeline.device import DeviceManager
from pipeline.main import multicast_listener
from pipeline.main import warmup
from pipeline.protocol import send_audio
from pipeline.security import validate_device, validate_device_name


NAME = "onju-A1B2C3"
IP = "192.168.1.50"
CONFIG = {"network": {"allowed_devices": {NAME: IP}, "multicast_group": "239.0.0.1",
                      "multicast_port": 12345}}


class DeviceBoundaryTests(unittest.TestCase):
    def test_exact_identity_pair_required(self):
        validate_device(CONFIG, NAME, IP)
        for name, ip in [("../../outside", IP), ("onju-FFFFFF", IP), (NAME, "192.168.1.51"),
                         (NAME, "127.0.0.1"), (NAME, 123), (NAME + "\n", IP)]:
            with self.subTest(name=name, ip=ip), self.assertRaises(ValueError):
                DeviceManager(CONFIG).create_device(name, ip, ptt=True)

    def test_no_allowlist_fails_closed(self):
        with self.assertRaises(ValueError):
            validate_device({}, NAME, IP)

    def test_persistence_rejects_traversal_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = {"base_url": "http://127.0.0.1:8080/v1", "api_key": "none",
                   "system_prompt": "test", "persist_dir": directory}
            for name in ["../outside", "/tmp/outside", "onju-AAAAAA/../../x", "onju-AAAAAA\n"]:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    ConversationalBackend(cfg, name)
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertEqual(validate_device_name(NAME), NAME)


class TransportBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_warmup_uses_valid_identity_without_persisting(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"conversation": {"backend": "conversational", "conversational": {
                "base_url": "http://127.0.0.1:8080/v1", "api_key": "none",
                "system_prompt": "test", "persist_dir": directory}}}
            with patch.object(ConversationalBackend, "send", new_callable=AsyncMock,
                              return_value="Hello"), \
                    patch("pipeline.main.tts.synthesize", new_callable=AsyncMock,
                          return_value=b"\0" * 6400), \
                    self.assertLogs("pipeline.main", level="INFO") as logs:
                await warmup(config)
            self.assertTrue(any("LLM  warmup OK" in line for line in logs.output))
            self.assertEqual(list(Path(directory).iterdir()), [])

    async def test_multicast_rejects_bad_name_and_unapproved_ip(self):
        manager = MagicMock()
        manager.create_device.side_effect = lambda name, ip, ptt: validate_device(CONFIG, name, ip)
        packets = [(b"../../escape abc", (IP, 12345)),
                   (NAME.encode() + b" abc", ("192.168.1.51", 12345)),
                   (NAME.encode() + b" abc PTT", (IP, 12345))]
        loop = asyncio.get_running_loop()
        receiver = AsyncMock(side_effect=packets + [asyncio.CancelledError()])
        with patch("pipeline.main.socket.socket"), patch.object(loop, "sock_recvfrom", receiver), \
                patch("pipeline.main.greet_device", new_callable=AsyncMock) as greet:
            with self.assertRaises(asyncio.CancelledError):
                await multicast_listener(CONFIG, manager)
        greet.assert_awaited_once()

    async def test_volume_cap_applies_to_transmitted_header(self):
        received = []
        ready = asyncio.Event()

        async def accept(reader, writer):
            received.append(await reader.readexactly(6))
            writer.close()
            await writer.wait_closed()
            ready.set()

        server = await asyncio.start_server(accept, "127.0.0.1", 0)
        try:
            port = server.sockets[0].getsockname()[1]
            for requested, expected in [(255, 16), (-1, 0), (15, 15)]:
                ready.clear()
                await send_audio("127.0.0.1", port, b"", volume=requested)
                await asyncio.wait_for(ready.wait(), 2)
                self.assertEqual(received[-1][3], expected)
        finally:
            server.close()
            await server.wait_closed()


if __name__ == "__main__":
    unittest.main()
