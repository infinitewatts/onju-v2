"""keep_warm: periodic tiny requests so the speech models stay loaded."""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pipeline.main import keep_warm

CONFIG = {
    "audio": {"sample_rate": 16000},
    "asr": {"url": "http://unused"},
    "tts": {"backend": "local"},
    "keep_warm": {"interval_s": 120},
}


class Stop(Exception):
    pass


def sleeper(rounds, seen):
    async def sleep(seconds):
        seen.append(seconds)
        if len(seen) > rounds:
            raise Stop
    return sleep


class KeepWarmTests(unittest.IsolatedAsyncioTestCase):
    async def run_rounds(self, config, devices, rounds=2, asr=None, tts=None):
        manager = SimpleNamespace(devices=devices)
        asr = asr or AsyncMock()
        tts = tts or AsyncMock()
        seen = []
        with patch("pipeline.main._warm_asr", asr), patch("pipeline.main._warm_tts", tts):
            try:
                await keep_warm(config, manager, sleep=sleeper(rounds, seen))
            except Stop:
                pass
        return asr, tts, seen

    async def test_off_by_default(self):
        config = {k: v for k, v in CONFIG.items() if k != "keep_warm"}
        asr, tts, seen = await self.run_rounds(config, {})
        self.assertEqual((asr.await_count, tts.await_count, seen), (0, 0, []))

    async def test_pings_both_models_each_interval(self):
        idle = SimpleNamespace(processing=False)
        asr, tts, seen = await self.run_rounds(CONFIG, {"a": idle})
        self.assertEqual((asr.await_count, tts.await_count), (2, 2))
        self.assertEqual(seen, [120.0, 120.0, 120.0])

    async def test_skips_while_a_turn_is_in_progress(self):
        busy = SimpleNamespace(processing=True)
        asr, tts, _ = await self.run_rounds(CONFIG, {"a": busy})
        self.assertEqual((asr.await_count, tts.await_count), (0, 0))

    async def test_skips_while_someone_is_speaking_then_resumes(self):
        dev = SimpleNamespace(processing=False, wake_gate=SimpleNamespace(active=True))
        manager_devices = {"a": dev}
        calls = []

        async def asr(config):
            calls.append("asr")

        seen = []

        async def sleep(seconds):
            seen.append(seconds)
            if len(seen) == 2:
                dev.wake_gate.active = False  # the turn is over
            if len(seen) > 2:
                raise Stop

        with patch("pipeline.main._warm_asr", asr), patch("pipeline.main._warm_tts", AsyncMock()):
            with self.assertRaises(Stop):
                await keep_warm(CONFIG, SimpleNamespace(devices=manager_devices), sleep=sleep)
        self.assertEqual(calls, ["asr"])

    async def test_bad_or_empty_settings_turn_it_off(self):
        for value in (None, 5, {"interval_s": "abc"}, {"interval_s": 0}, {"interval_s": float("nan")}):
            asr, tts, seen = await self.run_rounds({**CONFIG, "keep_warm": value}, {})
            self.assertEqual((asr.await_count, seen), (0, []), value)

    async def test_interval_has_a_floor(self):
        _, _, seen = await self.run_rounds({**CONFIG, "keep_warm": {"interval_s": 1}}, {}, rounds=0)
        self.assertEqual(seen, [10.0])

    async def test_remote_tts_is_not_pinged(self):
        config = {**CONFIG, "tts": {"backend": "elevenlabs"}}
        asr, tts, _ = await self.run_rounds(config, {})
        self.assertEqual((asr.await_count, tts.await_count), (2, 0))

    async def test_a_failing_ping_does_not_stop_the_loop(self):
        asr, tts, _ = await self.run_rounds(CONFIG, {}, asr=AsyncMock(side_effect=OSError("down")))
        self.assertEqual((asr.await_count, tts.await_count), (2, 2))


if __name__ == "__main__":
    unittest.main()
