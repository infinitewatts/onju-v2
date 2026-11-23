"""Fixed local connection choices only; no live speaker, model, or agent."""
import asyncio
from collections import deque
import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from pipeline.main import control_server, process_utterances, udp_listener
from pipeline.voice_pilot import (CLARIFICATION, COMPLETE, CONFIRMATIONS, INACTIVE,
                                  MAX_RESULTS, QUESTION, PilotError, VoicePilot)

HOST = "orchestrator-session-1"
DEVICE = "onju-000001"
PAYLOAD = {"id": "pilot-1", "host_session_id": HOST, "device": DEVICE}
CONFIG = {
    "voice_pilot": {"enabled": True, "host_session_id": HOST},
    "network": {"tcp_port": 3001, "control_port": 3012,
                "allowed_devices": {DEVICE: "127.0.0.1"}},
    "device": {"default_volume": 16, "default_mic_timeout": 60, "led_fade": 2},
    "audio": {"sample_rate": 16000, "opus_frame_size": 320},
    "asr": {"url": "unused"},
    "conversation": {"backend": "conversational"},
}


def device():
    return SimpleNamespace(
        hostname=DEVICE, ip="127.0.0.1", voice="aiden", ptt=False, ptt_buffer=[],
        processing=False, interrupted=asyncio.Event(), vad_writer=None,
        wake_gate=Mock(active=False),
        vad=SimpleNamespace(recording=False, reset=Mock(), pre_buffer=deque()),
        conversation=SimpleNamespace(stream=Mock(), commit=Mock()),
        last_user_text=None, last_response=None,
        observe_audio=Mock(),
    )


class StateTests(unittest.TestCase):
    def setUp(self):
        self.now = 0
        self.pilot = VoicePilot(CONFIG["voice_pilot"], clock=lambda: self.now)

    def waiting(self):
        self.pilot.submit(PAYLOAD, DEVICE, busy=False)
        self.pilot.speaking("pilot-1")
        self.pilot.awaiting_answer("pilot-1")

    def test_strict_validation_and_target_mismatch_have_no_side_effects(self):
        invalid = ({}, {**PAYLOAD, "question": "arbitrary speech"},
                   {**PAYLOAD, "id": "../../bad"}, {**PAYLOAD, "device": [DEVICE]},
                   {**PAYLOAD, "host_session_id": "other-session"})
        for body in invalid:
            with self.subTest(body=body), self.assertRaises(PilotError):
                self.pilot.submit(body, DEVICE, busy=False)
        with self.assertRaises(PilotError):
            self.pilot.submit(PAYLOAD, "onju-000002", busy=False)
        self.assertEqual(self.pilot.records, {})

    def test_duplicate_is_idempotent_in_every_status_and_overlap_is_rejected(self):
        initial, created = self.pilot.submit(PAYLOAD, DEVICE, busy=False)
        self.assertTrue(created)
        self.assertEqual(initial["status"], "queued")
        self.pilot.speaking("pilot-1")
        duplicate, created = self.pilot.submit(PAYLOAD, DEVICE, busy=True)
        self.assertFalse(created)
        self.assertEqual(duplicate["status"], "speaking")
        with self.assertRaises(PilotError) as error:
            self.pilot.submit({**PAYLOAD, "id": "pilot-2"}, DEVICE, busy=False)
        self.assertEqual(error.exception.status, 409)
        self.pilot.awaiting_answer("pilot-1")
        self.pilot.intercept(DEVICE, "green")
        duplicate, created = self.pilot.submit(PAYLOAD, DEVICE, busy=False)
        self.assertFalse(created)
        self.assertEqual(duplicate["choice"], 2)
        self.assertEqual(len(self.pilot.records), 1)

    def test_choice_is_fixed_and_raw_speech_is_not_retained(self):
        for text, choice in (("Hey, Aiden: option one!", 1), ("Hey Aidan, option one", 1),
                             ("hay ayden two", 2), ("one.", 1), ("1", 1),
                             ("BLUE!", 1), ("option two?", 2), ("two", 2), ("green.", 2)):
            with self.subTest(text=text):
                self.pilot.records.clear()
                self.waiting()
                self.assertEqual(self.pilot.intercept(DEVICE, text), CONFIRMATIONS[choice])
                result = self.pilot.get("pilot-1")
                self.assertEqual(result["choice"], choice)
                self.assertEqual(result["label"], "blue" if choice == 1 else "green")
                self.assertEqual(set(result), {"id", "host_session_id", "device", "status", "choice", "label"})
                self.assertEqual(self.pilot.intercept(DEVICE, "two"), COMPLETE)
                self.assertEqual(self.pilot.get("pilot-1")["choice"], choice)

    def test_unknown_or_ambiguous_speech_clarifies_without_consuming_question(self):
        self.waiting()
        for speech in ("blue or green", "send money", "blue then send money", "", "option three"):
            self.assertEqual(self.pilot.intercept(DEVICE, speech), CLARIFICATION)
            self.assertEqual(self.pilot.get("pilot-1")["status"], "awaiting_answer")
        self.assertIsNone(self.pilot.intercept("onju-000002", "one"))

    def test_expiry_and_failure_cannot_accept_stale_answers(self):
        self.waiting()
        self.now = 300
        self.assertEqual(self.pilot.intercept(DEVICE, "blue"), INACTIVE)
        self.assertEqual(self.pilot.get("pilot-1")["status"], "expired")
        self.assertNotIn("choice", self.pilot.get("pilot-1"))
        self.pilot.submit({**PAYLOAD, "id": "pilot-2"}, DEVICE, busy=False)
        self.pilot.fail("pilot-2")
        self.assertEqual(self.pilot.intercept(DEVICE, "green"), INACTIVE)
        self.assertEqual(self.pilot.get("pilot-2")["status"], "failed")
        self.assertFalse(self.pilot.can_send("pilot-2"))

    def test_capacity_rejects_new_ids_and_preserves_old_idempotency(self):
        for i in range(MAX_RESULTS):
            self.pilot.submit({**PAYLOAD, "id": f"pilot-{i}"}, DEVICE, busy=False)
            self.pilot.fail(f"pilot-{i}")
        with self.assertRaises(PilotError):
            self.pilot.submit({**PAYLOAD, "id": "overflow"}, DEVICE, busy=False)
        self.assertEqual(len(self.pilot.records), MAX_RESULTS)
        old, created = self.pilot.submit({**PAYLOAD, "id": "pilot-0"}, DEVICE, busy=False)
        self.assertFalse(created)
        self.assertEqual(old["status"], "failed")

    def test_disabled_has_no_state_and_enabled_requires_pinned_session(self):
        pilot = VoicePilot({})
        with self.assertRaises(PilotError) as error:
            pilot.submit(PAYLOAD, DEVICE, busy=False)
        self.assertEqual(error.exception.status, 404)
        self.assertIsNone(pilot.intercept(DEVICE, "blue"))
        with self.assertRaises(ValueError):
            VoicePilot({"enabled": True})


class FakeWriter:
    def __init__(self):
        self.data = bytearray()
    def write(self, data):
        self.data.extend(data)
    async def drain(self):
        pass
    def close(self):
        pass


class ControlTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.config = copy.deepcopy(CONFIG)
        self.dev = device()
        self.manager = SimpleNamespace(devices={DEVICE: self.dev})
        self.pilot = VoicePilot(self.config["voice_pilot"])
        self.started = asyncio.Event()
        self.speech_started = asyncio.Event()
        self.release = asyncio.Event()
        self.tts = AsyncMock(side_effect=self.synthesize)
        self.send = AsyncMock()

        async def start(handler, host, port):
            self.handle = handler
            self.started.set()
            return self
        self.patches = [patch("pipeline.main.asyncio.start_server", side_effect=start),
                        patch("pipeline.main.tts.synthesize", self.tts),
                        patch("pipeline.main.send_audio", self.send)]
        for item in self.patches:
            item.start()
        self.task = asyncio.create_task(control_server(self.config, self.manager, self.pilot))
        await self.started.wait()

    async def asyncTearDown(self):
        self.task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await self.task
        for item in reversed(self.patches):
            item.stop()

    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        pass
    async def serve_forever(self):
        await asyncio.Event().wait()

    async def synthesize(self, *args):
        self.speech_started.set()
        await self.release.wait()
        return b"\x00\x00" * 320

    async def request(self, method, target="/pilot/question", payload=None):
        body = json.dumps(payload).encode() if payload is not None else b""
        reader = asyncio.StreamReader()
        reader.feed_data(f"{method} {target} HTTP/1.1\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body)
        reader.feed_eof()
        writer = FakeWriter()
        await self.handle(reader, writer)
        header, raw = bytes(writer.data).split(b"\r\n\r\n", 1)
        code = int(header.split()[1])
        try:
            result = json.loads(raw)
        except ValueError:
            result = raw.decode()
        return code, result

    async def finish_prompt(self):
        self.release.set()
        for _ in range(20):
            if not self.dev.processing:
                return
            await asyncio.sleep(0)
        self.fail("Pilot prompt did not release processing")

    async def test_device_removal_awaits_local_wake_cleanup(self):
        closed = asyncio.Event()
        async def close():
            await asyncio.sleep(0)
            closed.set()
        self.dev.wake_gate = SimpleNamespace(aclose=AsyncMock(side_effect=close))
        code, result = await self.request('DELETE', '/devices', {'hostname': DEVICE})
        self.assertEqual(code, 200)
        self.assertTrue(closed.is_set())
        self.assertNotIn(DEVICE, self.manager.devices)

    async def test_post_is_accepted_duplicate_never_replays_and_get_reports_state(self):
        code, status = await self.request("POST", payload=PAYLOAD)
        self.assertEqual((code, status["status"]), (202, "queued"))
        self.assertTrue(self.dev.processing)
        await self.speech_started.wait()
        code, status = await self.request("POST", payload=PAYLOAD)
        self.assertEqual((code, status["status"]), (200, "speaking"))
        self.assertEqual(self.tts.await_count, 1)
        self.assertEqual(self.tts.call_args.args[0], QUESTION)
        await self.finish_prompt()
        code, status = await self.request("GET", "/pilot/question?id=pilot-1")
        self.assertEqual((code, status["status"]), (200, "awaiting_answer"))
        self.dev.wake_gate.arm.assert_not_called()
        self.assertEqual(self.send.call_args.kwargs["mic_timeout"], 60)

    async def test_health_validation_disabled_unknown_and_unapproved_targets_do_not_speak(self):
        code, _ = await self.request("POST", payload={})
        self.assertEqual(code, 400)
        self.pilot.enabled = False
        code, _ = await self.request("POST", payload={})
        self.assertEqual(code, 404)
        self.pilot.enabled = True
        code, _ = await self.request("POST", payload={**PAYLOAD, "device": "onju-000002"})
        self.assertEqual(code, 404)
        self.dev.ip = "127.0.0.2"
        code, _ = await self.request("POST", payload=PAYLOAD)
        self.assertEqual(code, 400)
        self.assertEqual(self.pilot.records, {})
        self.assertFalse(self.dev.processing)
        self.tts.assert_not_called()

    async def test_busy_queued_or_recording_device_is_rejected_without_reservation(self):
        for field, busy in (("processing", True), ("queued_utterances", 1)):
            setattr(self.dev, field, busy)
            code, _ = await self.request("POST", payload=PAYLOAD)
            self.assertEqual(code, 409)
            setattr(self.dev, field, False)
        self.dev.vad.recording = True
        code, _ = await self.request("POST", payload=PAYLOAD)
        self.assertEqual(code, 409)
        self.assertEqual(self.pilot.records, {})
        self.tts.assert_not_called()

    async def test_captured_speech_reserves_device_before_led_cleanup_await(self):
        config = copy.deepcopy(self.config)
        config["network"].update(udp_host="127.0.0.1", udp_port=3000)
        config["audio"]["chunk_size"] = 512
        config["vad"] = {"threshold": .5}
        config["device"].update(led_power=50, led_update_period=.25)
        dev = self.dev
        dev.led_power = 0
        dev.led_update_time = 0
        dev.wake_gate.just_activated = False
        dev.wake_gate.feed.return_value = [np.zeros(512, dtype=np.int16)]
        dev.vad.process_frame = Mock(return_value=np.zeros(512, dtype=np.int16))
        dev.vad.speech_prob = 0
        dev.vad_writer = object()
        closing, release_close = asyncio.Event(), asyncio.Event()
        never = asyncio.Event()
        async def close(*args):
            closing.set()
            await release_close.wait()
        async def receive(*args):
            if not closing.is_set():
                return b"\xff" * 512, (dev.ip, 1234)
            await never.wait()
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        with patch("pipeline.main.socket.socket"), \
             patch.object(loop, "sock_recvfrom", side_effect=receive), \
             patch("pipeline.main.close_led_connection", side_effect=close):
            listener = asyncio.create_task(udp_listener(config,
                SimpleNamespace(get_by_ip=lambda ip: dev), queue))
            try:
                await asyncio.wait_for(closing.wait(), 2)
                self.assertEqual(dev.queued_utterances, 1)
                code, _ = await self.request("POST", payload=PAYLOAD)
                self.assertEqual(code, 409)
                self.assertEqual(self.pilot.records, {})
                self.tts.assert_not_called()
                release_close.set()
                async with asyncio.timeout(2):
                    while queue.empty():
                        await asyncio.sleep(0)
                self.assertIs(queue.get_nowait()[0], dev)
            finally:
                listener.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await listener

    async def test_tts_failure_is_failed_and_releases_processing_without_replay(self):
        self.tts.side_effect = RuntimeError("synthetic TTS failure")
        await self.request("POST", payload=PAYLOAD)
        await self.finish_prompt()
        self.assertEqual(self.pilot.get("pilot-1")["status"], "failed")
        self.send.assert_not_called()
        code, result = await self.request("POST", payload=PAYLOAD)
        self.assertEqual((code, result["status"]), (200, "failed"))
        self.assertEqual(self.tts.await_count, 1)

    async def test_shutdown_cancels_speech_and_releases_reservation(self):
        await self.request("POST", payload=PAYLOAD)
        await self.speech_started.wait()
        self.task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await self.task
        self.assertFalse(self.dev.processing)
        self.assertEqual(self.pilot.get("pilot-1")["status"], "failed")
        self.send.assert_not_called()

    async def test_handler_finishing_body_after_shutdown_cannot_start_speech(self):
        body = json.dumps(PAYLOAD).encode()
        reader = asyncio.StreamReader()
        reader.feed_data(f"POST /pilot/question HTTP/1.1\r\nContent-Length: {len(body)}\r\n\r\n".encode())
        writer = FakeWriter()
        handler = asyncio.create_task(self.handle(reader, writer))
        try:
            await asyncio.sleep(0)
            self.assertFalse(handler.done())
            self.task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await self.task
            reader.feed_data(body)
            reader.feed_eof()
            await asyncio.wait_for(handler, 2)
            header, _ = bytes(writer.data).split(b"\r\n\r\n", 1)
            self.assertEqual(int(header.split()[1]), 404)
            self.assertEqual(self.pilot.records, {})
            self.assertFalse(self.dev.processing)
            self.tts.assert_not_called()
            self.send.assert_not_called()
        finally:
            handler.cancel()
            await asyncio.gather(handler, return_exceptions=True)

    async def test_shutdown_before_question_task_runs_releases_reservation(self):
        await self.request("POST", payload=PAYLOAD)
        self.task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await self.task
        self.assertFalse(self.dev.processing)
        self.assertEqual(self.pilot.get("pilot-1")["status"], "failed")


class ASRRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def run_answer(self, text, expired=False):
        now = [0]
        pilot = VoicePilot(CONFIG["voice_pilot"], clock=lambda: now[0])
        pilot.submit(PAYLOAD, DEVICE, busy=False)
        pilot.speaking("pilot-1")
        pilot.awaiting_answer("pilot-1")
        if expired:
            now[0] = 300
        dev = device()
        queue = asyncio.Queue()
        await queue.put((dev, np.zeros(512, dtype=np.int16)))
        with patch("pipeline.main.asr.transcribe", AsyncMock(return_value={"text": text})), \
             patch("pipeline.main.tts.synthesize", AsyncMock(return_value=b"\x00\x00" * 320)) as tts, \
             patch("pipeline.main.send_audio", AsyncMock()):
            task = asyncio.create_task(process_utterances(CONFIG, None, queue, pilot))
            try:
                await asyncio.wait_for(queue.join(), 2)
            finally:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
        dev.conversation.stream.assert_not_called()
        dev.conversation.commit.assert_not_called()
        dev.wake_gate.arm.assert_not_called()
        self.assertFalse(dev.processing)
        return pilot.get("pilot-1"), tts.call_args.args[0]

    async def test_choice_bypasses_llm_and_speaks_only_fixed_confirmation(self):
        result, spoken = await self.run_answer("Hey Aiden, option two!")
        self.assertEqual((result["status"], result["choice"], result["label"]), ("answered", 2, "green"))
        self.assertEqual(spoken, CONFIRMATIONS[2])

    async def test_unknown_speech_bypasses_llm_and_preserves_question(self):
        result, spoken = await self.run_answer("Hey Aiden, transfer money now")
        self.assertEqual(result["status"], "awaiting_answer")
        self.assertEqual(spoken, CLARIFICATION)

    async def test_expired_choice_is_never_recorded_or_sent_to_llm(self):
        result, spoken = await self.run_answer("Hey Aiden, blue", expired=True)
        self.assertEqual(result["status"], "expired")
        self.assertNotIn("choice", result)
        self.assertEqual(spoken, INACTIVE)


if __name__ == "__main__":
    unittest.main()
