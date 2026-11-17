import io
import unittest
import wave
from unittest.mock import AsyncMock, Mock, patch

from pipeline.services.tts import synthesize


class LocalSpeechTests(unittest.IsolatedAsyncioTestCase):
    async def test_style_is_separate_from_spoken_text_and_audio_is_resampled(self):
        wav = io.BytesIO()
        with wave.open(wav, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(24000)
            writer.writeframes(b"\x00\x00" * 2400)
        response = Mock(content=wav.getvalue())
        client = AsyncMock()
        client.post.return_value = response
        local = {"url": "http://localhost:8880", "model": "custom-voice",
                 "voice": "aiden", "lang_code": "English"}
        with patch("pipeline.services.tts.httpx.AsyncClient") as factory:
            factory.return_value.__aenter__.return_value = client
            for instruction in (None, "Speak calmly and clearly."):
                with self.subTest(instruction=instruction):
                    if instruction:
                        local["instruct"] = instruction
                    pcm = await synthesize("Hello there.", "default", {
                        "tts": {"backend": "local", "local": local}})
                    payload = client.post.call_args.kwargs["json"]
                    self.assertEqual(payload["input"], "Hello there.")
                    self.assertEqual(payload["voice"], "aiden")
                    if instruction:
                        self.assertEqual(payload["instruct"], instruction)
                    else:
                        self.assertNotIn("instruct", payload)
                    self.assertEqual(len(pcm), 3200)


if __name__ == "__main__":
    unittest.main()
