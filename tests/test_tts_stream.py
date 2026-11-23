"""Streamed TTS: WAV stream parser and synthesize_stream, no network."""
import io
import struct
import unittest
import wave
from unittest.mock import AsyncMock, MagicMock, patch

from pipeline.services.tts import WavStreamError, WavStreamParser, synthesize, synthesize_stream


def make_wav(pcm, rate=24000, channels=1):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def pcm_of(n, seed=1):
    return b"".join(struct.pack("<h", (i * 37 * seed) % 20000 - 10000) for i in range(n))


def join(parts):
    return b"".join(p[3] for p in parts)


class WavStreamParserTests(unittest.TestCase):
    def setUp(self):
        self.pcms = [pcm_of(960, 1), pcm_of(961, 2), pcm_of(100, 3)]
        self.stream = b"".join(make_wav(p) for p in self.pcms)
        self.expected = b"".join(self.pcms)

    def run_chunks(self, chunks):
        parser = WavStreamParser()
        out = []
        for c in chunks:
            out += parser.feed(c)
        parser.finish()
        return out

    def test_all_at_once(self):
        parts = self.run_chunks([self.stream])
        self.assertEqual(join(parts), self.expected)
        self.assertEqual(parts[0][:3], (24000, 1, 2))

    def test_byte_by_byte(self):
        self.assertEqual(join(self.run_chunks([bytes([b]) for b in self.stream])), self.expected)

    def test_split_mid_header_and_mid_data(self):
        first_len = len(make_wav(self.pcms[0]))
        for cut in (5, 20, 40, 44, 45, 500, first_len - 1, first_len + 7, first_len + 46):
            with self.subTest(cut=cut):
                self.assertEqual(
                    join(self.run_chunks([self.stream[:cut], self.stream[cut:]])), self.expected)

    def test_bad_data_lengths_fail(self):
        wav = bytearray(make_wav(pcm_of(100)))
        idx = wav.index(b"data") + 4
        for bad in (0, 0xFFFFFFFF, 10_000_000):
            with self.subTest(bad=bad):
                w = bytearray(wav)
                w[idx:idx + 4] = struct.pack("<I", bad)
                with self.assertRaises(WavStreamError):
                    WavStreamParser().feed(bytes(w))

    def test_bad_riff_and_garbage_fail(self):
        wav = bytearray(make_wav(pcm_of(100)))
        for off, val in ((4, 0), (4, 0xFFFFFFFF)):
            w = bytearray(wav)
            w[off:off + 4] = struct.pack("<I", val)
            with self.assertRaises(WavStreamError):
                WavStreamParser().feed(bytes(w))
        with self.assertRaises(WavStreamError):
            WavStreamParser().feed(b"ID3" + b"\x00" * 40)

    def test_truncated_stream_fails_on_finish(self):
        parser = WavStreamParser()
        parser.feed(self.stream[:-3])
        with self.assertRaises(WavStreamError):
            parser.finish()


class FakeResponse:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def raise_for_status(self):
        pass

    async def aiter_bytes(self):
        for c in self.chunks:
            yield c


def fake_client(chunks):
    response = FakeResponse(chunks)
    stream_cm = MagicMock()
    stream_cm.__aenter__ = AsyncMock(return_value=response)

    async def exit_(*a):
        response.closed = True
    stream_cm.__aexit__ = exit_
    client = MagicMock()
    client.stream = MagicMock(return_value=stream_cm)
    client.post = AsyncMock()
    factory = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=client)
    factory.return_value.__aexit__ = AsyncMock(return_value=False)
    return factory, client, response


LOCAL = {"url": "http://localhost:8880/", "model": "m", "voice": "aiden", "lang_code": "English",
         "instruct": "calm", "ref_text": "hi"}


def cfg(**extra):
    return {"tts": {"backend": "local", "local": {**LOCAL, **extra}}}


class SynthesizeStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_streams_resampled_pcm_with_stream_payload(self):
        pcms = [pcm_of(2400, 1), pcm_of(2400, 2), pcm_of(1201, 3)]
        raw = b"".join(make_wav(p) for p in pcms)
        chunks = [raw[i:i + 777] for i in range(0, len(raw), 777)]
        factory, client, response = fake_client(chunks)
        with patch("pipeline.services.tts.httpx.AsyncClient", factory):
            out = [p async for p in synthesize_stream("Hello there.", "v", cfg(stream=True))]
        self.assertGreater(len(out), 1)
        total = sum(len(p) for p in out)
        # one-shot reference path
        one_shot = MagicMock(content=make_wav(b"".join(pcms)))
        client2 = AsyncMock()
        client2.post.return_value = one_shot
        with patch("pipeline.services.tts.httpx.AsyncClient") as f2:
            f2.return_value.__aenter__.return_value = client2
            ref = await synthesize("Hello there.", "v", cfg())
        self.assertLessEqual(abs(total - len(ref)), 640)
        self.assertEqual(total % 2, 0)
        method, url = client.stream.call_args.args
        self.assertEqual((method, url), ("POST", "http://localhost:8880/v1/audio/speech"))
        streamed = client.stream.call_args.kwargs["json"]
        oneshot = client2.post.call_args.kwargs["json"]
        self.assertEqual(streamed.pop("stream"), True)
        self.assertEqual(streamed.pop("streaming_interval"), 0.4)
        self.assertEqual(streamed, oneshot)
        self.assertTrue(response.closed)

    async def test_streaming_interval_from_config(self):
        factory, client, _ = fake_client([make_wav(pcm_of(240))])
        with patch("pipeline.services.tts.httpx.AsyncClient", factory):
            _ = [p async for p in synthesize_stream("Hi.", "v", cfg(stream=True, streaming_interval=0.25))]
        self.assertEqual(client.stream.call_args.kwargs["json"]["streaming_interval"], 0.25)

    async def test_closing_generator_closes_http_stream(self):
        factory, client, response = fake_client([make_wav(pcm_of(2400))] * 3)
        with patch("pipeline.services.tts.httpx.AsyncClient", factory):
            gen = synthesize_stream("Hi.", "v", cfg(stream=True))
            await anext(gen)
            await gen.aclose()
        self.assertTrue(response.closed)

    async def test_bad_stream_raises(self):
        wav = bytearray(make_wav(pcm_of(100)))
        wav[wav.index(b"data") + 4:wav.index(b"data") + 8] = struct.pack("<I", 0)
        factory, _, _ = fake_client([bytes(wav)])
        with patch("pipeline.services.tts.httpx.AsyncClient", factory):
            with self.assertRaises(WavStreamError):
                _ = [p async for p in synthesize_stream("Hi.", "v", cfg(stream=True))]

    async def test_flag_off_yields_once_via_synthesize(self):
        with patch("pipeline.services.tts.synthesize", AsyncMock(return_value=b"\x01\x00" * 10)) as syn:
            out = [p async for p in synthesize_stream("Hi.", "v", cfg())]
        self.assertEqual(out, [b"\x01\x00" * 10])
        syn.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
