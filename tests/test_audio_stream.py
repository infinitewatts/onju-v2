"""OpusStreamEncoder and send_audio_stream: local sockets only."""
import asyncio
import unittest

from pipeline.audio import OpusStreamEncoder, opus_encode, opus_frames_to_tcp_payload
from pipeline.protocol import StreamStats, send_audio, send_audio_stream


def pcm(n_bytes):
    return bytes((i * 7) % 256 for i in range(n_bytes // 2 * 2))


class OpusStreamEncoderTests(unittest.TestCase):
    def test_awkward_pieces_match_one_shot_frame_count(self):
        data = pcm(640 * 7 + 222)
        enc = OpusStreamEncoder(16000, 320)
        frames = []
        pos = 0
        for size in (1, 2, 639, 641, 1000, 17, 2000):
            frames += enc.encode(data[pos:pos + size])
            pos += size
        frames += enc.encode(data[pos:])
        frames += enc.flush()
        self.assertEqual(len(frames), len(opus_encode(data, 16000, 320)))

    def test_flush_pads_once(self):
        enc = OpusStreamEncoder(16000, 320)
        self.assertEqual(enc.encode(b"\x01\x00" * 100), [])
        self.assertEqual(len(enc.flush()), 1)
        self.assertEqual(enc.flush(), [])

    def test_silence_frame_is_opus(self):
        enc = OpusStreamEncoder(16000, 320)
        self.assertGreater(len(enc.silence_frame()), 0)
        self.assertEqual(enc.encode(b""), [])


class Server:
    def __init__(self, close_after=None):
        self.data = b""
        self.close_after = close_after
        self.done = asyncio.Event()
        self.first_data = None

    async def handle(self, reader, writer):
        loop = asyncio.get_running_loop()
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                break
            self.first_data = self.first_data or loop.time()
            self.data += chunk
            if self.close_after and len(self.data) >= self.close_after:
                break
        writer.close()
        self.done.set()


class SendAudioStreamTests(unittest.IsolatedAsyncioTestCase):
    async def serve(self, **kw):
        srv = Server(**kw)
        server = await asyncio.start_server(srv.handle, "127.0.0.1", 0)
        self.addCleanup(server.close)
        return srv, server.sockets[0].getsockname()[1]

    async def test_bytes_match_send_audio(self):
        frames = [bytes([i]) * (10 + i) for i in range(9)]
        for timeout in (0, 60):
            with self.subTest(timeout=timeout):
                ref, port = await self.serve()
                await send_audio("127.0.0.1", port, opus_frames_to_tcp_payload(frames),
                                 mic_timeout=timeout, volume=14, fade=6)
                await asyncio.wait_for(ref.done.wait(), 2)
                got, port2 = await self.serve()

                async def batches():
                    yield frames[:3]
                    await asyncio.sleep(0.02)
                    yield frames[3:4]
                    yield []
                    yield frames[4:]
                stats = StreamStats()
                self.assertTrue(await send_audio_stream(
                    "127.0.0.1", port2, batches(), mic_timeout=timeout, volume=14, fade=6, stats=stats))
                await asyncio.wait_for(got.done.wait(), 2)
                self.assertEqual(got.data, ref.data)
                self.assertEqual(stats.frames_written, 9)

    async def test_empty_stream_matches_empty_send_audio(self):
        ref, p1 = await self.serve()
        await send_audio("127.0.0.1", p1, b"", mic_timeout=60)
        got, p2 = await self.serve()

        async def none():
            return
            yield
        self.assertTrue(await send_audio_stream("127.0.0.1", p2, none(), mic_timeout=60))
        await asyncio.wait_for(got.done.wait(), 2)
        await asyncio.wait_for(ref.done.wait(), 2)
        self.assertEqual(got.data, ref.data)

    async def test_nothing_written_before_first_batch(self):
        srv, port = await self.serve()
        release = asyncio.Event()

        async def batches():
            await release.wait()
            yield [b"abc"]

        task = asyncio.create_task(send_audio_stream("127.0.0.1", port, batches()))
        await asyncio.sleep(0.15)
        self.assertEqual(srv.data, b"")
        self.assertIsNone(srv.first_data)  # not even connected-and-written
        release.set()
        self.assertTrue(await asyncio.wait_for(task, 2))
        await asyncio.wait_for(srv.done.wait(), 2)
        self.assertTrue(srv.data.startswith(bytes([0xAA, 0, 60, 14, 6, 2]) + b"\x00\x03abc"))

    async def test_returns_false_when_peer_closes_mid_stream(self):
        srv, port = await self.serve(close_after=20)
        stats = StreamStats()

        async def batches():
            for _ in range(200):
                yield [b"x" * 500]
                await asyncio.sleep(0.005)

        ok = await asyncio.wait_for(
            send_audio_stream("127.0.0.1", port, batches(), stats=stats), 5)
        self.assertFalse(ok)
        self.assertLess(stats.frames_written, 200)

    async def test_connection_refused_returns_false(self):
        async def batches():
            yield [b"a"]
        self.assertFalse(await send_audio_stream("127.0.0.1", 1, batches(), timeout=1))

    async def test_may_send_false_returns_false(self):
        srv, port = await self.serve()

        async def batches():
            yield [b"a"]
        self.assertFalse(await send_audio_stream("127.0.0.1", port, batches(), may_send=lambda: False))
        await asyncio.wait_for(srv.done.wait(), 2)
        self.assertEqual(srv.data, b"")


if __name__ == "__main__":
    unittest.main()


class SilenceFrameTests(unittest.TestCase):
    def test_silence_frame_does_not_disturb_the_speech_encoder(self):
        import numpy as np
        from pipeline.audio import OpusStreamEncoder
        tone = (np.sin(np.arange(16000) * 0.2) * 8000).astype(np.int16).tobytes()
        plain = OpusStreamEncoder(16000, 320)
        expected = plain.encode(tone)
        mixed = OpusStreamEncoder(16000, 320)
        half = len(tone) // 2
        got = mixed.encode(tone[:half])
        mixed.silence_frame()
        got += mixed.encode(tone[half:])
        self.assertEqual(got, expected)
