import asyncio
import logging
import struct

log = logging.getLogger(__name__)


async def send_tcp(ip: str, port: int, header: bytes, data: bytes | None = None,
                   timeout: float = 5, may_send=None):
    writer = None
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(ip, port), timeout=timeout
        )
        # State may have changed while TCP connected. No suspension separates
        # this final guard from the first protocol write.
        if may_send is not None and not may_send():
            return False
        writer.write(header)
        if data:
            writer.write(data)
        await asyncio.wait_for(writer.drain(), timeout=timeout)
        return True
    except (asyncio.TimeoutError, ConnectionError, OSError):
        return False  # non-critical, device may be busy
    finally:
        if writer is not None:
            writer.close()
            try:
                await asyncio.wait_for(writer.wait_closed(), timeout=1)
            except (asyncio.TimeoutError, ConnectionError, OSError):
                pass


async def send_audio(ip: str, port: int, opus_payload: bytes, mic_timeout: int = 60, volume: int = 14, fade: int = 6, may_send=None):
    volume = max(0, min(16, int(volume)))
    # header[0]   0xAA for audio
    # header[1:2] mic timeout in seconds (big-endian)
    # header[3]   volume
    # header[4]   fade rate
    # header[5]   compression type (2 = Opus)
    header = bytes([
        0xAA,
        (mic_timeout >> 8) & 0xFF,
        mic_timeout & 0xFF,
        volume,
        fade,
        2,  # Opus
    ])
    # Onju floors the audio header's timeout at 20 seconds. Follow the Opus
    # terminator with its explicit timeout command on the SAME TCP connection,
    # so zero really stops listening and shorter follow-up windows are honored
    # only after playback, without a separate connection racing the audio.
    payload = opus_payload or b"\x00\x00"
    if mic_timeout < 20:
        payload += bytes([0xDD, (mic_timeout >> 8) & 0xFF, mic_timeout & 0xFF, 0, 0, 0])
    return await send_tcp(ip, port, header, payload, may_send=may_send)


class StreamStats:
    """Filled in by send_audio_stream: Opus frames actually written."""

    def __init__(self):
        self.frames_written = 0


async def send_audio_stream(ip: str, port: int, frames, mic_timeout: int = 60, volume: int = 14,
                            fade: int = 6, may_send=None, timeout: float = 5, stats=None):
    """Like send_audio, but `frames` is an async iterator of lists of Opus frames.

    Connects only after the first batch is available. The wire bytes for the
    same frames are identical to send_audio's. `stats.frames_written` counts
    frames accepted by the local socket (an upper bound on what the peer played). Returns True on success, False otherwise.
    """
    volume = max(0, min(16, int(volume)))
    header = bytes([0xAA, (mic_timeout >> 8) & 0xFF, mic_timeout & 0xFF, volume, fade, 2])
    stats = stats if stats is not None else StreamStats()

    def encode(batch):
        return b"".join(struct.pack(">H", len(f)) + f for f in batch)

    first = await anext(frames, None)
    writer = None
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(ip, port), timeout=timeout)
        if may_send is not None and not may_send():
            return False
        first = first or []
        writer.write(header + encode(first))
        await asyncio.wait_for(writer.drain(), timeout=timeout)
        stats.frames_written += len(first)
        async for batch in frames:
            if reader.at_eof():
                return False
            if not batch:
                continue
            writer.write(encode(batch))
            await asyncio.wait_for(writer.drain(), timeout=timeout)
            stats.frames_written += len(batch)
        tail = b"\x00\x00"
        if mic_timeout < 20:
            tail += bytes([0xDD, (mic_timeout >> 8) & 0xFF, mic_timeout & 0xFF, 0, 0, 0])
        writer.write(tail)
        await asyncio.wait_for(writer.drain(), timeout=timeout)
        return True
    except (asyncio.TimeoutError, ConnectionError, OSError):
        return False
    finally:
        if writer is not None:
            writer.close()
            try:
                await asyncio.wait_for(writer.wait_closed(), timeout=1)
            except (asyncio.TimeoutError, ConnectionError, OSError):
                pass


async def send_led_blink(ip: str, port: int, intensity: int, r: int = 255, g: int = 255, b: int = 255, fade: int = 6):
    # header[0]   0xCC for LED blink
    # header[1]   starting intensity
    # header[2:4] RGB
    # header[5]   fade rate
    header = bytes([0xCC, intensity, r, g, b, fade])
    await send_tcp(ip, port, header, timeout=0.1)


async def send_stop_listening(ip: str, port: int, hold_s: int = 0):
    """Unused by pipeline — kept for sesame-esp32-bridge compatibility."""
    header = bytes([0xDD, (hold_s >> 8) & 0xFF, hold_s & 0xFF, 0, 0, 0])
    await send_tcp(ip, port, header, timeout=0.2)


async def open_led_connection(ip: str, port: int, timeout: float = 1) -> asyncio.StreamWriter | None:
    """Open a persistent TCP connection for streaming LED blink commands."""
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(ip, port), timeout=timeout
        )
        return writer
    except (asyncio.TimeoutError, ConnectionError, OSError):
        return None


def write_led_blink(writer: asyncio.StreamWriter, intensity: int,
                    r: int = 255, g: int = 255, b: int = 255, fade: int = 6) -> bool:
    """Write a LED blink command to an open connection. Non-async (6-byte buffer write)."""
    header = bytes([0xCC, intensity, r, g, b, fade])
    try:
        writer.write(header)
        return True
    except (ConnectionError, OSError):
        return False


async def close_led_connection(writer: asyncio.StreamWriter):
    """Close a persistent LED connection."""
    try:
        writer.close()
        await writer.wait_closed()
    except (ConnectionError, OSError):
        pass
