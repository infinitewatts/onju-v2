import audioop
import io
import logging
import os
import struct

import httpx
from pydub import AudioSegment

from pipeline.text_normalize import normalize_for_speech

log = logging.getLogger(__name__)


async def synthesize(text: str, voice: str, config: dict) -> bytes:
    """Convert text to 16kHz mono PCM bytes using the configured TTS backend."""
    backend = config["tts"]["backend"]
    if config["tts"].get("normalize_text", True):
        text = normalize_for_speech(text)
    if backend == "elevenlabs":
        return await _elevenlabs(text, voice, config)
    if backend == "local":
        return await _local(text, config)
    raise ValueError(f"Unknown TTS backend: {backend}")


class WavStreamError(ValueError):
    """The TTS stream is not a well-formed run of WAV files."""


class WavStreamParser:
    """Incrementally parse a concatenation of complete PCM WAV files.

    feed(bytes) returns a list of (sample_rate, channels, sample_width, pcm)
    tuples. PCM is aligned to whole sample blocks; leftover bytes wait for the
    next feed. Anything suspicious raises WavStreamError rather than guessing.

    The total size in each RIFF header is not used: the local TTS server writes
    a value 8 bytes short of the real file. A file ends where the next "RIFF"
    header (or the end of the stream) follows a complete chunk.
    """

    _MAX_SKIP = 1 << 20
    _MAX_DATA = 64 << 20

    def __init__(self):
        self._buf = bytearray()
        self._state = "riff"
        self._fmt = None
        self._has_data = True   # nothing is owed before the first file starts
        self._remaining = 0     # bytes left in the current data/skip chunk
        self._pad = 0

    def _take(self, n):
        if len(self._buf) < n:
            return None
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out

    def feed(self, data: bytes) -> list:
        self._buf += data
        out = []
        while True:
            if self._state == "riff":
                head = self._take(12)
                if head is None:
                    break
                if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
                    raise WavStreamError("not a RIFF/WAVE stream")
                self._fmt = None
                self._has_data = False
                self._state = "chunk"
            elif self._state == "chunk":
                if len(self._buf) < 4:
                    break
                if self._buf[:4] == b"RIFF":  # the next file starts here
                    if not self._has_data:
                        raise WavStreamError("WAV file without audio data")
                    self._state = "riff"
                    continue
                head = self._take(8)
                if head is None:
                    break
                cid, size = head[:4], struct.unpack("<I", head[4:])[0]
                if cid == b"fmt ":
                    if size < 16 or size > 64:
                        raise WavStreamError("bad fmt chunk size")
                    self._remaining, self._state = size, "fmt"
                elif cid == b"data":
                    if self._fmt is None:
                        raise WavStreamError("data before fmt")
                    if size > self._MAX_DATA:
                        raise WavStreamError("untrusted data length")
                    if size % (self._fmt[1] * self._fmt[2]):
                        raise WavStreamError("data length not block aligned")
                    self._has_data = True
                    self._remaining, self._state = size, "data"
                else:
                    if size > self._MAX_SKIP:
                        raise WavStreamError("oversized unknown chunk")
                    self._remaining, self._state = size, "skip"
                self._pad = size & 1
            elif self._state == "fmt":
                body = self._take(self._remaining)
                if body is None:
                    break
                tag, ch, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
                if tag not in (1, 0xFFFE) or bits != 16 or ch not in (1, 2) or rate <= 0:
                    raise WavStreamError("unsupported WAV format")
                self._fmt = (rate, ch, bits // 8)
                self._state = "pad"
            elif self._state == "skip":
                if self._take(self._remaining) is None:
                    break
                self._state = "pad"
            elif self._state == "data":
                if self._remaining == 0:  # empty piece: nothing to emit
                    self._state = "pad"
                    continue
                block = self._fmt[1] * self._fmt[2]
                n = min(self._remaining, len(self._buf))
                n -= n % block
                if n <= 0:
                    break
                out.append((*self._fmt[:2], self._fmt[2], self._take(n)))
                self._remaining -= n
                if self._remaining == 0:
                    self._state = "pad"
            elif self._state == "pad":
                if self._pad and self._take(self._pad) is None:
                    break
                self._state = "chunk"
        return out

    def finish(self):
        """Raise if the stream ended mid-file."""
        if self._buf or self._state not in ("chunk", "riff") or not self._has_data:
            raise WavStreamError("stream ended mid-WAV")


async def synthesize_stream(text: str, voice: str, config: dict):
    """Yield 16 kHz mono 16-bit PCM pieces as the TTS backend produces them.

    Only the local backend with tts.local.stream enabled really streams; every
    other case yields the one-shot synthesize() result once.
    """
    tts_cfg = config["tts"]
    local_cfg = tts_cfg.get("local") or {}
    if tts_cfg["backend"] != "local" or not local_cfg.get("stream"):
        yield await synthesize(text, voice, config)
        return
    if tts_cfg.get("normalize_text", True):
        text = normalize_for_speech(text)
    url = local_cfg["url"].rstrip("/") + "/v1/audio/speech"
    payload = _local_payload(text, local_cfg)
    payload["stream"] = True
    payload["streaming_interval"] = local_cfg.get("streaming_interval", 0.4)

    parser = WavStreamParser()
    rate_state = None
    state_rate = None
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                for rate, channels, width, pcm in parser.feed(chunk):
                    if channels == 2:
                        pcm = audioop.tomono(pcm, width, 0.5, 0.5)
                    if rate != 16000:
                        if rate != state_rate:
                            rate_state, state_rate = None, rate
                        pcm, rate_state = audioop.ratecv(pcm, width, 1, rate, 16000, rate_state)
                    if pcm:
                        yield pcm
            parser.finish()


async def _elevenlabs(text: str, voice_name: str, config: dict) -> bytes:
    el_cfg = config["tts"]["elevenlabs"]
    api_key = el_cfg["api_key"]
    voice_id = el_cfg["voices"].get(voice_name, el_cfg["voices"].get(el_cfg["default_voice"]))

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {"text": text}

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        mp3_bytes = resp.content

    audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    log.debug(f"TTS: {len(text)} chars -> {len(audio)}ms audio")
    return audio.raw_data


def _local_payload(text: str, local_cfg: dict) -> dict:
    payload = {
        "model": local_cfg["model"],
        "input": text,
        "response_format": "wav",
    }

    for key in ("voice", "lang_code", "instruct"):
        if local_cfg.get(key):
            payload[key] = local_cfg[key]

    # Voice cloning: pass ref_audio path (server reads from disk)
    ref_audio = local_cfg.get("ref_audio")
    if ref_audio:
        payload["ref_audio"] = os.path.abspath(ref_audio)
    ref_text = local_cfg.get("ref_text")
    if ref_text:
        payload["ref_text"] = ref_text
    return payload


async def _local(text: str, config: dict) -> bytes:
    local_cfg = config["tts"]["local"]
    url = local_cfg["url"].rstrip("/") + "/v1/audio/speech"
    payload = _local_payload(text, local_cfg)

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        wav_bytes = resp.content

    audio = AudioSegment.from_wav(io.BytesIO(wav_bytes))
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    log.debug(f"TTS local: {len(text)} chars -> {len(audio)}ms audio")
    return audio.raw_data
