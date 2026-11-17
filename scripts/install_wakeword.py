"""Download the pinned local Hey Aiden keyword model (no account/API key).

Run from the repository: .venv/bin/python scripts/install_wakeword.py
Only model files are installed; this does not enable or start a microphone.
"""
import argparse
import hashlib
import io
from pathlib import Path
import tarfile
import urllib.request

MODEL = "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/{MODEL}.tar.bz2"
SHA256 = "68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6"
FILES = (
    "tokens.txt",
    "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx",
    "decoder-epoch-13-avg-2-chunk-8-left-64.onnx",
    "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx",
)
ENGLISH_MODEL = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
PROFILES = {
    "bilingual": (MODEL, URL, SHA256, FILES),
    "english": (
        ENGLISH_MODEL,
        f"https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/{ENGLISH_MODEL}.tar.bz2",
        "f170013b4716e41b62b9bfd809687c207cef798ef9bc6534d524e17af9b6561a",
        ("tokens.txt", "encoder-epoch-12-avg-2-chunk-16-left-64.onnx",
         "decoder-epoch-12-avg-2-chunk-16-left-64.onnx",
         "joiner-epoch-12-avg-2-chunk-16-left-64.onnx"),
    ),
}


def install(destination: Path, profile: str = "bilingual"):
    if profile not in PROFILES:
        raise ValueError("Unknown wake model profile")
    model, url, checksum, files = PROFILES[profile]
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(40_000_001)
    if len(data) > 40_000_000 or hashlib.sha256(data).hexdigest() != checksum:
        raise ValueError("Wake model download failed size/checksum verification")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:bz2") as archive:
        prepared = {}
        for name in files:
            member = archive.getmember(f"{model}/{name}")
            if not member.isfile() or member.size > 15_000_000:
                raise ValueError("Unexpected model archive member")
            with archive.extractfile(member) as source:
                prepared[name] = source.read()
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in prepared.items():
        output = destination / name
        temporary = destination / (name + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(output)
    print(f"Installed verified local wake model in {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="bilingual")
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    destination = args.destination or Path("data/wakeword-english" if args.profile == "english" else "data/wakeword")
    install(destination, args.profile)
