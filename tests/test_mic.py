#!/usr/bin/env python3
"""
Receive and record audio from ESP32 via UDP.
Auto-detects u-law compressed or raw PCM packets.

Usage:
    python test_mic_receiver.py
    python test_mic_receiver.py --duration 30 --output recording.wav
"""
import argparse
import ipaddress
import os
import socket
import sys
import time
import wave
from collections import deque

import numpy as np

from pipeline.audio import decode_ulaw

SAMPLE_RATE = 16000
CHUNK_SIZE = 512
CHUNK_BYTES_RAW = CHUNK_SIZE * 2
CHUNK_BYTES_COMPRESSED = CHUNK_SIZE


def main():
    parser = argparse.ArgumentParser(description="ESP32 mic UDP receiver")
    parser.add_argument("--port", type=int, default=3000, help="UDP port (default: 3000)")
    parser.add_argument("--output", type=str, default="test_recording.wav", help="Output WAV file")
    parser.add_argument("--duration", type=int, default=10, help="Recording duration in seconds (0 = infinite)")
    parser.add_argument("--device-ip", type=ipaddress.IPv4Address, help="Accept only this board's IPv4 address")
    parser.add_argument("--wait-timeout", type=int, default=15, help="Seconds to wait for first audio packet")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(1.0)

    print(f"Listening on UDP port {args.port}...")
    print(f"Output: {args.output}")
    print("Waiting for packets...\n")

    audio_frames = []
    packet_count = 0
    total_bytes = 0
    start_time = None
    last_stats_time = None
    waiting_since = time.monotonic()
    rms_history = deque(maxlen=20)

    try:
        while True:
            if start_time is None and time.monotonic() - waiting_since >= args.wait_timeout:
                break
            if start_time is not None and args.duration > 0 and time.time() - start_time >= args.duration:
                break
            try:
                data, addr = sock.recvfrom(2048)
                if args.device_ip and addr[0] != str(args.device_ip):
                    continue
                if len(data) not in (CHUNK_BYTES_COMPRESSED, CHUNK_BYTES_RAW):
                    continue

                if start_time is None:
                    start_time = time.time()
                    last_stats_time = start_time
                    print(f"Receiving from {addr[0]}:{addr[1]}\n")

                packet_count += 1
                total_bytes += len(data)

                if len(data) == CHUNK_BYTES_COMPRESSED:
                    samples = decode_ulaw(data)
                    mode = "u-law"
                elif len(data) == CHUNK_BYTES_RAW:
                    samples = np.frombuffer(data, dtype=np.int16)
                    mode = "raw"
                else:
                    continue

                audio_frames.append(samples)
                rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
                rms_history.append(rms)

                current_time = time.time()
                if current_time - last_stats_time >= 1.0:
                    elapsed = current_time - start_time
                    bandwidth_kbps = (total_bytes * 8) / (elapsed * 1000)
                    avg_rms = np.mean(list(rms_history))
                    print(f"\r[{elapsed:6.1f}s] Packets: {packet_count:4d} | "
                          f"Bandwidth: {bandwidth_kbps:5.1f} kbps | "
                          f"RMS: {avg_rms:6.0f} | "
                          f"Mode: {mode}",
                          end="", flush=True)
                    last_stats_time = current_time

                if args.duration > 0 and (current_time - start_time) >= args.duration:
                    print("\n\nRecording duration reached.")
                    break

            except socket.timeout:
                if start_time is not None:
                    if args.duration > 0:
                        continue  # Preserve the requested capture window across packet gaps.
                    print("\n\nTimeout - no packets for 1 second")
                    break
                if time.monotonic() - waiting_since >= args.wait_timeout:
                    break
                continue

    except KeyboardInterrupt:
        print("\n\nInterrupted")
    finally:
        sock.close()

    if not audio_frames:
        print("No audio received!")
        sys.exit(1)

    audio_data = np.concatenate(audio_frames)

    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "wb") as output:
        with wave.open(output, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())

    duration = len(audio_data) / SAMPLE_RATE
    transmitted_kb = total_bytes / 1024
    print(f"\nSaved {duration:.1f}s to {args.output} ({transmitted_kb:.1f} KB received, {packet_count} packets)")


if __name__ == "__main__":
    main()
