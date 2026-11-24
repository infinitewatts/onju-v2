"""Microphone stream accounting: sequence gaps and device counter reports.

Pure functions and counters only. Never holds audio samples or text.

Wire formats (all numbers big-endian):
  legacy audio   512 bytes u-law
  audio + seq    517 bytes: 512 u-law, 0xA5, 0x01, u8 boot id, u16 seq
  stats report   29 bytes: "OJST", 0x01, u16 seq, u32 send_fail, u32 send_retry,
                 u32 late_reads, u16 max_read_gap_ms, i16 last_errno, i8 rssi,
                 u8 channel, u8 bssid_last, u16 stack_free, u8 boot_id
"""
import struct
import time

AUDIO_BYTES = 512
TRAILER_MAGIC = 0xA5
TRAILER_VERSION = 0x01
TRAILER_SIZE = 5
PACKET_SIZE = AUDIO_BYTES + TRAILER_SIZE
REPORT_MAGIC = b"OJST"
REPORT_VERSION = 0x01
_REPORT = struct.Struct(">4sBHIIIHhbBBHB")
REPORT_SIZE = _REPORT.size  # 29
REPORT_FIELDS = ("seq", "send_fail", "send_retry", "late_reads", "max_read_gap_ms",
                 "last_errno", "rssi", "channel", "bssid_last", "stack_free", "boot_id")

MAX_GAP = 3000        # a bigger jump is a restart, not loss
SILENCE_RESET_S = 2.0  # after this long without packets, a backward jump is a device restart
BUCKETS = ("1", "2-3", "4-7", "8-15", "16+")


def parse_datagram(data: bytes, chunk_bytes: int = AUDIO_BYTES):
    """Return ("audio", payload, seq_or_None, boot_id_or_None), ("report", dict), or None."""
    n = len(data)
    if n == chunk_bytes:
        return ("audio", data, None, None)
    if n == chunk_bytes + TRAILER_SIZE and data[chunk_bytes] == TRAILER_MAGIC \
            and data[chunk_bytes + 1] == TRAILER_VERSION:
        return ("audio", data[:chunk_bytes],
                (data[chunk_bytes + 3] << 8) | data[chunk_bytes + 4], data[chunk_bytes + 2])
    if n == REPORT_SIZE and data[:4] == REPORT_MAGIC and data[4] == REPORT_VERSION:
        values = _REPORT.unpack(data)
        return ("report", dict(zip(REPORT_FIELDS, values[2:])))
    return None


def _bucket(gap: int) -> str:
    if gap == 1:
        return "1"
    if gap <= 3:
        return "2-3"
    if gap <= 7:
        return "4-7"
    if gap <= 15:
        return "8-15"
    return "16+"


class MicStreamMonitor:
    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self.seq_packets = 0
        self.seq_gaps = 0
        self.seq_lost = 0
        self.seq_reordered = 0
        self.seq_resets = 0
        self.gap_histogram = {b: 0 for b in BUCKETS}
        self.report = None
        self.report_at = None
        self._expected = None
        self._boot = None
        self._last_at = None

    def observe(self, seq, boot_id=None):
        """Record one sequenced audio packet. Legacy packets (seq None) are ignored."""
        if seq is None:
            return
        now = self._clock()
        self.seq_packets += 1
        silent = self._last_at is None or now - self._last_at > SILENCE_RESET_S
        self._last_at = now
        restarted = self._expected is not None and boot_id != self._boot
        self._boot = boot_id
        if self._expected is None:
            self._expected = (seq + 1) & 0xFFFF
            return
        delta = (seq - self._expected) & 0xFFFF
        if restarted:
            # A new boot id: the device powered up again and numbers from zero.
            # Never loss, wherever the old numbering was.
            self.seq_resets += 1
        elif delta == 0:
            pass
        elif delta <= MAX_GAP:
            # Counted even after a long silence: the device numbers only the
            # frames it captured, so playback, mute and reconnects leave no gap.
            self.seq_gaps += 1
            self.seq_lost += delta
            self.gap_histogram[_bucket(delta)] += 1
        elif not silent and delta >= 0x10000 - MAX_GAP:
            self.seq_reordered += 1  # late or duplicate; expectation stays put
            return
        else:
            self.seq_resets += 1     # a jump too large to be loss
        self._expected = (seq + 1) & 0xFFFF

    def observe_report(self, report: dict):
        self.report = dict(report)
        self.report_at = self._clock()

    def to_dict(self) -> dict:
        return {
            "seq_packets": self.seq_packets,
            "seq_gaps": self.seq_gaps,
            "seq_lost": self.seq_lost,
            "seq_reordered": self.seq_reordered,
            "seq_resets": self.seq_resets,
            "gap_histogram": dict(self.gap_histogram),
            "device_report": dict(self.report) if self.report is not None else None,
            "device_report_age_s": (round(max(0.0, self._clock() - self.report_at), 2)
                                    if self.report_at is not None else None),
        }
