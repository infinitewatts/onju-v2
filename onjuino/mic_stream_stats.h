// Microphone stream counters. Plain C++ (no Arduino dependencies) so it can be
// tested on a computer: tests/mic_stream_stats_test.cpp.
//
// Measurement only. Each audio packet carries a sequence number so the server
// can tell a lost packet from a quiet one, and every ~10 s the device sends a
// small counter report so the server can tell "never sent" from "lost in the
// air" and "I2S reader stalled".
//
// Audio packet (517 bytes):  [0..511] u-law audio
//                            [512] 0xA5 magic  [513] 0x01 version
//                            [514] boot id (random per power-up, so the server
//                                  can tell a restart from lost packets)
//                            [515..516] sequence number, big-endian
// Report packet (29 bytes), all numbers big-endian:
//   [0..3] "OJST"  [4] version 0x01  [5..6] seq  [7..10] send_fail
//   [11..14] send_retry  [15..18] late_reads  [19..20] max_read_gap_ms
//   [21..22] last_errno (i16)  [23] rssi (i8)  [24] wifi channel
//   [25] last byte of AP BSSID  [26..27] least free stack the mic task has had, bytes
//   [28] boot id
#ifndef MIC_STREAM_STATS_H
#define MIC_STREAM_STATS_H

#include <stdint.h>
#include <stddef.h>

#define MIC_AUDIO_BYTES 512
#define MIC_TRAILER_MAGIC 0xA5
#define MIC_TRAILER_VERSION 0x01
#define MIC_TRAILER_SIZE 5
#define MIC_PACKET_SIZE (MIC_AUDIO_BYTES + MIC_TRAILER_SIZE)   // 517
#define MIC_REPORT_VERSION 0x01
#define MIC_REPORT_SIZE 29
#define MIC_REPORT_EVERY_FRAMES 312   // about 10 s of 32 ms frames
#define MIC_LATE_READ_MS 64           // interval between reads counted as late
#define MIC_SEND_RETRIES 2            // extra tries after a failed send
#define MIC_ERR_NO_PACKET (-1)        // the packet could not even be started (no errno)

struct MicStreamStats {
    uint16_t seq;              // sequence number of the NEXT frame
    uint8_t boot_id;           // random per power-up
    uint32_t send_fail;        // sends that still failed after retries
    uint32_t send_retry;       // retries performed
    int16_t last_errno;        // errno of the last failed try
    uint32_t late_reads;       // read intervals above MIC_LATE_READ_MS
    uint16_t max_read_gap_ms;  // largest late interval since the last report
};

inline void mic_stats_init(MicStreamStats *s, uint8_t boot_id) {
    s->boot_id = boot_id;
    s->seq = 0; s->send_fail = 0; s->send_retry = 0;
    s->last_errno = 0; s->late_reads = 0; s->max_read_gap_ms = 0;
}

// Returns the sequence number for this frame and advances (wraps at 65535).
// Advances whether or not the send later succeeds, so failures show as gaps.
inline uint16_t mic_stats_next_seq(MicStreamStats *s) { return s->seq++; }

// Writes the 5-byte trailer at buf[0..4].
inline void mic_trailer_write(uint8_t *buf, uint8_t boot_id, uint16_t seq) {
    buf[0] = MIC_TRAILER_MAGIC;
    buf[1] = MIC_TRAILER_VERSION;
    buf[2] = boot_id;
    buf[3] = (uint8_t)(seq >> 8);
    buf[4] = (uint8_t)(seq & 0xFF);
}

// Interval in ms between two consecutive completed reads.
inline void mic_stats_record_read_interval(MicStreamStats *s, uint32_t interval_ms) {
    if (interval_ms <= MIC_LATE_READ_MS) return;
    s->late_reads++;
    uint16_t v = interval_ms > 65535u ? 65535u : (uint16_t)interval_ms;
    if (v > s->max_read_gap_ms) s->max_read_gap_ms = v;
}

// retries = extra tries performed for this frame; ok = final result.
inline void mic_stats_record_send(MicStreamStats *s, bool ok, uint32_t retries, int err) {
    s->send_retry += retries;
    if (!ok) {
        s->send_fail++;
        s->last_errno = (int16_t)err;
    }
}

static inline void mic_put16(uint8_t *p, uint16_t v) { p[0] = (uint8_t)(v >> 8); p[1] = (uint8_t)v; }
static inline void mic_put32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16); p[2] = (uint8_t)(v >> 8); p[3] = (uint8_t)v;
}

// Packs MIC_REPORT_SIZE bytes into out, then resets max_read_gap_ms.
// The seq field is the next sequence number to be used.
inline size_t mic_stats_pack_report(MicStreamStats *s, int8_t rssi, uint8_t channel,
                                    uint8_t bssid_last, uint32_t stack_free_bytes, uint8_t *out) {
    out[0] = 'O'; out[1] = 'J'; out[2] = 'S'; out[3] = 'T';
    out[4] = MIC_REPORT_VERSION;
    mic_put16(out + 5, s->seq);
    mic_put32(out + 7, s->send_fail);
    mic_put32(out + 11, s->send_retry);
    mic_put32(out + 15, s->late_reads);
    mic_put16(out + 19, s->max_read_gap_ms);
    mic_put16(out + 21, (uint16_t)s->last_errno);
    out[23] = (uint8_t)rssi;
    out[24] = channel;
    out[25] = bssid_last;
    mic_put16(out + 26, stack_free_bytes > 65535u ? 65535u : (uint16_t)stack_free_bytes);
    out[28] = s->boot_id;
    s->max_read_gap_ms = 0;
    return MIC_REPORT_SIZE;
}

#endif
