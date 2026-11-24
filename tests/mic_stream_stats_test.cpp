// Native check of the microphone stream counters used by the firmware.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "../onjuino/mic_stream_stats.h"

static void expect(bool ok, const char *what) {
    if (!ok) { std::printf("FAIL: %s\n", what); std::exit(1); }
}

int main() {
    static_assert(MIC_PACKET_SIZE == 516, "audio packet size");
    static_assert(MIC_REPORT_SIZE == 26, "report size");

    uint8_t t[4];
    mic_trailer_write(t, 0x1234);
    expect(t[0] == 0xA5 && t[1] == 0x01 && t[2] == 0x12 && t[3] == 0x34, "trailer bytes");

    MicStreamStats s;
    mic_stats_init(&s);
    expect(mic_stats_next_seq(&s) == 0 && mic_stats_next_seq(&s) == 1, "seq starts at 0");
    s.seq = 65535;
    expect(mic_stats_next_seq(&s) == 65535, "last seq");
    expect(s.seq == 0, "seq wraps to 0");

    mic_stats_record_read_interval(&s, 32);
    mic_stats_record_read_interval(&s, 64);
    expect(s.late_reads == 0 && s.max_read_gap_ms == 0, "on-time reads ignored");
    mic_stats_record_read_interval(&s, 65);
    mic_stats_record_read_interval(&s, 200);
    mic_stats_record_read_interval(&s, 100);
    expect(s.late_reads == 3 && s.max_read_gap_ms == 200, "late reads and max");
    mic_stats_record_read_interval(&s, 1000000);
    expect(s.max_read_gap_ms == 65535, "max saturates");

    mic_stats_record_send(&s, true, 0, 0);
    expect(s.send_fail == 0 && s.send_retry == 0 && s.last_errno == 0, "clean send");
    mic_stats_record_send(&s, true, 1, 12);
    expect(s.send_retry == 1 && s.send_fail == 0 && s.last_errno == 0, "retry then ok");
    mic_stats_record_send(&s, false, 2, 12);
    expect(s.send_retry == 3 && s.send_fail == 1 && s.last_errno == 12, "final failure");

    s.seq = 0xABCD; s.send_fail = 0x01020304; s.send_retry = 0x05060708;
    s.late_reads = 0x090A0B0C; s.max_read_gap_ms = 0x0D0E; s.last_errno = -2;
    uint8_t r[MIC_REPORT_SIZE];
    expect(mic_stats_pack_report(&s, -67, 11, 0xEF, r) == 26, "report size returned");
    const uint8_t want[26] = {'O','J','S','T', 0x01, 0xAB,0xCD, 1,2,3,4, 5,6,7,8, 9,10,11,12,
                              0x0D,0x0E, 0xFF,0xFE, (uint8_t)-67, 11, 0xEF};
    expect(std::memcmp(r, want, 26) == 0, "report bytes");
    expect(s.max_read_gap_ms == 0, "report resets max gap");
    expect(s.late_reads == 0x090A0B0C, "report keeps other counters");

    std::printf("PASS: mic stream stats\n");
    return 0;
}
