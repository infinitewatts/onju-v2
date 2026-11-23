// Native check of the firmware mu-law encoder against its own decode table.
#include <cstdio>
#include <cstdlib>
#include "../onjuino/audio_compression.h"

static void expect(bool ok, const char *what) {
    if (!ok) { std::printf("FAIL: %s\n", what); std::exit(1); }
}

int main() {
    // Loud speech must survive: the old 0x1FFF ceiling returned 8316 for all of these.
    expect(ulaw_to_linear(linear_to_ulaw(20000)) > 19000, "20000 keeps its level");
    expect(ulaw_to_linear(linear_to_ulaw(-20000)) < -19000, "-20000 keeps its level");
    expect(ulaw_to_linear(linear_to_ulaw(32767)) == 32124, "positive full scale");
    expect(ulaw_to_linear(linear_to_ulaw(-32768)) == -32124, "negative full scale does not wrap");
    // Every level stays within one mu-law step of itself.
    for (int v = -32768; v <= 32767; v++) {
        int clamped = v > 32635 ? 32635 : (v < -32635 ? -32635 : v);
        int back = ulaw_to_linear(linear_to_ulaw((int16_t)v));
        int magnitude = clamped < 0 ? -clamped : clamped;
        int tolerance = 8 + magnitude / 16;
        expect(std::abs(back - clamped) <= tolerance, "round trip within one step");
        expect(v == 0 || (back >= 0) == (v > 0) || back == 0, "sign preserved");
    }
    std::printf("PASS: mu-law\n");
    return 0;
}
