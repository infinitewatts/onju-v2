// Native check of the two-microphone combiner used by the firmware.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../onjuino/mic_combine.h"

static void expect(bool ok, const char *what) {
    if (!ok) { std::printf("FAIL: %s\n", what); std::exit(1); }
}

static const int FRAMES = 512;
static uint32_t seed = 12345;
static double noise() {  // uniform in [-1, 1)
    seed = seed * 1664525u + 1013904223u;
    return (double)(int32_t)seed / 2147483648.0;
}
static int32_t word(double sample16) {  // 16-bit level as the 18-bit-in-32 I2S word
    return (int32_t)((int64_t)llround(sample16) * (1 << 14));
}
static double speech(int n) {  // broadband, speech-band test signal
    return 3000.0 * (std::sin(n * 0.11) + 0.7 * std::sin(n * 0.37 + 1.0) + 0.5 * std::sin(n * 0.83 + 2.0));
}

struct Result { double signal_err; double noise_power; int lag; MicMode mode; };

// Feed `blocks` blocks; mic 2 hears the talker `delay` samples after mic 1 (negative: before).
static Result run(int delay, double hiss, double gain2, int blocks, double dc1 = 0, double dc2 = 0) {
    MicCombiner m;
    mic_combiner_init(&m, (float)dc1, (float)dc2);
    std::vector<int32_t> raw(2 * FRAMES);
    std::vector<int16_t> work(2 * (MIC_MAX_LAG + FRAMES)), out(FRAMES);
    Result r = {0, 0, 0, MIC_BOTH};
    double err = 0, npow = 0; long count = 0;
    for (int b = 0; b < blocks; b++) {
        std::vector<double> n1(FRAMES), n2(FRAMES);
        for (int i = 0; i < FRAMES; i++) {
            int n = b * FRAMES + i;
            n1[i] = hiss * noise(); n2[i] = hiss * noise();
            raw[2 * i] = word(speech(n) + n1[i] + dc1);
            raw[2 * i + 1] = word(gain2 * (speech(n - delay) + n2[i]) + dc2);
        }
        mic_combine(&m, raw.data(), FRAMES, work.data(), out.data(), 0.001f);
        if (b >= blocks / 2) {  // after settling
            int late = delay > 0 ? delay : 0;  // output is aligned to the later arrival
            for (int i = 8; i < FRAMES; i++) {
                int n = b * FRAMES + i;
                double want = speech(n - late);
                err += (out[i] - want) * (out[i] - want);
                npow += hiss * hiss / 3.0;  // variance of one mic's hiss
                count++;
            }
        }
    }
    r.signal_err = err / count; r.noise_power = npow / count; r.lag = m.lag; r.mode = m.mode;
    return r;
}

int main() {
    // The delay between the microphones is found in both directions and at zero.
    for (int delay = -MIC_MAX_LAG; delay <= MIC_MAX_LAG; delay++) {
        Result r = run(delay, 300.0, 1.0, 60);
        expect(r.mode == MIC_BOTH, "both microphones used when both are healthy");
        expect(r.lag == delay, "delay between microphones found");
        // Averaging two aligned mics halves the hiss power (within tolerance).
        expect(r.signal_err < 0.65 * r.noise_power, "hiss reduced versus one microphone");
    }
    // A dead second microphone (silent) or dead first microphone falls back to the live one.
    Result dead2 = run(0, 300.0, 0.0, 60);
    expect(dead2.mode == MIC_FIRST_ONLY, "silent second microphone ignored");
    expect(dead2.signal_err < 1.2 * dead2.noise_power, "single-microphone quality kept");
    {
        MicCombiner m; mic_combiner_init(&m, 0, 0);
        std::vector<int32_t> raw(2 * FRAMES);
        std::vector<int16_t> work(2 * (MIC_MAX_LAG + FRAMES)), out(FRAMES);
        for (int b = 0; b < 60; b++) {
            for (int i = 0; i < FRAMES; i++) { raw[2 * i] = 0; raw[2 * i + 1] = word(speech(b * FRAMES + i)); }
            mic_combine(&m, raw.data(), FRAMES, work.data(), out.data(), 0.001f);
        }
        expect(m.mode == MIC_SECOND_ONLY, "silent first microphone ignored");
        expect(std::abs(out[100] - (int)llround(speech(59 * FRAMES + 100))) < 40, "second microphone passed through");
    }
    // Large, different DC offsets are removed from each microphone.
    Result dc = run(1, 300.0, 1.0, 120, 5000.0, -7000.0);
    expect(dc.lag == 1 && dc.signal_err < 0.9 * dc.noise_power, "per-microphone DC removed");
    // Full-scale input saturates instead of wrapping, and only hiss keeps zero delay.
    {
        MicCombiner m; mic_combiner_init(&m, 0, 0);
        std::vector<int32_t> raw(2 * FRAMES);
        std::vector<int16_t> work(2 * (MIC_MAX_LAG + FRAMES)), out(FRAMES);
        for (int i = 0; i < FRAMES; i++) { raw[2 * i] = raw[2 * i + 1] = (i % 64 < 32) ? INT32_MAX : INT32_MIN; }
        mic_combine(&m, raw.data(), FRAMES, work.data(), out.data(), 0.0f);
        expect(out[10] == 32767 && out[40] == -32768, "loud input clips, never wraps");
        mic_combiner_init(&m, 0, 0);
        for (int b = 0; b < 60; b++) {
            for (int i = 0; i < FRAMES; i++) { raw[2 * i] = word(300 * noise()); raw[2 * i + 1] = word(300 * noise()); }
            mic_combine(&m, raw.data(), FRAMES, work.data(), out.data(), 0.001f);
        }
        expect(m.lag == 0 && m.mode == MIC_BOTH, "hiss alone keeps zero delay");
    }
    // A delay between two whole samples settles on one neighbour and stays there.
    for (double frac : {0.5, 1.5, 2.5}) {
        MicCombiner m; mic_combiner_init(&m, 0, 0);
        std::vector<int32_t> raw(2 * FRAMES);
        std::vector<int16_t> work(2 * (MIC_MAX_LAG + FRAMES)), out(FRAMES);
        int changes = 0, last = 0;
        for (int b = 0; b < 200; b++) {
            for (int i = 0; i < FRAMES; i++) {
                double n = b * FRAMES + i;
                double s1 = 3000.0 * (std::sin(n * 0.11) + 0.7 * std::sin(n * 0.37 + 1.0) + 0.5 * std::sin(n * 0.83 + 2.0));
                double d = n - frac;
                double s2 = 3000.0 * (std::sin(d * 0.11) + 0.7 * std::sin(d * 0.37 + 1.0) + 0.5 * std::sin(d * 0.83 + 2.0));
                raw[2 * i] = word(s1 + 300 * noise()); raw[2 * i + 1] = word(s2 + 300 * noise());
            }
            mic_combine(&m, raw.data(), FRAMES, work.data(), out.data(), 0.001f);
            if (b >= 40 && m.lag != last) changes++;
            last = m.lag;
        }
        expect(changes == 0, "delay does not flip once settled");
        expect(m.lag == (int)frac || m.lag == (int)frac + 1, "fractional delay lands on a neighbouring sample");
    }
    // A slot with no working mic reads as loud noise: it is rejected, during speech too.
    {
        MicCombiner m; mic_combiner_init(&m, 0, 0);
        std::vector<int32_t> raw(2 * FRAMES);
        std::vector<int16_t> work(2 * (MIC_MAX_LAG + FRAMES)), out(FRAMES);
        int wrong = 0;
        for (int b = 0; b < 200; b++) {
            bool talking = b >= 100;
            for (int i = 0; i < FRAMES; i++) {
                double real = (talking ? 2.0 * speech(b * FRAMES + i) : 0) + 40 * noise();
                raw[2 * i] = word(real);
                raw[2 * i + 1] = (int32_t)(noise() * 2147483000.0);
            }
            mic_combine(&m, raw.data(), FRAMES, work.data(), out.data(), 0.001f);
            if (b >= 30 && m.mode != MIC_FIRST_ONLY) wrong++;
        }
        expect(wrong == 0, "noise-filled slot stays rejected while the good mic hears speech");
    }
    // A mic right at the quiet threshold does not flip the choice back and forth.
    {
        MicCombiner m; mic_combiner_init(&m, 0, 0);
        std::vector<int32_t> raw(2 * FRAMES);
        std::vector<int16_t> work(2 * (MIC_MAX_LAG + FRAMES)), out(FRAMES);
        int changes = 0; MicMode last = MIC_BOTH;
        for (int b = 0; b < 400; b++) {
            for (int i = 0; i < FRAMES; i++) {
                double s = speech(b * FRAMES + i);
                raw[2 * i] = word(s + 300 * noise()); raw[2 * i + 1] = word(0.125 * (s + 300 * noise()));
            }
            mic_combine(&m, raw.data(), FRAMES, work.data(), out.data(), 0.001f);
            if (b >= 40 && m.mode != last) changes++;
            last = m.mode;
        }
        expect(changes <= 1, "choice of mic does not flap at the threshold");
    }
    std::printf("PASS: mic combiner\n");
    return 0;
}
