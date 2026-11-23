// Two-microphone combiner. Plain C++ (no Arduino dependencies) so it can be
// tested on a computer: tests/mic_combine_test.cpp.
//
// Both microphones hear the talker, one slightly later than the other. The
// combiner finds that delay (up to MIC_MAX_LAG samples), lines the two signals
// up and averages them. Speech adds up; each microphone's own hiss does not.
// If one microphone is dead or far quieter than the other, only the good one
// is used, so the result is never worse than a single microphone.
#ifndef MIC_COMBINE_H
#define MIC_COMBINE_H

#include <stdint.h>
#include <string.h>

#define MIC_MAX_LAG 3              // samples; 3 at 16 kHz covers mics up to ~64 mm apart
#define MIC_LAGS (2 * MIC_MAX_LAG + 1)
#define MIC_DEAD_RATIO 8           // a mic this many times quieter than the other is ignored

enum MicMode { MIC_BOTH = 0, MIC_FIRST_ONLY = 1, MIC_SECOND_ONLY = 2 };

struct MicCombiner {
    float dc[2];                   // slow DC trackers, one per microphone
    float level[2];                // smoothed mean absolute level per microphone
    float corr[MIC_LAGS];          // smoothed alignment score per candidate delay
    int16_t tail[2][MIC_MAX_LAG];  // last samples of the previous block
    int lag;                       // chosen delay: >0 delays the first mic, <0 the second
    MicMode mode;
};

static inline void mic_combiner_init(MicCombiner *m, float dc_first, float dc_second) {
    memset(m, 0, sizeof(*m));
    m->dc[0] = dc_first;
    m->dc[1] = dc_second;
    m->mode = MIC_BOTH;
}

static inline int16_t mic_saturate(int32_t v) {
    return v > 32767 ? 32767 : (v < -32768 ? -32768 : (int16_t)v);
}

// raw: interleaved 32-bit I2S words (first mic, second mic, ...), `frames` pairs.
// work: scratch for 2 * (MIC_MAX_LAG + frames) int16 values.
// out: `frames` combined samples.
static inline void mic_combine(MicCombiner *m, const int32_t *raw, int frames,
                               int16_t *work, int16_t *out, float dc_alpha) {
    int16_t *ch[2] = { work, work + MIC_MAX_LAG + frames };
    uint32_t sum_abs[2] = { 0, 0 };

    for (int c = 0; c < 2; c++) {
        memcpy(ch[c], m->tail[c], sizeof(m->tail[c]));
        float dc = m->dc[c];
        for (int i = 0; i < frames; i++) {
            // 18-bit sample: remove DC in 32 bits, then saturate (never wrap).
            int32_t sample = raw[2 * i + c] >> 14;
            dc += dc_alpha * (sample - dc);
            int16_t centered = mic_saturate(sample - (int32_t)dc);
            ch[c][MIC_MAX_LAG + i] = centered;
            sum_abs[c] += (uint32_t)(centered < 0 ? -(int32_t)centered : centered);
        }
        m->dc[c] = dc;
        memcpy(m->tail[c], ch[c] + frames, sizeof(m->tail[c]));
        m->level[c] += ((float)sum_abs[c] / frames - m->level[c]) / 16.0f;
    }

    if (m->level[1] * MIC_DEAD_RATIO < m->level[0]) m->mode = MIC_FIRST_ONLY;
    else if (m->level[0] * MIC_DEAD_RATIO < m->level[1]) m->mode = MIC_SECOND_ONLY;
    else m->mode = MIC_BOTH;

    if (m->mode != MIC_BOTH) {
        const int16_t *only = ch[m->mode == MIC_FIRST_ONLY ? 0 : 1] + MIC_MAX_LAG;
        memcpy(out, only, frames * sizeof(int16_t));
        return;
    }

    // Score each candidate delay. Loud, coherent sound (a talker) dominates the
    // smoothed score; uncorrelated hiss averages out near zero.
    int best = MIC_MAX_LAG;  // index of zero delay
    for (int k = 0; k < MIC_LAGS; k++) {
        int lag = k - MIC_MAX_LAG;
        int d0 = lag > 0 ? lag : 0, d1 = lag < 0 ? -lag : 0;
        const int16_t *a = ch[0] + MIC_MAX_LAG - d0, *b = ch[1] + MIC_MAX_LAG - d1;
        int64_t acc = 0;
        for (int i = 0; i < frames; i++) acc += (int32_t)a[i] * b[i];
        m->corr[k] += ((float)acc / frames - m->corr[k]) * 0.1f;
        if (m->corr[k] > m->corr[best]) best = k;
    }
    // Keep zero delay unless another delay is clearly better and the two
    // microphones really are hearing the same sound (not just their own hiss).
    float coherent = 0.25f * m->level[0] * m->level[1];
    if (m->corr[best] <= m->corr[MIC_MAX_LAG] * 1.05f || m->corr[best] <= coherent) best = MIC_MAX_LAG;
    m->lag = best - MIC_MAX_LAG;

    int d0 = m->lag > 0 ? m->lag : 0, d1 = m->lag < 0 ? -m->lag : 0;
    const int16_t *a = ch[0] + MIC_MAX_LAG - d0, *b = ch[1] + MIC_MAX_LAG - d1;
    for (int i = 0; i < frames; i++) out[i] = (int16_t)(((int32_t)a[i] + b[i]) / 2);
}

#endif
