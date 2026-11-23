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
#define MIC_GARBAGE_LEVEL 6000     // sustained level no real room produces: an unconnected or faulty slot
#define MIC_ALIVE_LEVEL 4          // below this a mic is producing nothing at all
#define MIC_EXCLUDE_BLOCKS 312     // keep a rejected mic out for about 10 s after the last bad block
#define MIC_LAG_CONFIRM 6          // blocks a new delay must win in a row before it is used

enum MicMode { MIC_BOTH = 0, MIC_FIRST_ONLY = 1, MIC_SECOND_ONLY = 2 };

struct MicCombiner {
    float dc[2];                   // slow DC trackers, one per microphone
    float level[2];                // smoothed mean absolute level per microphone
    float corr[MIC_LAGS];          // smoothed alignment score per candidate delay
    int16_t tail[2][MIC_MAX_LAG];  // last samples of the previous block
    int lag;                       // chosen delay: >0 delays the first mic, <0 the second
    int lag_candidate;             // delay waiting to be confirmed
    int lag_streak;                // blocks in a row the candidate has won
    int excluded[2];               // blocks left during which each mic stays rejected
    bool primed;                   // first block seen
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
        // The first block sets the level outright, so a slow ramp from zero
        // cannot make a healthy mic look dead next to a noisy one.
        float now_level = (float)sum_abs[c] / frames;
        if (!m->primed) m->level[c] = now_level;
        else m->level[c] += (now_level - m->level[c]) / 16.0f;
    }

    m->primed = true;

    // Reject a mic that is far quieter than the other (dead or blocked), or one
    // that is impossibly loud while the other is alive and quiet (a slot with no
    // working mic reads as noise). A rejected mic stays out for a while, so the
    // choice cannot flip back and forth mid-sentence.
    for (int c = 0; c < 2; c++) {
        float mine = m->level[c], other = m->level[1 - c];
        bool weak = mine * MIC_DEAD_RATIO < other && other <= MIC_GARBAGE_LEVEL;
        bool garbage = mine > MIC_GARBAGE_LEVEL && mine > other * MIC_DEAD_RATIO && other >= MIC_ALIVE_LEVEL;
        if (weak || garbage) m->excluded[c] = MIC_EXCLUDE_BLOCKS;
        else if (m->excluded[c] > 0) m->excluded[c]--;
    }
    if (m->excluded[1] > 0 && m->excluded[0] == 0) m->mode = MIC_FIRST_ONLY;
    else if (m->excluded[0] > 0 && m->excluded[1] == 0) m->mode = MIC_SECOND_ONLY;
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
    // Change the delay only when another one is clearly better than the one
    // in use, the two mics really are hearing the same sound (not just their
    // own hiss), and it keeps winning for several blocks. A true delay that
    // falls between two whole samples would otherwise flip back and forth,
    // and every flip skips or repeats a sample.
    float coherent = 0.25f * m->level[0] * m->level[1];
    int current = m->lag + MIC_MAX_LAG;
    if (best != current && m->corr[best] > m->corr[current] * 1.03f && m->corr[best] > coherent) {
        if (best - MIC_MAX_LAG == m->lag_candidate) m->lag_streak++;
        else { m->lag_candidate = best - MIC_MAX_LAG; m->lag_streak = 1; }
        if (m->lag_streak >= MIC_LAG_CONFIRM) { m->lag = m->lag_candidate; m->lag_streak = 0; }
    } else {
        m->lag_streak = 0;
    }

    int d0 = m->lag > 0 ? m->lag : 0, d1 = m->lag < 0 ? -m->lag : 0;
    const int16_t *a = ch[0] + MIC_MAX_LAG - d0, *b = ch[1] + MIC_MAX_LAG - d1;
    for (int i = 0; i < frames; i++) out[i] = (int16_t)(((int32_t)a[i] + b[i]) / 2);
}

#endif
