#ifndef GTCRN_H
#define GTCRN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    GTCRN_SAMPLE_RATE = 16000,
    GTCRN_FFT_SIZE = 512,
    GTCRN_HOP_SIZE = 256,
    GTCRN_BINS = 257,
    GTCRN_ERB_BINS = 129,
    GTCRN_CORE_BINS = 33,
    GTCRN_CHANNELS = 16,
    GTCRN_CONV_HISTORY = 16,
    GTCRN_TRA_LAYERS = 3,
    GTCRN_PATHS = 2
};

typedef struct {
    /* Encoder/decoder causal feature history. */
    int8_t conv[GTCRN_PATHS][GTCRN_CHANNELS]
               [GTCRN_CONV_HISTORY][GTCRN_CORE_BINS];
    /* TRA hidden size is 16; Q15 avoids recurrent INT8 drift. */
    int16_t tra[GTCRN_PATHS][GTCRN_TRA_LAYERS][GTCRN_CHANNELS];
    /* Two DPGRNN blocks, one hidden vector for every compressed band. */
    int16_t inter[GTCRN_PATHS][GTCRN_CORE_BINS][GTCRN_CHANNELS];
} gtcrn_state_t;

typedef struct {
    int16_t real;
    int16_t imag;
} gtcrn_complex_q15_t;

void gtcrn_state_reset(gtcrn_state_t *state);

#ifdef __cplusplus
}
#endif

#endif
