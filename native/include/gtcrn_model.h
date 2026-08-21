#ifndef GTCRN_MODEL_H
#define GTCRN_MODEL_H

#include <stddef.h>
#include <stdint.h>

#include "gtcrn.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    GTCRN_MODEL_OK = 0,
    GTCRN_MODEL_BAD_ARGUMENT = -1,
    GTCRN_MODEL_BAD_WEIGHTS = -2
};

typedef struct {
    const uint8_t *weights_blob;
    size_t weights_blob_size;

#ifdef GTCRN_Q15_STATE
    int16_t enc_conv_cache[3][GTCRN_CHANNELS][10][GTCRN_CORE_BINS];
    int16_t dec_conv_cache[3][GTCRN_CHANNELS][10][GTCRN_CORE_BINS];
    int16_t enc_tra_cache[3][GTCRN_CHANNELS];
    int16_t dec_tra_cache[3][GTCRN_CHANNELS];
    int16_t inter_cache[2][GTCRN_CORE_BINS][GTCRN_CHANNELS];
#else
    float enc_conv_cache[3][GTCRN_CHANNELS][10][GTCRN_CORE_BINS];
    float dec_conv_cache[3][GTCRN_CHANNELS][10][GTCRN_CORE_BINS];
    float enc_tra_cache[3][GTCRN_CHANNELS];
    float dec_tra_cache[3][GTCRN_CHANNELS];
    float inter_cache[2][GTCRN_CORE_BINS][GTCRN_CHANNELS];
#endif

    float en_out0[GTCRN_CHANNELS][65];
    float en_out1[GTCRN_CHANNELS][GTCRN_CORE_BINS];
    float en_out2[GTCRN_CHANNELS][GTCRN_CORE_BINS];
    float en_out3[GTCRN_CHANNELS][GTCRN_CORE_BINS];
    float en_out4[GTCRN_CHANNELS][GTCRN_CORE_BINS];
    float scratch_a[24][GTCRN_ERB_BINS];
    float scratch_b[24][GTCRN_ERB_BINS];
    float scratch_c[GTCRN_CORE_BINS][GTCRN_CHANNELS];
    float scratch_d[GTCRN_CORE_BINS][GTCRN_CHANNELS];
} gtcrn_model_t;

int gtcrn_model_init(gtcrn_model_t *model,
                     const uint8_t *weights_blob,
                     size_t weights_blob_size);

void gtcrn_model_reset(gtcrn_model_t *model);

int gtcrn_model_process_frame(gtcrn_model_t *model,
                              const float input_spec[GTCRN_BINS][2],
                              float output_spec[GTCRN_BINS][2]);

size_t gtcrn_model_sizeof(void);
int gtcrn_model_init_f32(void *model, const uint8_t *weights_blob,
                         size_t weights_blob_size);
int gtcrn_process_frame_f32(void *model, const float *input_ri,
                            float *output_ri);

#ifdef __cplusplus
}
#endif

#endif
