#ifndef GTCRN_AUDIO_H
#define GTCRN_AUDIO_H

#include <stddef.h>
#include <stdint.h>

#include "gtcrn_model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Offline reference front end matching torch.stft/istft center=True. */
int gtcrn_enhance_audio_f32(gtcrn_model_t *model,
                            const float *input,
                            size_t sample_count,
                            float *output);
int gtcrn_stft_frame_f32(const float *input, size_t sample_count,
                         size_t frame_index, float output[GTCRN_BINS][2]);
int gtcrn_irfft_frame_f32(const float input[GTCRN_BINS][2],
                          float output[GTCRN_FFT_SIZE]);

/* Heap-free causal 256-sample streaming front end (one-hop algorithmic delay). */
typedef struct {
    gtcrn_model_t *model;
    void *forward_fft;
    void *inverse_fft;
    /* 6 KiB each; KissFFT 512-point real plans require less on 32/64-bit builds. */
    uint64_t forward_storage[768];
    uint64_t inverse_storage[768];
    float input_history[GTCRN_HOP_SIZE];
    float ola_tail[GTCRN_HOP_SIZE];
    float window[GTCRN_FFT_SIZE];
} gtcrn_audio_stream_t;

int gtcrn_audio_stream_init(gtcrn_audio_stream_t *stream, gtcrn_model_t *model);
size_t gtcrn_audio_stream_sizeof(void);
void gtcrn_audio_stream_reset(gtcrn_audio_stream_t *stream);
int gtcrn_audio_stream_process(gtcrn_audio_stream_t *stream,
                               const float input[GTCRN_HOP_SIZE],
                               float output[GTCRN_HOP_SIZE]);

#ifdef __cplusplus
}
#endif

#endif
