#ifndef GTCRN_STM32_PORT_H
#define GTCRN_STM32_PORT_H

#include "gtcrn_audio.h"

#include <stdint.h>

typedef struct {
    gtcrn_model_t model;
    gtcrn_audio_stream_t audio;
} gtcrn_stm32_context_t;

int gtcrn_stm32_init(gtcrn_stm32_context_t *context,
                     const uint8_t *weights, size_t weights_size);
int gtcrn_stm32_process_s16(gtcrn_stm32_context_t *context,
                            const int16_t input[GTCRN_HOP_SIZE],
                            int16_t output[GTCRN_HOP_SIZE]);

#endif
