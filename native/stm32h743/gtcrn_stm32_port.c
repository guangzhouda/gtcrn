#include "gtcrn_stm32_port.h"

#include <math.h>

int gtcrn_stm32_init(gtcrn_stm32_context_t *context,
                     const uint8_t *weights, size_t weights_size) {
    if (context == NULL) return GTCRN_MODEL_BAD_ARGUMENT;
    int status = gtcrn_model_init(&context->model, weights, weights_size);
    if (status != GTCRN_MODEL_OK) return status;
    return gtcrn_audio_stream_init(&context->audio, &context->model);
}

int gtcrn_stm32_process_s16(gtcrn_stm32_context_t *context,
                            const int16_t input[GTCRN_HOP_SIZE],
                            int16_t output[GTCRN_HOP_SIZE]) {
    if (context == NULL || input == NULL || output == NULL)
        return GTCRN_MODEL_BAD_ARGUMENT;
    float input_f32[GTCRN_HOP_SIZE], output_f32[GTCRN_HOP_SIZE];
    for (size_t i = 0; i < GTCRN_HOP_SIZE; ++i)
        input_f32[i] = input[i] * (1.0f / 32768.0f);
    int status = gtcrn_audio_stream_process(&context->audio, input_f32, output_f32);
    if (status != GTCRN_MODEL_OK) return status;
    for (size_t i = 0; i < GTCRN_HOP_SIZE; ++i) {
        long value = lroundf(output_f32[i] * 32768.0f);
        if (value > 32767) value = 32767;
        if (value < -32768) value = -32768;
        output[i] = (int16_t)value;
    }
    return GTCRN_MODEL_OK;
}
