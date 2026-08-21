#include "gtcrn_audio.h"

#include "kiss_fftr.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static size_t reflect_index(long index, size_t length) {
    if (length < 2) return 0;
    while (index < 0 || index >= (long)length) {
        if (index < 0) index = -index;
        if (index >= (long)length) index = 2L * (long)length - 2L - index;
    }
    return (size_t)index;
}

int gtcrn_audio_stream_init(gtcrn_audio_stream_t *stream, gtcrn_model_t *model) {
    if (stream == NULL || model == NULL) return GTCRN_MODEL_BAD_ARGUMENT;
    memset(stream, 0, sizeof(*stream));
    stream->model = model;
    size_t forward_bytes = sizeof(stream->forward_storage);
    size_t inverse_bytes = sizeof(stream->inverse_storage);
    stream->forward_fft = kiss_fftr_alloc(GTCRN_FFT_SIZE, 0,
                                          stream->forward_storage, &forward_bytes);
    stream->inverse_fft = kiss_fftr_alloc(GTCRN_FFT_SIZE, 1,
                                          stream->inverse_storage, &inverse_bytes);
    if (stream->forward_fft == NULL || stream->inverse_fft == NULL)
        return GTCRN_MODEL_BAD_ARGUMENT;
    for (size_t n = 0; n < GTCRN_FFT_SIZE; ++n) {
        const float hann = 0.5f - 0.5f * cosf((float)(2.0 * M_PI * n / GTCRN_FFT_SIZE));
        stream->window[n] = sqrtf(hann);
    }
    return GTCRN_MODEL_OK;
}

size_t gtcrn_audio_stream_sizeof(void) {
    return sizeof(gtcrn_audio_stream_t);
}

void gtcrn_audio_stream_reset(gtcrn_audio_stream_t *stream) {
    if (stream == NULL) return;
    memset(stream->input_history, 0, sizeof(stream->input_history));
    memset(stream->ola_tail, 0, sizeof(stream->ola_tail));
    if (stream->model != NULL) gtcrn_model_reset(stream->model);
}

int gtcrn_audio_stream_process(gtcrn_audio_stream_t *stream,
                               const float input[GTCRN_HOP_SIZE],
                               float output[GTCRN_HOP_SIZE]) {
    if (stream == NULL || stream->model == NULL || input == NULL || output == NULL ||
        stream->forward_fft == NULL || stream->inverse_fft == NULL)
        return GTCRN_MODEL_BAD_ARGUMENT;
    kiss_fft_scalar time_frame[GTCRN_FFT_SIZE];
    kiss_fft_cpx frequency[GTCRN_BINS];
    float spectrum[GTCRN_BINS][2], enhanced[GTCRN_BINS][2];
    for (size_t n = 0; n < GTCRN_HOP_SIZE; ++n) {
        time_frame[n] = stream->input_history[n] * stream->window[n];
        time_frame[GTCRN_HOP_SIZE + n] = input[n] * stream->window[GTCRN_HOP_SIZE + n];
        stream->input_history[n] = input[n];
    }
    kiss_fftr((kiss_fftr_cfg)stream->forward_fft, time_frame, frequency);
    for (size_t bin = 0; bin < GTCRN_BINS; ++bin) {
        spectrum[bin][0] = frequency[bin].r;
        spectrum[bin][1] = frequency[bin].i;
    }
    int status = gtcrn_model_process_frame(stream->model, spectrum, enhanced);
    if (status != GTCRN_MODEL_OK) return status;
    for (size_t bin = 0; bin < GTCRN_BINS; ++bin) {
        frequency[bin].r = enhanced[bin][0];
        frequency[bin].i = enhanced[bin][1];
    }
    kiss_fftri((kiss_fftr_cfg)stream->inverse_fft, frequency, time_frame);
    for (size_t n = 0; n < GTCRN_HOP_SIZE; ++n) {
        const float first = time_frame[n] * stream->window[n] / GTCRN_FFT_SIZE;
        const float second = time_frame[GTCRN_HOP_SIZE + n]
                           * stream->window[GTCRN_HOP_SIZE + n] / GTCRN_FFT_SIZE;
        output[n] = stream->ola_tail[n] + first;
        stream->ola_tail[n] = second;
    }
    return GTCRN_MODEL_OK;
}

int gtcrn_stft_frame_f32(const float *input, size_t sample_count,
                         size_t frame_index, float output[GTCRN_BINS][2]) {
    if (input == NULL || output == NULL || sample_count < 2)
        return GTCRN_MODEL_BAD_ARGUMENT;
    kiss_fftr_cfg forward = kiss_fftr_alloc(GTCRN_FFT_SIZE, 0, NULL, NULL);
    if (forward == NULL) return GTCRN_MODEL_BAD_ARGUMENT;
    kiss_fft_scalar time_frame[GTCRN_FFT_SIZE];
    kiss_fft_cpx frequency[GTCRN_BINS];
    const size_t offset = frame_index * GTCRN_HOP_SIZE;
    for (size_t n = 0; n < GTCRN_FFT_SIZE; ++n) {
        const float hann = 0.5f - 0.5f * cosf((float)(2.0 * M_PI * n / GTCRN_FFT_SIZE));
        const long source = (long)(offset + n) - GTCRN_FFT_SIZE / 2;
        time_frame[n] = input[reflect_index(source, sample_count)] * sqrtf(hann);
    }
    kiss_fftr(forward, time_frame, frequency);
    for (size_t bin = 0; bin < GTCRN_BINS; ++bin) {
        output[bin][0] = frequency[bin].r;
        output[bin][1] = frequency[bin].i;
    }
    free(forward);
    return GTCRN_MODEL_OK;
}

int gtcrn_irfft_frame_f32(const float input[GTCRN_BINS][2],
                          float output[GTCRN_FFT_SIZE]) {
    if (input == NULL || output == NULL) return GTCRN_MODEL_BAD_ARGUMENT;
    kiss_fftr_cfg inverse = kiss_fftr_alloc(GTCRN_FFT_SIZE, 1, NULL, NULL);
    if (inverse == NULL) return GTCRN_MODEL_BAD_ARGUMENT;
    kiss_fft_cpx frequency[GTCRN_BINS];
    kiss_fft_scalar time_frame[GTCRN_FFT_SIZE];
    for (size_t bin = 0; bin < GTCRN_BINS; ++bin) {
        frequency[bin].r = input[bin][0];
        frequency[bin].i = input[bin][1];
    }
    kiss_fftri(inverse, frequency, time_frame);
    for (size_t n = 0; n < GTCRN_FFT_SIZE; ++n)
        output[n] = time_frame[n] / GTCRN_FFT_SIZE;
    free(inverse);
    return GTCRN_MODEL_OK;
}

int gtcrn_enhance_audio_f32(gtcrn_model_t *model,
                            const float *input,
                            size_t sample_count,
                            float *output) {
    if (model == NULL || input == NULL || output == NULL || sample_count < 2)
        return GTCRN_MODEL_BAD_ARGUMENT;

    const size_t padded_count = sample_count + GTCRN_FFT_SIZE;
    const size_t frame_count = sample_count / GTCRN_HOP_SIZE + 1;
    float *accumulator = (float *)calloc(padded_count, sizeof(float));
    float *normalizer = (float *)calloc(padded_count, sizeof(float));
    kiss_fftr_cfg forward = kiss_fftr_alloc(GTCRN_FFT_SIZE, 0, NULL, NULL);
    kiss_fftr_cfg inverse = kiss_fftr_alloc(GTCRN_FFT_SIZE, 1, NULL, NULL);
    if (accumulator == NULL || normalizer == NULL || forward == NULL || inverse == NULL) {
        free(accumulator);
        free(normalizer);
        free(forward);
        free(inverse);
        return GTCRN_MODEL_BAD_ARGUMENT;
    }

    float window[GTCRN_FFT_SIZE];
    for (size_t n = 0; n < GTCRN_FFT_SIZE; ++n) {
        const float hann = 0.5f - 0.5f * cosf((float)(2.0 * M_PI * n / GTCRN_FFT_SIZE));
        window[n] = sqrtf(hann);
    }

    kiss_fft_scalar time_frame[GTCRN_FFT_SIZE];
    kiss_fft_cpx frequency[GTCRN_BINS];
    float model_input[GTCRN_BINS][2];
    float model_output[GTCRN_BINS][2];
    for (size_t frame = 0; frame < frame_count; ++frame) {
        const size_t offset = frame * GTCRN_HOP_SIZE;
        for (size_t n = 0; n < GTCRN_FFT_SIZE; ++n) {
            const long source = (long)(offset + n) - GTCRN_FFT_SIZE / 2;
            time_frame[n] = input[reflect_index(source, sample_count)] * window[n];
        }
        kiss_fftr(forward, time_frame, frequency);
        for (size_t bin = 0; bin < GTCRN_BINS; ++bin) {
            model_input[bin][0] = frequency[bin].r;
            model_input[bin][1] = frequency[bin].i;
        }
        const int status = gtcrn_model_process_frame(model, model_input, model_output);
        if (status != GTCRN_MODEL_OK) {
            free(accumulator); free(normalizer); free(forward); free(inverse);
            return status;
        }
        for (size_t bin = 0; bin < GTCRN_BINS; ++bin) {
            frequency[bin].r = model_output[bin][0];
            frequency[bin].i = model_output[bin][1];
        }
        kiss_fftri(inverse, frequency, time_frame);
        for (size_t n = 0; n < GTCRN_FFT_SIZE; ++n) {
            const size_t index = offset + n;
            if (index >= padded_count) break;
            const float synthesis = time_frame[n] / GTCRN_FFT_SIZE * window[n];
            accumulator[index] += synthesis;
            normalizer[index] += window[n] * window[n];
        }
    }

    for (size_t n = 0; n < sample_count; ++n) {
        const size_t index = n + GTCRN_FFT_SIZE / 2;
        output[n] = normalizer[index] > 1.0e-11f
            ? accumulator[index] / normalizer[index] : 0.0f;
    }
    free(accumulator);
    free(normalizer);
    free(forward);
    free(inverse);
    return GTCRN_MODEL_OK;
}
