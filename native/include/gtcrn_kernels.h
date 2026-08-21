#ifndef GTCRN_KERNELS_H
#define GTCRN_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t multiplier;
    int8_t shift;
    int32_t output_zero_point;
} gtcrn_requant_t;

int8_t gtcrn_saturate_i8(int32_t value);
int16_t gtcrn_saturate_i16(int32_t value);
int32_t gtcrn_requantize(int32_t value, const gtcrn_requant_t *params);

void gtcrn_linear_s8(const int8_t *input,
                     const int8_t *weights,
                     const int32_t *bias,
                     const gtcrn_requant_t *requant,
                     int8_t *output,
                     size_t input_size,
                     size_t output_size,
                     int32_t input_zero_point);

void gtcrn_linear_f32(const float *input, const float *weights,
                      const float *bias, float *output,
                      size_t input_size, size_t output_size);

void gtcrn_conv2d_f32(const float *input, const float *weights,
                      const float *bias, float *output,
                      size_t input_channels, size_t input_height,
                      size_t input_width, size_t output_channels,
                      size_t kernel_height, size_t kernel_width,
                      size_t stride_height, size_t stride_width,
                      size_t dilation_height, size_t dilation_width,
                      size_t pad_height, size_t pad_width, size_t groups);

void gtcrn_conv2d_s8(const int8_t *input, const int8_t *weights,
                     const int32_t *bias, const gtcrn_requant_t *requant,
                     int8_t *output, size_t input_channels,
                     size_t input_height, size_t input_width,
                     size_t output_channels, size_t kernel_height,
                     size_t kernel_width, size_t stride_height,
                     size_t stride_width, size_t dilation_height,
                     size_t dilation_width, size_t pad_height,
                     size_t pad_width, size_t groups,
                     int32_t input_zero_point);
void gtcrn_prelu_f32(float *data, size_t count, float alpha);
void gtcrn_layer_norm_f32(float *data, const float *gamma,
                          const float *beta, size_t count, float epsilon);

/* PyTorch-compatible single-direction GRU. Gate order is reset/update/new. */
void gtcrn_gru_f32(const float *input, float *hidden,
                   const float *weight_ih, const float *weight_hh,
                   const float *bias_ih, const float *bias_hh,
                   size_t input_size, size_t hidden_size);

#ifdef __cplusplus
}
#endif

#endif
