#include "gtcrn_kernels.h"

#include <math.h>

int8_t gtcrn_saturate_i8(int32_t value) {
    if (value > 127) return 127;
    if (value < -128) return -128;
    return (int8_t)value;
}

int16_t gtcrn_saturate_i16(int32_t value) {
    if (value > 32767) return 32767;
    if (value < -32768) return -32768;
    return (int16_t)value;
}

int32_t gtcrn_requantize(int32_t value, const gtcrn_requant_t *params) {
    int64_t scaled = (int64_t)value * params->multiplier;
    scaled = (scaled + (INT64_C(1) << 30)) >> 31;
    if (params->shift > 0) {
        scaled <<= params->shift;
    } else if (params->shift < 0) {
        const int right = -params->shift;
        scaled = (scaled + (INT64_C(1) << (right - 1))) >> right;
    }
    return (int32_t)scaled + params->output_zero_point;
}

void gtcrn_linear_s8(const int8_t *input,
                     const int8_t *weights,
                     const int32_t *bias,
                     const gtcrn_requant_t *requant,
                     int8_t *output,
                     size_t input_size,
                     size_t output_size,
                     int32_t input_zero_point) {
    for (size_t out = 0; out < output_size; ++out) {
        int32_t accumulator = bias ? bias[out] : 0;
        const int8_t *row = weights + out * input_size;
        for (size_t in = 0; in < input_size; ++in) {
            accumulator += ((int32_t)input[in] - input_zero_point) * row[in];
        }
        output[out] = gtcrn_saturate_i8(gtcrn_requantize(accumulator, &requant[out]));
    }
}

void gtcrn_linear_f32(const float *input, const float *weights,
                      const float *bias, float *output,
                      size_t input_size, size_t output_size) {
    for (size_t out = 0; out < output_size; ++out) {
        float sum = bias ? bias[out] : 0.0f;
        const float *row = weights + out * input_size;
        for (size_t in = 0; in < input_size; ++in) sum += row[in] * input[in];
        output[out] = sum;
    }
}

static size_t output_extent(size_t input, size_t kernel, size_t stride,
                            size_t dilation, size_t pad) {
    return (input + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
}

void gtcrn_conv2d_f32(const float *input, const float *weights,
                      const float *bias, float *output,
                      size_t input_channels, size_t input_height,
                      size_t input_width, size_t output_channels,
                      size_t kernel_height, size_t kernel_width,
                      size_t stride_height, size_t stride_width,
                      size_t dilation_height, size_t dilation_width,
                      size_t pad_height, size_t pad_width, size_t groups) {
    const size_t output_height = output_extent(input_height, kernel_height,
                                                stride_height, dilation_height, pad_height);
    const size_t output_width = output_extent(input_width, kernel_width,
                                               stride_width, dilation_width, pad_width);
    const size_t input_group = input_channels / groups;
    const size_t output_group = output_channels / groups;
    for (size_t oc = 0; oc < output_channels; ++oc) {
        const size_t group = oc / output_group;
        for (size_t oh = 0; oh < output_height; ++oh) {
            for (size_t ow = 0; ow < output_width; ++ow) {
                float sum = bias ? bias[oc] : 0.0f;
                for (size_t icg = 0; icg < input_group; ++icg) {
                    const size_t ic = group * input_group + icg;
                    for (size_t kh = 0; kh < kernel_height; ++kh) {
                        const int ih = (int)(oh * stride_height + kh * dilation_height) - (int)pad_height;
                        if (ih < 0 || ih >= (int)input_height) continue;
                        for (size_t kw = 0; kw < kernel_width; ++kw) {
                            const int iw = (int)(ow * stride_width + kw * dilation_width) - (int)pad_width;
                            if (iw < 0 || iw >= (int)input_width) continue;
                            const size_t input_index = (ic * input_height + (size_t)ih) * input_width + (size_t)iw;
                            const size_t weight_index = (((oc * input_group + icg) * kernel_height + kh) * kernel_width) + kw;
                            sum += input[input_index] * weights[weight_index];
                        }
                    }
                }
                output[(oc * output_height + oh) * output_width + ow] = sum;
            }
        }
    }
}

void gtcrn_conv2d_s8(const int8_t *input, const int8_t *weights,
                     const int32_t *bias, const gtcrn_requant_t *requant,
                     int8_t *output, size_t input_channels,
                     size_t input_height, size_t input_width,
                     size_t output_channels, size_t kernel_height,
                     size_t kernel_width, size_t stride_height,
                     size_t stride_width, size_t dilation_height,
                     size_t dilation_width, size_t pad_height,
                     size_t pad_width, size_t groups,
                     int32_t input_zero_point) {
    const size_t output_height = output_extent(input_height, kernel_height,
                                                stride_height, dilation_height, pad_height);
    const size_t output_width = output_extent(input_width, kernel_width,
                                               stride_width, dilation_width, pad_width);
    const size_t input_group = input_channels / groups;
    const size_t output_group = output_channels / groups;
    for (size_t oc = 0; oc < output_channels; ++oc) {
        const size_t group = oc / output_group;
        for (size_t oh = 0; oh < output_height; ++oh) {
            for (size_t ow = 0; ow < output_width; ++ow) {
                int32_t sum = bias ? bias[oc] : 0;
                for (size_t icg = 0; icg < input_group; ++icg) {
                    const size_t ic = group * input_group + icg;
                    for (size_t kh = 0; kh < kernel_height; ++kh) {
                        const int ih = (int)(oh * stride_height + kh * dilation_height) - (int)pad_height;
                        if (ih < 0 || ih >= (int)input_height) continue;
                        for (size_t kw = 0; kw < kernel_width; ++kw) {
                            const int iw = (int)(ow * stride_width + kw * dilation_width) - (int)pad_width;
                            if (iw < 0 || iw >= (int)input_width) continue;
                            const size_t input_index = (ic * input_height + (size_t)ih) * input_width + (size_t)iw;
                            const size_t weight_index = (((oc * input_group + icg) * kernel_height + kh) * kernel_width) + kw;
                            sum += ((int32_t)input[input_index] - input_zero_point) * weights[weight_index];
                        }
                    }
                }
                const size_t output_index = (oc * output_height + oh) * output_width + ow;
                output[output_index] = gtcrn_saturate_i8(gtcrn_requantize(sum, &requant[oc]));
            }
        }
    }
}

void gtcrn_prelu_f32(float *data, size_t count, float alpha) {
    for (size_t i = 0; i < count; ++i) {
        if (data[i] < 0.0f) data[i] *= alpha;
    }
}

void gtcrn_layer_norm_f32(float *data, const float *gamma,
                          const float *beta, size_t count, float epsilon) {
    float mean = 0.0f;
    for (size_t i = 0; i < count; ++i) mean += data[i];
    mean /= (float)count;
    float variance = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        const float delta = data[i] - mean;
        variance += delta * delta;
    }
    variance /= (float)count;
    const float inverse_std = 1.0f / sqrtf(variance + epsilon);
    for (size_t i = 0; i < count; ++i) {
        data[i] = (data[i] - mean) * inverse_std * gamma[i] + beta[i];
    }
}

static float sigmoidf_stable(float value) {
    if (value >= 0.0f) {
        const float e = expf(-value);
        return 1.0f / (1.0f + e);
    }
    const float e = expf(value);
    return e / (1.0f + e);
}

void gtcrn_gru_f32(const float *input, float *hidden,
                   const float *weight_ih, const float *weight_hh,
                   const float *bias_ih, const float *bias_hh,
                   size_t input_size, size_t hidden_size) {
    /* GTCRN's largest hidden size is 16. Avoid heap allocation on MCU. */
    float reset[16], update[16], candidate[16];
    if (hidden_size > 16) return;
    for (size_t gate = 0; gate < 3; ++gate) {
        for (size_t out = 0; out < hidden_size; ++out) {
            const size_t row_index = gate * hidden_size + out;
            float x_sum = bias_ih[row_index];
            float h_sum = bias_hh[row_index];
            for (size_t in = 0; in < input_size; ++in)
                x_sum += weight_ih[row_index * input_size + in] * input[in];
            for (size_t in = 0; in < hidden_size; ++in)
                h_sum += weight_hh[row_index * hidden_size + in] * hidden[in];
            if (gate == 0) reset[out] = sigmoidf_stable(x_sum + h_sum);
            else if (gate == 1) update[out] = sigmoidf_stable(x_sum + h_sum);
            else candidate[out] = x_sum + reset[out] * h_sum;
        }
    }
    for (size_t i = 0; i < hidden_size; ++i) {
        candidate[i] = tanhf(candidate[i]);
        hidden[i] = (1.0f - update[i]) * candidate[i] + update[i] * hidden[i];
    }
}
