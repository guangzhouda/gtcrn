#include "gtcrn_model.h"

#include "gtcrn_kernels.h"
#include "weights_index.h"

#include <math.h>
#include <string.h>

#ifdef GTCRN_Q15_STATE
typedef int16_t state_value_t;
static float state_read(state_value_t value, float scale) { return value * scale; }
static state_value_t state_write(float value, float scale) {
    long q = lroundf(value / scale);
    if (q > 32767) q = 32767;
    if (q < -32768) q = -32768;
    return (state_value_t)q;
}
#else
typedef float state_value_t;
static float state_read(state_value_t value, float scale) { (void)scale; return value; }
static state_value_t state_write(float value, float scale) { (void)scale; return value; }
#endif

#define WFLOAT(model, name) \
    ((const float *)((model)->weights_blob + GTCRN_TENSOR_##name##_OFFSET))

typedef struct {
    const void *data;
    const float *scale;
} weight_ref_t;

#ifdef GTCRN_INT8_WEIGHTS
#define WREF(model, name) \
    { (const void *)((model)->weights_blob + GTCRN_TENSOR_##name##_OFFSET), \
      (const float *)((model)->weights_blob + GTCRN_TENSOR_##name##_SCALE_OFFSET) }
#else
#define WREF(model, name) \
    { (const void *)((model)->weights_blob + GTCRN_TENSOR_##name##_OFFSET), NULL }
#endif

#define COPY_CHW(dst, src, channels, width) \
    do { \
        for (size_t copy_c = 0; copy_c < (channels); ++copy_c) \
            memcpy((dst)[copy_c], (src)[copy_c], (width) * sizeof(float)); \
    } while (0)

typedef struct {
    weight_ref_t weight;
    const float *bias;
    size_t in_channels;
    size_t out_channels;
    size_t kernel_width;
    size_t stride_width;
    size_t pad_width;
    size_t groups;
    int is_deconv_freq;
    int use_tanh;
    const float *prelu;
} conv_block_t;

typedef struct {
    weight_ref_t point1_weight;
    const float *point1_bias;
    weight_ref_t depth_weight;
    const float *depth_bias;
    weight_ref_t point2_weight;
    const float *point2_bias;
    const float *point_prelu;
    const float *depth_prelu;
    weight_ref_t tra_wih;
    weight_ref_t tra_whh;
    const float *tra_bih;
    const float *tra_bhh;
    weight_ref_t tra_fc_weight;
    const float *tra_fc_bias;
    size_t dilation_t;
    int is_decoder;
} gt_block_t;

typedef struct {
    weight_ref_t rnn1_wih;
    weight_ref_t rnn1_whh;
    const float *rnn1_bih;
    const float *rnn1_bhh;
    weight_ref_t rnn2_wih;
    weight_ref_t rnn2_whh;
    const float *rnn2_bih;
    const float *rnn2_bhh;
    weight_ref_t rnn1_wih_r;
    weight_ref_t rnn1_whh_r;
    const float *rnn1_bih_r;
    const float *rnn1_bhh_r;
    weight_ref_t rnn2_wih_r;
    weight_ref_t rnn2_whh_r;
    const float *rnn2_bih_r;
    const float *rnn2_bhh_r;
    weight_ref_t fc_weight;
    const float *fc_bias;
    const float *ln_weight;
    const float *ln_bias;
} intra_block_t;

typedef struct {
    weight_ref_t rnn1_wih;
    weight_ref_t rnn1_whh;
    const float *rnn1_bih;
    const float *rnn1_bhh;
    weight_ref_t rnn2_wih;
    weight_ref_t rnn2_whh;
    const float *rnn2_bih;
    const float *rnn2_bhh;
    weight_ref_t fc_weight;
    const float *fc_bias;
    const float *ln_weight;
    const float *ln_bias;
} inter_block_t;

static float sigmoidf_local(float value) {
    if (value >= 0.0f) {
        const float e = expf(-value);
        return 1.0f / (1.0f + e);
    }
    const float e = expf(value);
    return e / (1.0f + e);
}

static void matvec(weight_ref_t weight, const float *input, const float *bias,
                   float *output, size_t input_size, size_t output_size) {
#ifdef GTCRN_INT8_WEIGHTS
    int8_t quantized[128];
    float max_abs = 0.0f;
    for (size_t i = 0; i < input_size; ++i) {
        const float value = fabsf(input[i]);
        if (value > max_abs) max_abs = value;
    }
    const float input_scale = max_abs > 1.0e-12f ? max_abs / 127.0f : 1.0f;
    for (size_t i = 0; i < input_size; ++i) {
        long value = lroundf(input[i] / input_scale);
        if (value > 127) value = 127;
        if (value < -127) value = -127;
        quantized[i] = (int8_t)value;
    }
    const int8_t *weights = (const int8_t *)weight.data;
    for (size_t out = 0; out < output_size; ++out) {
        int32_t accumulator = 0;
        const int8_t *row = weights + out * input_size;
        for (size_t i = 0; i < input_size; ++i)
            accumulator += (int32_t)quantized[i] * row[i];
        output[out] = (bias ? bias[out] : 0.0f)
            + accumulator * input_scale * weight.scale[out];
    }
#else
    gtcrn_linear_f32(input, (const float *)weight.data, bias, output,
                     input_size, output_size);
#endif
}

static float dot_row(weight_ref_t weight, size_t row_index,
                     const float *input, size_t input_size) {
    float output;
    weight_ref_t row = weight;
#ifdef GTCRN_INT8_WEIGHTS
    row.data = (const int8_t *)weight.data + row_index * input_size;
    row.scale = weight.scale + row_index;
#else
    row.data = (const float *)weight.data + row_index * input_size;
#endif
    matvec(row, input, NULL, &output, input_size, 1);
    return output;
}

static void prelu(float data[][GTCRN_ERB_BINS], size_t channels,
                  size_t width, const float *alpha) {
    const float a = alpha ? alpha[0] : 0.0f;
    for (size_t c = 0; c < channels; ++c)
        for (size_t f = 0; f < width; ++f)
            if (data[c][f] < 0.0f) data[c][f] *= a;
}

static void tanh_act(float data[][GTCRN_ERB_BINS], size_t channels, size_t width) {
    for (size_t c = 0; c < channels; ++c)
        for (size_t f = 0; f < width; ++f)
            data[c][f] = tanhf(data[c][f]);
}

static void conv1xk(const float input[][GTCRN_ERB_BINS], float output[][GTCRN_ERB_BINS],
                    size_t in_channels, size_t input_width,
                    const conv_block_t *cfg) {
    memset(output, 0, 24 * GTCRN_ERB_BINS * sizeof(float));
    const size_t out_width = cfg->is_deconv_freq
        ? input_width * cfg->stride_width - 1
        : (input_width + 2 * cfg->pad_width - cfg->kernel_width) / cfg->stride_width + 1;
    const size_t in_group = in_channels / cfg->groups;
    const size_t out_group = cfg->out_channels / cfg->groups;
    for (size_t group = 0; group < cfg->groups; ++group) {
        for (size_t ow = 0; ow < out_width; ++ow) {
            float patch[128];
            size_t patch_index = 0;
            for (size_t icg = 0; icg < in_group; ++icg) {
                const size_t ic = group * in_group + icg;
                for (size_t kw = 0; kw < cfg->kernel_width; ++kw) {
                    int iw;
                    if (cfg->is_deconv_freq) {
                        const int padded = (int)ow + (int)kw - (int)(cfg->kernel_width - 3);
                        iw = ((padded & 1) != 0) ? -1000000 : padded / 2;
                    } else {
                        iw = (int)(ow * cfg->stride_width + kw) - (int)cfg->pad_width;
                    }
                    patch[patch_index++] = (iw < 0 || iw >= (int)input_width)
                        ? 0.0f : input[ic][iw];
                }
            }
            for (size_t ocg = 0; ocg < out_group; ++ocg) {
                const size_t oc = group * out_group + ocg;
                output[oc][ow] = (cfg->bias ? cfg->bias[oc] : 0.0f)
                    + dot_row(cfg->weight, oc, patch, patch_index);
            }
        }
    }
    if (cfg->use_tanh) tanh_act(output, cfg->out_channels, out_width);
    else prelu(output, cfg->out_channels, out_width, cfg->prelu);
}

static void sfe3(const float input[][GTCRN_ERB_BINS], float output[][GTCRN_ERB_BINS],
                 size_t channels, size_t width) {
    for (size_t c = 0; c < channels; ++c) {
        for (size_t f = 0; f < width; ++f) {
            output[c * 3 + 0][f] = (f == 0) ? 0.0f : input[c][f - 1];
            output[c * 3 + 1][f] = input[c][f];
            output[c * 3 + 2][f] = (f + 1 == width) ? 0.0f : input[c][f + 1];
        }
    }
}

static void pointwise_bins(const float input[][GTCRN_ERB_BINS],
                           float output[][GTCRN_ERB_BINS],
                           size_t in_channels, size_t out_channels,
                           size_t width, weight_ref_t weight,
                           const float *bias) {
    for (size_t f = 0; f < width; ++f) {
        float vector[32], result[32];
        for (size_t ic = 0; ic < in_channels; ++ic) vector[ic] = input[ic][f];
        matvec(weight, vector, bias, result, in_channels, out_channels);
        for (size_t oc = 0; oc < out_channels; ++oc) output[oc][f] = result[oc];
    }
}

static void pointwise_core(const float input[][GTCRN_ERB_BINS],
                           float output[8][GTCRN_CORE_BINS],
                           weight_ref_t weight, const float *bias) {
    for (size_t f = 0; f < GTCRN_CORE_BINS; ++f) {
        float vector[GTCRN_CHANNELS], result[8];
        for (size_t ic = 0; ic < GTCRN_CHANNELS; ++ic) vector[ic] = input[ic][f];
        matvec(weight, vector, bias, result, GTCRN_CHANNELS, 8);
        for (size_t oc = 0; oc < 8; ++oc) output[oc][f] = result[oc];
    }
}

static void erb_bm(gtcrn_model_t *model, const float in[3][GTCRN_BINS],
                   float out[][GTCRN_ERB_BINS]) {
    (void)model;
    float band[3][GTCRN_ERB_BINS];
    for (size_t c = 0; c < 3; ++c) {
        for (size_t f = 0; f < 65; ++f) band[c][f] = in[c][f];
        for (size_t o = 0; o < 64; ++o) {
            float sum = 0.0f;
            for (uint16_t entry = gtcrn_erb_row_ptr[o];
                 entry < gtcrn_erb_row_ptr[o + 1]; ++entry)
                sum += gtcrn_erb_values[entry]
                     * in[c][65 + gtcrn_erb_col_index[entry]];
            band[c][65 + o] = sum;
        }
    }
    sfe3(band, out, 3, GTCRN_ERB_BINS);
}

static void erb_bs(gtcrn_model_t *model, const float in[][GTCRN_ERB_BINS],
                   float out[2][GTCRN_BINS]) {
    (void)model;
    for (size_t c = 0; c < 2; ++c) {
        for (size_t f = 0; f < 65; ++f) out[c][f] = in[c][f];
        memset(&out[c][65], 0, 192 * sizeof(float));
        for (size_t row = 0; row < 64; ++row)
            for (uint16_t entry = gtcrn_erb_row_ptr[row];
                 entry < gtcrn_erb_row_ptr[row + 1]; ++entry)
                out[c][65 + gtcrn_erb_col_index[entry]] +=
                    gtcrn_erb_values[entry] * in[c][65 + row];
    }
}

static void stream_depth_conv(const float input[][GTCRN_ERB_BINS],
                              float output[][GTCRN_ERB_BINS],
                              state_value_t cache[GTCRN_CHANNELS][10][GTCRN_CORE_BINS],
                              const gt_block_t *cfg) {
    const size_t history = 2 * cfg->dilation_t;
    for (size_t c = 0; c < GTCRN_CHANNELS; ++c) {
        for (size_t f = 0; f < GTCRN_CORE_BINS; ++f) {
            float patch[9];
            size_t patch_index = 0;
            for (size_t kt = 0; kt < 3; ++kt) {
                const size_t t = kt * cfg->dilation_t;
                for (size_t kw = 0; kw < 3; ++kw) {
                    const int fw = (int)f + (int)kw - 1;
                    patch[patch_index++] = (fw < 0 || fw >= GTCRN_CORE_BINS)
                        ? 0.0f : ((t == history) ? input[c][fw]
                                                : state_read(cache[c][t][fw], 1.0f / 256.0f));
                }
            }
            output[c][f] = cfg->depth_bias[c]
                + dot_row(cfg->depth_weight, c, patch, patch_index);
        }
        for (size_t t = 0; t + 1 < history; ++t)
            memcpy(cache[c][t], cache[c][t + 1], GTCRN_CORE_BINS * sizeof(state_value_t));
        for (size_t f = 0; f < GTCRN_CORE_BINS; ++f)
            cache[c][history - 1][f] = state_write(input[c][f], 1.0f / 256.0f);
    }
}

static void gru_one(const float *input, float *hidden,
                    weight_ref_t wih, weight_ref_t whh,
                    const float *bih, const float *bhh,
                    size_t input_size, size_t hidden_size, float *output);

static void tra(float x[8][GTCRN_CORE_BINS], state_value_t hidden[GTCRN_CHANNELS],
                const gt_block_t *cfg) {
    float z[8], fc[8], hidden_f[GTCRN_CHANNELS];
    for (size_t i = 0; i < GTCRN_CHANNELS; ++i)
        hidden_f[i] = state_read(hidden[i], 1.0f / 32767.0f);
    for (size_t c = 0; c < 8; ++c) {
        float sum = 0.0f;
        for (size_t f = 0; f < GTCRN_CORE_BINS; ++f) sum += x[c][f] * x[c][f];
        z[c] = sum / (float)GTCRN_CORE_BINS;
    }
    gru_one(z, hidden_f, cfg->tra_wih, cfg->tra_whh, cfg->tra_bih,
            cfg->tra_bhh, 8, GTCRN_CHANNELS, hidden_f);
    for (size_t i = 0; i < GTCRN_CHANNELS; ++i)
        hidden[i] = state_write(hidden_f[i], 1.0f / 32767.0f);
    matvec(cfg->tra_fc_weight, hidden_f, cfg->tra_fc_bias, fc, 16, 8);
    for (size_t c = 0; c < 8; ++c) {
        const float a = sigmoidf_local(fc[c]);
        for (size_t f = 0; f < GTCRN_CORE_BINS; ++f) x[c][f] *= a;
    }
}

static void gt_block(gtcrn_model_t *model,
                     const float input[][GTCRN_ERB_BINS],
                     float output[][GTCRN_ERB_BINS],
                     state_value_t cache[GTCRN_CHANNELS][10][GTCRN_CORE_BINS],
                     state_value_t tra_hidden[GTCRN_CHANNELS],
                     const gt_block_t *cfg) {
    float input_copy[GTCRN_CHANNELS][GTCRN_ERB_BINS];
    float (*sfe)[GTCRN_ERB_BINS] = model->scratch_a;
    float (*h1)[GTCRN_ERB_BINS] = model->scratch_b;
    float (*depth)[GTCRN_ERB_BINS] = model->scratch_a;
    float half[8][GTCRN_CORE_BINS];

    for (size_t c = 0; c < GTCRN_CHANNELS; ++c)
        memcpy(input_copy[c], input[c], GTCRN_CORE_BINS * sizeof(float));
    sfe3(input_copy, sfe, 8, GTCRN_CORE_BINS);
    pointwise_bins(sfe, h1, 24, 16,
                   GTCRN_CORE_BINS, cfg->point1_weight, cfg->point1_bias);
    prelu(h1, 16, GTCRN_CORE_BINS, cfg->point_prelu);
    stream_depth_conv(h1, depth, cache, cfg);
    prelu(depth, 16, GTCRN_CORE_BINS, cfg->depth_prelu);
    pointwise_core(depth, half,
                   cfg->point2_weight, cfg->point2_bias);
    tra(half, tra_hidden, cfg);
    for (size_t c = 0; c < 8; ++c) {
        memcpy(output[2 * c], half[c], GTCRN_CORE_BINS * sizeof(float));
        memcpy(output[2 * c + 1], input_copy[8 + c], GTCRN_CORE_BINS * sizeof(float));
    }
}

static void gru_one(const float *input, float *hidden,
                    weight_ref_t wih, weight_ref_t whh,
                    const float *bih, const float *bhh,
                    size_t input_size, size_t hidden_size, float *output) {
    float x_gate[48], h_gate[48], reset[16], update[16], candidate[16];
    matvec(wih, input, bih, x_gate, input_size, hidden_size * 3);
    matvec(whh, hidden, bhh, h_gate, hidden_size, hidden_size * 3);
    for (size_t i = 0; i < hidden_size; ++i) {
        reset[i] = sigmoidf_local(x_gate[i] + h_gate[i]);
        update[i] = sigmoidf_local(x_gate[hidden_size + i] + h_gate[hidden_size + i]);
        candidate[i] = tanhf(x_gate[2 * hidden_size + i]
                             + reset[i] * h_gate[2 * hidden_size + i]);
    }
    for (size_t i = 0; i < hidden_size; ++i) {
        hidden[i] = (1.0f - update[i]) * candidate[i] + update[i] * hidden[i];
        output[i] = hidden[i];
    }
}

static void layer_norm_2d(float x[GTCRN_CORE_BINS][GTCRN_CHANNELS],
                          const float *gamma, const float *beta) {
    float mean = 0.0f;
    const size_t count = GTCRN_CORE_BINS * GTCRN_CHANNELS;
    for (size_t f = 0; f < GTCRN_CORE_BINS; ++f)
        for (size_t c = 0; c < GTCRN_CHANNELS; ++c) mean += x[f][c];
    mean /= (float)count;
    float var = 0.0f;
    for (size_t f = 0; f < GTCRN_CORE_BINS; ++f) {
        for (size_t c = 0; c < GTCRN_CHANNELS; ++c) {
            const float d = x[f][c] - mean;
            var += d * d;
        }
    }
    var /= (float)count;
    const float inv = 1.0f / sqrtf(var + 1.0e-8f);
    for (size_t f = 0; f < GTCRN_CORE_BINS; ++f)
        for (size_t c = 0; c < GTCRN_CHANNELS; ++c)
            x[f][c] = (x[f][c] - mean) * inv * gamma[f * 16 + c] + beta[f * 16 + c];
}

static void dpgrnn(gtcrn_model_t *model,
                   const float input[][GTCRN_ERB_BINS],
                   float output[][GTCRN_ERB_BINS],
                   state_value_t inter_cache[GTCRN_CORE_BINS][GTCRN_CHANNELS],
                   const intra_block_t *intra,
                   const inter_block_t *inter) {
    float (*x)[GTCRN_CHANNELS] = model->scratch_c;
    float (*tmp)[GTCRN_CHANNELS] = model->scratch_d;
    for (size_t f = 0; f < GTCRN_CORE_BINS; ++f)
        for (size_t c = 0; c < GTCRN_CHANNELS; ++c) x[f][c] = input[c][f];

    float h1[4] = {0}, h2[4] = {0}, h1r[4] = {0}, h2r[4] = {0};
    float y_fwd[GTCRN_CORE_BINS][16], y_rev[GTCRN_CORE_BINS][16];
    for (size_t f = 0; f < GTCRN_CORE_BINS; ++f) {
        gru_one(&x[f][0], h1, intra->rnn1_wih, intra->rnn1_whh,
                intra->rnn1_bih, intra->rnn1_bhh, 8, 4, &y_fwd[f][0]);
        gru_one(&x[f][8], h2, intra->rnn2_wih, intra->rnn2_whh,
                intra->rnn2_bih, intra->rnn2_bhh, 8, 4, &y_fwd[f][8]);
    }
    for (size_t ri = 0; ri < GTCRN_CORE_BINS; ++ri) {
        const size_t f = GTCRN_CORE_BINS - 1 - ri;
        gru_one(&x[f][0], h1r, intra->rnn1_wih_r, intra->rnn1_whh_r,
                intra->rnn1_bih_r, intra->rnn1_bhh_r, 8, 4, &y_rev[f][4]);
        gru_one(&x[f][8], h2r, intra->rnn2_wih_r, intra->rnn2_whh_r,
                intra->rnn2_bih_r, intra->rnn2_bhh_r, 8, 4, &y_rev[f][12]);
    }
    for (size_t f = 0; f < GTCRN_CORE_BINS; ++f) {
        float lin_in[16];
        memcpy(&lin_in[0], &y_fwd[f][0], 4 * sizeof(float));
        memcpy(&lin_in[4], &y_rev[f][4], 4 * sizeof(float));
        memcpy(&lin_in[8], &y_fwd[f][8], 4 * sizeof(float));
        memcpy(&lin_in[12], &y_rev[f][12], 4 * sizeof(float));
        matvec(intra->fc_weight, lin_in, intra->fc_bias, tmp[f], 16, 16);
    }
    layer_norm_2d(tmp, intra->ln_weight, intra->ln_bias);
    for (size_t f = 0; f < GTCRN_CORE_BINS; ++f)
        for (size_t c = 0; c < GTCRN_CHANNELS; ++c) x[f][c] += tmp[f][c];

    for (size_t f = 0; f < GTCRN_CORE_BINS; ++f) {
        float y[16], hidden1[8], hidden2[8];
        for (size_t c = 0; c < 8; ++c) {
            hidden1[c] = state_read(inter_cache[f][c], 1.0f / 32767.0f);
            hidden2[c] = state_read(inter_cache[f][8 + c], 1.0f / 32767.0f);
        }
        gru_one(&x[f][0], hidden1, inter->rnn1_wih, inter->rnn1_whh,
                inter->rnn1_bih, inter->rnn1_bhh, 8, 8, &y[0]);
        gru_one(&x[f][8], hidden2, inter->rnn2_wih, inter->rnn2_whh,
                inter->rnn2_bih, inter->rnn2_bhh, 8, 8, &y[8]);
        for (size_t c = 0; c < 8; ++c) {
            inter_cache[f][c] = state_write(hidden1[c], 1.0f / 32767.0f);
            inter_cache[f][8 + c] = state_write(hidden2[c], 1.0f / 32767.0f);
        }
        matvec(inter->fc_weight, y, inter->fc_bias, tmp[f], 16, 16);
    }
    layer_norm_2d(tmp, inter->ln_weight, inter->ln_bias);
    for (size_t f = 0; f < GTCRN_CORE_BINS; ++f)
        for (size_t c = 0; c < GTCRN_CHANNELS; ++c)
            output[c][f] = x[f][c] + tmp[f][c];
}

static conv_block_t enc_conv0(gtcrn_model_t *m) {
    conv_block_t c = {WREF(m, FUSED_ENCODER_EN_CONVS_0_CONV_WEIGHT),
                      WFLOAT(m, FUSED_ENCODER_EN_CONVS_0_CONV_BIAS), 9, 16,
                      5, 2, 2, 1, 0, 0,
                      WFLOAT(m, RAW_ENCODER_EN_CONVS_0_ACT_WEIGHT)};
    return c;
}

static conv_block_t enc_conv1(gtcrn_model_t *m) {
    conv_block_t c = {WREF(m, FUSED_ENCODER_EN_CONVS_1_CONV_WEIGHT),
                      WFLOAT(m, FUSED_ENCODER_EN_CONVS_1_CONV_BIAS), 16, 16,
                      5, 2, 2, 2, 0, 0,
                      WFLOAT(m, RAW_ENCODER_EN_CONVS_1_ACT_WEIGHT)};
    return c;
}

#define GT_BLOCK_ENCODER(m, i, dil) \
    { WREF(m, FUSED_ENCODER_EN_CONVS_##i##_POINT_CONV1_WEIGHT), \
      WFLOAT(m, FUSED_ENCODER_EN_CONVS_##i##_POINT_CONV1_BIAS), \
      WREF(m, FUSED_ENCODER_EN_CONVS_##i##_DEPTH_CONV_WEIGHT), \
      WFLOAT(m, FUSED_ENCODER_EN_CONVS_##i##_DEPTH_CONV_BIAS), \
      WREF(m, FUSED_ENCODER_EN_CONVS_##i##_POINT_CONV2_WEIGHT), \
      WFLOAT(m, FUSED_ENCODER_EN_CONVS_##i##_POINT_CONV2_BIAS), \
      WFLOAT(m, RAW_ENCODER_EN_CONVS_##i##_POINT_ACT_WEIGHT), \
      WFLOAT(m, RAW_ENCODER_EN_CONVS_##i##_DEPTH_ACT_WEIGHT), \
      WREF(m, RAW_ENCODER_EN_CONVS_##i##_TRA_ATT_GRU_WEIGHT_IH_L0), \
      WREF(m, RAW_ENCODER_EN_CONVS_##i##_TRA_ATT_GRU_WEIGHT_HH_L0), \
      WFLOAT(m, RAW_ENCODER_EN_CONVS_##i##_TRA_ATT_GRU_BIAS_IH_L0), \
      WFLOAT(m, RAW_ENCODER_EN_CONVS_##i##_TRA_ATT_GRU_BIAS_HH_L0), \
      WREF(m, RAW_ENCODER_EN_CONVS_##i##_TRA_ATT_FC_WEIGHT), \
      WFLOAT(m, RAW_ENCODER_EN_CONVS_##i##_TRA_ATT_FC_BIAS), dil, 0 }

#define GT_BLOCK_DECODER(m, i, dil) \
    { WREF(m, FUSED_DECODER_DE_CONVS_##i##_POINT_CONV1_WEIGHT), \
      WFLOAT(m, FUSED_DECODER_DE_CONVS_##i##_POINT_CONV1_BIAS), \
      WREF(m, FUSED_DECODER_DE_CONVS_##i##_DEPTH_CONV_WEIGHT), \
      WFLOAT(m, FUSED_DECODER_DE_CONVS_##i##_DEPTH_CONV_BIAS), \
      WREF(m, FUSED_DECODER_DE_CONVS_##i##_POINT_CONV2_WEIGHT), \
      WFLOAT(m, FUSED_DECODER_DE_CONVS_##i##_POINT_CONV2_BIAS), \
      WFLOAT(m, RAW_DECODER_DE_CONVS_##i##_POINT_ACT_WEIGHT), \
      WFLOAT(m, RAW_DECODER_DE_CONVS_##i##_DEPTH_ACT_WEIGHT), \
      WREF(m, RAW_DECODER_DE_CONVS_##i##_TRA_ATT_GRU_WEIGHT_IH_L0), \
      WREF(m, RAW_DECODER_DE_CONVS_##i##_TRA_ATT_GRU_WEIGHT_HH_L0), \
      WFLOAT(m, RAW_DECODER_DE_CONVS_##i##_TRA_ATT_GRU_BIAS_IH_L0), \
      WFLOAT(m, RAW_DECODER_DE_CONVS_##i##_TRA_ATT_GRU_BIAS_HH_L0), \
      WREF(m, RAW_DECODER_DE_CONVS_##i##_TRA_ATT_FC_WEIGHT), \
      WFLOAT(m, RAW_DECODER_DE_CONVS_##i##_TRA_ATT_FC_BIAS), dil, 1 }

#define INTRA_BLOCK(m, n) \
    { WREF(m, RAW_DPGRNN##n##_INTRA_RNN_RNN1_WEIGHT_IH_L0), \
      WREF(m, RAW_DPGRNN##n##_INTRA_RNN_RNN1_WEIGHT_HH_L0), \
      WFLOAT(m, RAW_DPGRNN##n##_INTRA_RNN_RNN1_BIAS_IH_L0), \
      WFLOAT(m, RAW_DPGRNN##n##_INTRA_RNN_RNN1_BIAS_HH_L0), \
      WREF(m, RAW_DPGRNN##n##_INTRA_RNN_RNN2_WEIGHT_IH_L0), \
      WREF(m, RAW_DPGRNN##n##_INTRA_RNN_RNN2_WEIGHT_HH_L0), \
      WFLOAT(m, RAW_DPGRNN##n##_INTRA_RNN_RNN2_BIAS_IH_L0), \
      WFLOAT(m, RAW_DPGRNN##n##_INTRA_RNN_RNN2_BIAS_HH_L0), \
      WREF(m, RAW_DPGRNN##n##_INTRA_RNN_RNN1_WEIGHT_IH_L0_REVERSE), \
      WREF(m, RAW_DPGRNN##n##_INTRA_RNN_RNN1_WEIGHT_HH_L0_REVERSE), \
      WFLOAT(m, RAW_DPGRNN##n##_INTRA_RNN_RNN1_BIAS_IH_L0_REVERSE), \
      WFLOAT(m, RAW_DPGRNN##n##_INTRA_RNN_RNN1_BIAS_HH_L0_REVERSE), \
      WREF(m, RAW_DPGRNN##n##_INTRA_RNN_RNN2_WEIGHT_IH_L0_REVERSE), \
      WREF(m, RAW_DPGRNN##n##_INTRA_RNN_RNN2_WEIGHT_HH_L0_REVERSE), \
      WFLOAT(m, RAW_DPGRNN##n##_INTRA_RNN_RNN2_BIAS_IH_L0_REVERSE), \
      WFLOAT(m, RAW_DPGRNN##n##_INTRA_RNN_RNN2_BIAS_HH_L0_REVERSE), \
      WREF(m, RAW_DPGRNN##n##_INTRA_FC_WEIGHT), \
      WFLOAT(m, RAW_DPGRNN##n##_INTRA_FC_BIAS), \
      WFLOAT(m, RAW_DPGRNN##n##_INTRA_LN_WEIGHT), \
      WFLOAT(m, RAW_DPGRNN##n##_INTRA_LN_BIAS) }

#define INTER_BLOCK(m, n) \
    { WREF(m, RAW_DPGRNN##n##_INTER_RNN_RNN1_WEIGHT_IH_L0), \
      WREF(m, RAW_DPGRNN##n##_INTER_RNN_RNN1_WEIGHT_HH_L0), \
      WFLOAT(m, RAW_DPGRNN##n##_INTER_RNN_RNN1_BIAS_IH_L0), \
      WFLOAT(m, RAW_DPGRNN##n##_INTER_RNN_RNN1_BIAS_HH_L0), \
      WREF(m, RAW_DPGRNN##n##_INTER_RNN_RNN2_WEIGHT_IH_L0), \
      WREF(m, RAW_DPGRNN##n##_INTER_RNN_RNN2_WEIGHT_HH_L0), \
      WFLOAT(m, RAW_DPGRNN##n##_INTER_RNN_RNN2_BIAS_IH_L0), \
      WFLOAT(m, RAW_DPGRNN##n##_INTER_RNN_RNN2_BIAS_HH_L0), \
      WREF(m, RAW_DPGRNN##n##_INTER_FC_WEIGHT), \
      WFLOAT(m, RAW_DPGRNN##n##_INTER_FC_BIAS), \
      WFLOAT(m, RAW_DPGRNN##n##_INTER_LN_WEIGHT), \
      WFLOAT(m, RAW_DPGRNN##n##_INTER_LN_BIAS) }

int gtcrn_model_init(gtcrn_model_t *model,
                     const uint8_t *weights_blob,
                     size_t weights_blob_size) {
    if (!model || !weights_blob) return GTCRN_MODEL_BAD_ARGUMENT;
    if (weights_blob_size < GTCRN_WEIGHTS_BLOB_SIZE) return GTCRN_MODEL_BAD_WEIGHTS;
    memset(model, 0, sizeof(*model));
    model->weights_blob = weights_blob;
    model->weights_blob_size = weights_blob_size;
    return GTCRN_MODEL_OK;
}

size_t gtcrn_model_sizeof(void) {
    return sizeof(gtcrn_model_t);
}

int gtcrn_model_init_f32(void *model, const uint8_t *weights_blob,
                         size_t weights_blob_size) {
    return gtcrn_model_init((gtcrn_model_t *)model, weights_blob, weights_blob_size);
}

int gtcrn_process_frame_f32(void *model, const float *input_ri,
                            float *output_ri) {
    if (model == NULL || input_ri == NULL || output_ri == NULL)
        return GTCRN_MODEL_BAD_ARGUMENT;
    return gtcrn_model_process_frame((gtcrn_model_t *)model,
                                     (const float (*)[2])input_ri,
                                     (float (*)[2])output_ri);
}

void gtcrn_model_reset(gtcrn_model_t *model) {
    const uint8_t *blob;
    size_t size;
    if (!model) return;
    blob = model->weights_blob;
    size = model->weights_blob_size;
    memset(model, 0, sizeof(*model));
    model->weights_blob = blob;
    model->weights_blob_size = size;
}

int gtcrn_model_process_frame(gtcrn_model_t *model,
                              const float input_spec[GTCRN_BINS][2],
                              float output_spec[GTCRN_BINS][2]) {
    if (!model || !model->weights_blob || !input_spec || !output_spec)
        return GTCRN_MODEL_BAD_ARGUMENT;

    float feat3[3][GTCRN_BINS];
    for (size_t f = 0; f < GTCRN_BINS; ++f) {
        const float real = input_spec[f][0];
        const float imag = input_spec[f][1];
        feat3[0][f] = sqrtf(real * real + imag * imag + 1.0e-12f);
        feat3[1][f] = real;
        feat3[2][f] = imag;
    }

    erb_bm(model, feat3, model->scratch_a);
    conv_block_t c0 = enc_conv0(model);
    conv1xk(model->scratch_a, model->scratch_b,
            9, GTCRN_ERB_BINS, &c0);
    COPY_CHW(model->en_out0, model->scratch_b, 16, 65);

    conv_block_t c1 = enc_conv1(model);
    conv1xk(model->scratch_b, model->scratch_a,
            16, 65, &c1);
    COPY_CHW(model->en_out1, model->scratch_a, 16, GTCRN_CORE_BINS);

    gt_block_t ge2 = GT_BLOCK_ENCODER(model, 2, 1);
    gt_block(model, model->scratch_a, model->scratch_b,
             model->enc_conv_cache[0], model->enc_tra_cache[0], &ge2);
    COPY_CHW(model->en_out2, model->scratch_b, 16, GTCRN_CORE_BINS);

    gt_block_t ge3 = GT_BLOCK_ENCODER(model, 3, 2);
    gt_block(model, model->scratch_b, model->scratch_a,
             model->enc_conv_cache[1], model->enc_tra_cache[1], &ge3);
    COPY_CHW(model->en_out3, model->scratch_a, 16, GTCRN_CORE_BINS);

    gt_block_t ge4 = GT_BLOCK_ENCODER(model, 4, 5);
    gt_block(model, model->scratch_a, model->scratch_b,
             model->enc_conv_cache[2], model->enc_tra_cache[2], &ge4);
    COPY_CHW(model->en_out4, model->scratch_b, 16, GTCRN_CORE_BINS);

    intra_block_t intra1 = INTRA_BLOCK(model, 1);
    inter_block_t inter1 = INTER_BLOCK(model, 1);
    dpgrnn(model, model->scratch_b, model->scratch_a,
           model->inter_cache[0], &intra1, &inter1);

    intra_block_t intra2 = INTRA_BLOCK(model, 2);
    inter_block_t inter2 = INTER_BLOCK(model, 2);
    dpgrnn(model, model->scratch_a, model->scratch_b,
           model->inter_cache[1], &intra2, &inter2);

    for (size_t c = 0; c < 16; ++c)
        for (size_t f = 0; f < GTCRN_CORE_BINS; ++f)
            model->scratch_b[c][f] += model->en_out4[c][f];
    gt_block_t gd0 = GT_BLOCK_DECODER(model, 0, 5);
    gt_block(model, model->scratch_b, model->scratch_a,
             model->dec_conv_cache[0], model->dec_tra_cache[0], &gd0);

    for (size_t c = 0; c < 16; ++c)
        for (size_t f = 0; f < GTCRN_CORE_BINS; ++f)
            model->scratch_a[c][f] += model->en_out3[c][f];
    gt_block_t gd1 = GT_BLOCK_DECODER(model, 1, 2);
    gt_block(model, model->scratch_a, model->scratch_b,
             model->dec_conv_cache[1], model->dec_tra_cache[1], &gd1);

    for (size_t c = 0; c < 16; ++c)
        for (size_t f = 0; f < GTCRN_CORE_BINS; ++f)
            model->scratch_b[c][f] += model->en_out2[c][f];
    gt_block_t gd2 = GT_BLOCK_DECODER(model, 2, 1);
    gt_block(model, model->scratch_b, model->scratch_a,
             model->dec_conv_cache[2], model->dec_tra_cache[2], &gd2);

    for (size_t c = 0; c < 16; ++c)
        for (size_t f = 0; f < GTCRN_CORE_BINS; ++f)
            model->scratch_a[c][f] += model->en_out1[c][f];
    conv_block_t d3 = {WREF(model, FUSED_DECODER_DE_CONVS_3_CONV_WEIGHT),
                       WFLOAT(model, FUSED_DECODER_DE_CONVS_3_CONV_BIAS),
                       16, 16, 5, 2, 2, 2, 1, 0,
                       WFLOAT(model, RAW_DECODER_DE_CONVS_3_ACT_WEIGHT)};
    conv1xk(model->scratch_a, model->scratch_b,
            16, GTCRN_CORE_BINS, &d3);

    for (size_t c = 0; c < 16; ++c)
        for (size_t f = 0; f < 65; ++f)
            model->scratch_b[c][f] += model->en_out0[c][f];
    conv_block_t d4 = {WREF(model, FUSED_DECODER_DE_CONVS_4_CONV_WEIGHT),
                       WFLOAT(model, FUSED_DECODER_DE_CONVS_4_CONV_BIAS),
                       16, 2, 5, 2, 2, 1, 1, 1, 0};
    conv1xk(model->scratch_b, model->scratch_a,
            16, 65, &d4);

    float mask[2][GTCRN_BINS];
    erb_bs(model, model->scratch_a, mask);
    for (size_t f = 0; f < GTCRN_BINS; ++f) {
        const float real = input_spec[f][0];
        const float imag = input_spec[f][1];
        output_spec[f][0] = real * mask[0][f] - imag * mask[1][f];
        output_spec[f][1] = imag * mask[0][f] + real * mask[1][f];
    }
    return GTCRN_MODEL_OK;
}
