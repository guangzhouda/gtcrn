#include "gtcrn_kernels.h"
#include "gtcrn.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static void assert_close(float actual, float expected, float tolerance) {
    assert(fabsf(actual - expected) <= tolerance);
}

int main(void) {
    gtcrn_state_t state;
    for (size_t i = 0; i < sizeof(state); ++i) ((unsigned char *)&state)[i] = 0xA5;
    gtcrn_state_reset(&state);
    for (size_t i = 0; i < sizeof(state); ++i) assert(((unsigned char *)&state)[i] == 0);
    const float input[] = {1.0f, -2.0f};
    const float weights[] = {2.0f, 3.0f, -1.0f, 4.0f};
    const float bias[] = {0.5f, -0.5f};
    float output[2];
    gtcrn_linear_f32(input, weights, bias, output, 2, 2);
    assert_close(output[0], -3.5f, 1e-6f);
    assert_close(output[1], -9.5f, 1e-6f);

    float norm[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float gamma[] = {1, 1, 1, 1};
    const float beta[] = {0, 0, 0, 0};
    gtcrn_layer_norm_f32(norm, gamma, beta, 4, 1e-8f);
    assert_close(norm[0], -1.3416407f, 1e-5f);
    assert_close(norm[3], 1.3416407f, 1e-5f);

    assert(gtcrn_saturate_i8(200) == 127);
    assert(gtcrn_saturate_i8(-200) == -128);

    const int8_t q_input[] = {2, -4};
    const int8_t q_weights[] = {3, 2};
    const int32_t q_bias[] = {0};
    const gtcrn_requant_t q_params[] = {{INT32_C(1073741824), 0, 0}};
    int8_t q_output[1];
    gtcrn_linear_s8(q_input, q_weights, q_bias, q_params, q_output, 2, 1, 0);
    assert(q_output[0] == -1); /* (2*3 - 4*2) * 0.5 */

    const float conv_input[] = {1, 2, 3, 4, 5};
    const float conv_weight[] = {1, 0, -1};
    float conv_output[5];
    gtcrn_conv2d_f32(conv_input, conv_weight, 0, conv_output,
                     1, 1, 5, 1, 1, 3, 1, 1, 1, 1, 0, 1, 1);
    assert_close(conv_output[0], -2.0f, 1e-6f);
    assert_close(conv_output[2], -2.0f, 1e-6f);
    assert_close(conv_output[4], 4.0f, 1e-6f);

    const float gru_input[] = {1.0f, -1.0f};
    float gru_hidden[] = {0.8f, -0.4f};
    const float gru_w_ih[12] = {0};
    const float gru_w_hh[12] = {0};
    const float gru_b_ih[6] = {0};
    const float gru_b_hh[6] = {0};
    gtcrn_gru_f32(gru_input, gru_hidden, gru_w_ih, gru_w_hh,
                   gru_b_ih, gru_b_hh, 2, 2);
    assert_close(gru_hidden[0], 0.4f, 1e-6f);
    assert_close(gru_hidden[1], -0.2f, 1e-6f);

    puts("gtcrn kernel tests passed");
    return 0;
}
