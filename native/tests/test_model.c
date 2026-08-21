#include "gtcrn_model.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

int main(void) {
    gtcrn_model_t model;
    const uint8_t invalid_weights[16] = {0};
    assert(gtcrn_model_sizeof() == sizeof(gtcrn_model_t));
    assert(gtcrn_model_init_f32(&model, invalid_weights, sizeof(invalid_weights))
           == GTCRN_MODEL_BAD_WEIGHTS);
    puts("gtcrn model ABI test passed");
    return 0;
}
