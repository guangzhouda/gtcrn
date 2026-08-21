#include "gtcrn_audio.h"

#include <assert.h>
#include <string.h>

int main(void) {
    gtcrn_model_t model;
    gtcrn_audio_stream_t stream;
    memset(&model, 0, sizeof(model));
    assert(gtcrn_audio_stream_init(&stream, &model) == GTCRN_MODEL_OK);
    assert(stream.forward_fft != NULL);
    assert(stream.inverse_fft != NULL);
    gtcrn_audio_stream_reset(&stream);
    for (size_t i = 0; i < GTCRN_HOP_SIZE; ++i) {
        assert(stream.input_history[i] == 0.0f);
        assert(stream.ola_tail[i] == 0.0f);
    }
    return 0;
}
