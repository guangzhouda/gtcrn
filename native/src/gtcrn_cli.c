#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include "gtcrn_audio.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint8_t *read_binary(const char *path, size_t *size) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) return NULL;
    if (fseek(file, 0, SEEK_END) != 0) { fclose(file); return NULL; }
    const long length = ftell(file);
    if (length <= 0 || fseek(file, 0, SEEK_SET) != 0) { fclose(file); return NULL; }
    uint8_t *data = (uint8_t *)malloc((size_t)length);
    if (data == NULL || fread(data, 1, (size_t)length, file) != (size_t)length) {
        free(data); fclose(file); return NULL;
    }
    fclose(file);
    *size = (size_t)length;
    return data;
}

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) {
        fprintf(stderr, "usage: %s weights.bin input.wav output.wav [--stream]\n", argv[0]);
        return 2;
    }
    const int streaming = argc == 5 && strcmp(argv[4], "--stream") == 0;
    if (argc == 5 && !streaming) return 2;
    size_t weights_size = 0;
    uint8_t *weights = read_binary(argv[1], &weights_size);
    if (weights == NULL) { fprintf(stderr, "cannot read weights\n"); return 3; }

    unsigned channels = 0, sample_rate = 0;
    drwav_uint64 frame_count = 0;
    float *input = drwav_open_file_and_read_pcm_frames_f32(
        argv[2], &channels, &sample_rate, &frame_count, NULL);
    if (input == NULL || channels != 1 || sample_rate != GTCRN_SAMPLE_RATE) {
        fprintf(stderr, "input must be mono 16 kHz WAV\n");
        drwav_free(input, NULL); free(weights); return 4;
    }
    float *output = (float *)malloc((size_t)frame_count * sizeof(float));
    gtcrn_model_t *model = (gtcrn_model_t *)malloc(sizeof(gtcrn_model_t));
    if (output == NULL || model == NULL) {
        drwav_free(input, NULL); free(output); free(model); free(weights); return 5;
    }
    int status = gtcrn_model_init(model, weights, weights_size);
    if (status == GTCRN_MODEL_OK && streaming) {
        gtcrn_audio_stream_t stream;
        status = gtcrn_audio_stream_init(&stream, model);
        for (size_t offset = 0; status == GTCRN_MODEL_OK && offset < (size_t)frame_count;
             offset += GTCRN_HOP_SIZE) {
            float in_hop[GTCRN_HOP_SIZE] = {0}, out_hop[GTCRN_HOP_SIZE];
            size_t count = (size_t)frame_count - offset;
            if (count > GTCRN_HOP_SIZE) count = GTCRN_HOP_SIZE;
            memcpy(in_hop, input + offset, count * sizeof(float));
            status = gtcrn_audio_stream_process(&stream, in_hop, out_hop);
            memcpy(output + offset, out_hop, count * sizeof(float));
        }
    } else if (status == GTCRN_MODEL_OK) {
        status = gtcrn_enhance_audio_f32(model, input, (size_t)frame_count, output);
    }
    if (status != GTCRN_MODEL_OK) {
        fprintf(stderr, "inference failed: %d\n", status);
        drwav_free(input, NULL); free(output); free(model); free(weights); return 6;
    }

    drwav_data_format format = {drwav_container_riff, DR_WAVE_FORMAT_PCM,
                                1, GTCRN_SAMPLE_RATE, 16};
    drwav_int16 *pcm16 = (drwav_int16 *)malloc((size_t)frame_count * sizeof(drwav_int16));
    if (pcm16 == NULL) {
        drwav_free(input, NULL); free(output); free(model); free(weights); return 7;
    }
    drwav_f32_to_s16(pcm16, output, (size_t)frame_count);
    drwav writer;
    if (!drwav_init_file_write(&writer, argv[3], &format, NULL)) {
        fprintf(stderr, "cannot create output WAV\n");
        drwav_free(input, NULL); free(pcm16); free(output); free(model); free(weights); return 8;
    }
    drwav_write_pcm_frames(&writer, frame_count, pcm16);
    drwav_uninit(&writer);
    printf("samples: %llu\n", (unsigned long long)frame_count);
    printf("output: %s\n", argv[3]);
    drwav_free(input, NULL); free(pcm16); free(output); free(model); free(weights);
    return 0;
}
