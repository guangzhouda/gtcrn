# Native GTCRN inference

This directory contains a runtime-free, fixed-shape implementation path for the
streaming GTCRN model. ONNX Runtime is not used. It includes a complete FP32
reference and a runtime-free INT8/Q15 implementation.

The public streaming contract is one complex STFT frame per call:

- input/output: 257 complex bins;
- hop: 256 samples at 16 kHz (16 ms);
- convolution cache: `[2, 16, 16, 33]`;
- TRA cache: `[2, 3, 16]`;
- inter-GRU cache: `[2, 33, 16]`.

Export FP32 streaming weights, then build the host library and tests:

```sh
python native/tools/export_weights.py --out-dir native/out/gtcrn_weights_f32 --quantize none --verify
cmake -S native -B native/build
cmake --build native/build
ctest --test-dir native/build --output-on-failure
```

Run the C model over the repository test WAV:

```sh
python native/tools/run_c_inference.py --lib native/build/libgtcrn_native.dll --output-wav test_wavs/enh_c_f32.wav
```

The floating-point kernels are the numerical oracle. The signed-int8 linear
kernel uses int32 accumulation and per-output-channel requantization. On H743,
replace hot kernels with CMSIS-NN implementations only after matching these
reference tests.

Export the INT8 package and build the INT8/Q15 target:

```sh
python native/tools/export_weights.py --quantize int8-weight --verify
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build --config Release
native/build/gtcrn_cli_int8 native/out/gtcrn_weights/weights.bin input.wav output.wav
native/build/gtcrn_cli_int8 native/out/gtcrn_weights/weights.bin input.wav stream.wav --stream
```

This writes `weights.bin`, `manifest.json`, and `weights_index.h` under
`native/out/gtcrn_weights`. All learned convolution, linear, and GRU MACs use
signed INT8 inputs/weights, INT32 accumulation, and per-output weight scales.
Cross-frame convolution and recurrent states are Q15. Biases, fixed ERB
matrices, LayerNorm/PReLU, nonlinearities, and inter-layer scratch remain float,
so this is a hybrid integer implementation rather than a bit-exact all-integer
graph. The generated directory is deliberately ignored by Git.

`gtcrn_audio_stream_init/process` is a heap-free causal KissFFT front end with
256-sample blocks and one-hop (16 ms) algorithmic delay. The WAV CLI uses dr_wav
as a host test harness; embedded SAI/I2S DMA should call the streaming API.

On `test_wavs/mix.wav`, `enh_c_int8_q15.wav` has correlation 0.999576 and RMSE
0.001975 against pure-C FP32. Tight per-stage skip buffers and 129-bin scratch
reduce the INT8/Q15 model object from 160,528 to 107,280 bytes without changing
the samples. The heap-free streaming audio context is 16,408 bytes, for 123,688
bytes of persistent RAM. The packed
INT8 parameter file is 67,320 bytes. The fixed ERB transform is stored once as
a 2,040-byte sparse table in generated C read-only data instead of two dense
96 KiB matrices. This removes 98,304 bytes from the weight blob and reduces ERB
work from 61,440 to 1,910 MAC per frame without retraining.

Generate a short, per-layer numerical reference trace:

```sh
python native/tools/generate_golden.py --frames 4
```

The resulting ignored `native/out/golden.npz` contains every leaf-module
output plus all recurrent caches after each frame.
