# STM32H743 integration notes

Build `gtcrn_native_int8` with `GTCRN_INT8_WEIGHTS` and `GTCRN_Q15_STATE`.
The model object is currently 107,280 bytes and the heap-free streaming audio
context is 16,408 bytes, totaling 123,688 bytes of persistent RAM. The packed
weight file is 67,320 bytes and the generated exact sparse ERB table occupies
2,040 bytes of read-only program data. GCC's host-side static stack report gives
an approximately 29,856-byte deepest call chain; the ARM map/stack report must
be checked separately. Persistent state plus stack therefore does not fit in
the H743's 128 KiB DTCM alone. Keep the main context in AXI SRAM, profile, then
move selected hot state into DTCM. Put DMA buffers in D2 SRAM, not DTCM.

Suggested placement:

- ITCM: hot convolution, requantization, and GRU functions;
- AXI SRAM: `gtcrn_model_t` (32-byte aligned);
- DTCM: stack and selected hot buffers after map-file profiling;
- Flash/AXI SRAM: packed weights;
- D2 SRAM: SAI/I2S ping-pong DMA buffers.

Enable I-Cache and D-Cache. If a DMA buffer is placed in a cacheable region,
clean/invalidate cache lines at ownership transitions and align buffers to the
cache-line size.

Use `gtcrn_cycle_counter.h` to measure each stage. At 480 MHz and a 256-sample
hop, the hard budget is 7,680,000 cycles. Keep the inference path below about
6,000,000 cycles to reserve time for FFT, DMA handling, and interrupts.

The host streaming front end keeps heap-free KissFFT plans inside
`gtcrn_audio_stream_t`. On STM32, keep KissFFT initially or replace its two RFFT
calls with CMSIS-DSP `arm_rfft_fast_f32`; the model API stays unchanged.
CMSIS-NN can later replace the `matvec` and convolution hot loops. Neither path
needs a graph runtime or ONNX parser.

Add `gtcrn_stm32_port.c`, the native model/audio sources, and KissFFT sources to
the CubeIDE project. Define `GTCRN_INT8_WEIGHTS=1`, `GTCRN_Q15_STATE=1`, and
compile with Cortex-M7 DSP/FPU options (`-mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16
-mfloat-abi=hard`). Place one 32-byte-aligned `gtcrn_stm32_context_t` in AXI
SRAM. Each SAI/I2S half-transfer callback passes exactly 256 mono PCM16 samples
to `gtcrn_stm32_process_s16`; send its 256 output samples to the playback DMA
half that is not currently owned by DMA.

Convert `weights.bin` to a linkable flash object with the toolchain's `objcopy`
or include it as a CubeIDE binary resource, then pass its start/end linker
symbols to `gtcrn_stm32_init`. Do not copy the blob into RAM.
