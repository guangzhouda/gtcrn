#!/usr/bin/env python3
"""Run the native GTCRN frame ABI over a WAV file.

Expected C ABI:

    size_t gtcrn_model_sizeof(void);
    int gtcrn_model_init_f32(void *model, const uint8_t *weights, size_t size);
    int gtcrn_process_frame_f32(void *model, const float *input_ri,
                                float *output_ri);

`input_ri` and `output_ri` are contiguous `[257, 2]` float32 arrays holding
real/imaginary STFT bins. Model storage is opaque to this driver.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
GTCRN_SAMPLE_RATE = 16000
GTCRN_FFT_SIZE = 512
GTCRN_HOP_SIZE = 256
GTCRN_BINS = 257
GTCRN_ERB_BINS = 129
GTCRN_CORE_BINS = 33
GTCRN_CHANNELS = 16
GTCRN_CONV_HISTORY = 16
GTCRN_TRA_LAYERS = 3
GTCRN_PATHS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run native C GTCRN inference through ctypes."
    )
    parser.add_argument("--lib", type=Path, required=True,
                        help="Path to the native shared library (.dll/.so/.dylib).")
    parser.add_argument("--weights-dir", type=Path,
                        default=REPO_ROOT / "native/out/gtcrn_weights_f32",
                        help="Directory containing manifest.json and weights.bin.")
    parser.add_argument("--input-wav", type=Path,
                        default=REPO_ROOT / "test_wavs/mix.wav")
    parser.add_argument("--output-wav", type=Path,
                        default=REPO_ROOT / "native/out/enh_c.wav")
    parser.add_argument("--golden", type=Path,
                        default=REPO_ROOT / "native/out/golden.npz")
    parser.add_argument("--compare-golden", action="store_true",
                        help="Compare frame outputs against native/out/golden.npz.")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Limit frames for ABI bring-up; 0 means all frames.")
    parser.add_argument("--rtol", type=float, default=1.0e-4)
    parser.add_argument("--atol", type=float, default=1.0e-4)
    return parser.parse_args()


def load_weights(weights_dir: Path) -> tuple[ctypes.Array[ctypes.c_uint8], int]:
    manifest_path = weights_dir / "manifest.json"
    weights_path = weights_dir / "weights.bin"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"weights blob not found: {weights_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    blob = weights_path.read_bytes()
    if manifest.get("quantize") != "none":
        raise ValueError(
            "run_c_inference.py currently expects f32 weights; use "
            "native/out/gtcrn_weights_f32 or export with --quantize none"
        )
    buffer_type = ctypes.c_uint8 * len(blob)
    return buffer_type.from_buffer_copy(blob), len(blob)


def configure_abi(lib: ctypes.CDLL) -> tuple[ctypes._CFuncPtr, ctypes._CFuncPtr, int]:
    required = ("gtcrn_model_sizeof", "gtcrn_model_init_f32", "gtcrn_process_frame_f32")
    if any(not hasattr(lib, symbol) for symbol in required):
        raise AttributeError("native library is missing the complete GTCRN model ABI")
    lib.gtcrn_model_sizeof.argtypes = []
    lib.gtcrn_model_sizeof.restype = ctypes.c_size_t
    init = lib.gtcrn_model_init_f32
    init.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    init.restype = ctypes.c_int
    process = lib.gtcrn_process_frame_f32
    process.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    process.restype = ctypes.c_int
    return init, process, int(lib.gtcrn_model_sizeof())


def read_mono_wav(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != GTCRN_SAMPLE_RATE:
        raise ValueError(f"expected 16 kHz WAV, got {sample_rate}: {path}")
    if audio.ndim != 1:
        raise ValueError(f"expected mono WAV: {path}")
    return np.ascontiguousarray(audio, dtype=np.float32)


def stft_frames(audio: np.ndarray) -> torch.Tensor:
    window = torch.hann_window(GTCRN_FFT_SIZE).sqrt()
    return torch.view_as_real(
        torch.stft(
            torch.from_numpy(audio),
            GTCRN_FFT_SIZE,
            GTCRN_HOP_SIZE,
            GTCRN_FFT_SIZE,
            window,
            return_complex=True,
        )
    ).contiguous()


def write_istft(frames_ri: np.ndarray, output_path: Path, length: int) -> None:
    window = torch.hann_window(GTCRN_FFT_SIZE).sqrt()
    complex_frames = torch.view_as_complex(torch.from_numpy(frames_ri).contiguous())
    audio = torch.istft(
        complex_frames,
        GTCRN_FFT_SIZE,
        GTCRN_HOP_SIZE,
        GTCRN_FFT_SIZE,
        window,
        length=length,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio.detach().cpu().numpy(), GTCRN_SAMPLE_RATE)


def compare_frame(golden: np.lib.npyio.NpzFile, frame_index: int,
                  output_frame: np.ndarray, rtol: float, atol: float) -> None:
    key = f"frame_{frame_index:03d}.output"
    if key not in golden:
        raise KeyError(f"golden tensor not found: {key}")
    expected = np.asarray(golden[key]).squeeze()
    if expected.shape != output_frame.shape:
        raise AssertionError(
            f"{key} shape mismatch: expected {expected.shape}, got {output_frame.shape}"
        )
    if not np.allclose(expected, output_frame, rtol=rtol, atol=atol):
        max_abs = float(np.max(np.abs(expected - output_frame)))
        rms = math.sqrt(float(np.mean(np.square(expected - output_frame))))
        raise AssertionError(f"{key} mismatch: max_abs={max_abs:.6g}, rms={rms:.6g}")


def main() -> int:
    args = parse_args()
    lib = ctypes.CDLL(str(args.lib))
    init, process, model_size = configure_abi(lib)
    weights_blob, weights_size = load_weights(args.weights_dir)
    model = ctypes.create_string_buffer(model_size)
    status = init(model, weights_blob, weights_size)
    if status != 0:
        raise RuntimeError(f"gtcrn_model_init_f32 failed: {status}")

    audio = read_mono_wav(args.input_wav)
    spectrum = stft_frames(audio)
    frame_count = spectrum.shape[1]
    if args.max_frames > 0:
        frame_count = min(frame_count, args.max_frames)

    outputs = np.zeros((GTCRN_BINS, spectrum.shape[1], 2), dtype=np.float32)
    golden = np.load(args.golden) if args.compare_golden else None
    frame_times_ms: list[float] = []
    try:
        for frame_index in range(frame_count):
            input_frame = np.ascontiguousarray(spectrum[:, frame_index, :].numpy(),
                                               dtype=np.float32)
            output_frame = np.zeros((GTCRN_BINS, 2), dtype=np.float32)
            started = time.perf_counter()
            status = process(
                model,
                input_frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                output_frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            frame_times_ms.append((time.perf_counter() - started) * 1000.0)
            if status != 0:
                raise RuntimeError(f"gtcrn_process_frame_f32 failed at frame {frame_index}: {status}")
            outputs[:, frame_index, :] = output_frame
            if golden is not None:
                compare_frame(golden, frame_index, output_frame, args.rtol, args.atol)
    finally:
        if golden is not None:
            golden.close()

    if frame_count < spectrum.shape[1]:
        outputs[:, frame_count:, :] = spectrum[:, frame_count:, :].numpy()
    write_istft(outputs, args.output_wav, len(audio))
    print(f"frames: {frame_count}")
    print(f"mean_frame_ms: {float(np.mean(frame_times_ms)):.3f}")
    print(f"p95_frame_ms: {float(np.percentile(frame_times_ms, 95)):.3f}")
    print(f"max_frame_ms: {float(np.max(frame_times_ms)):.3f}")
    print(f"rtf: {float(np.mean(frame_times_ms)) / 16.0:.4f}")
    print(f"output: {args.output_wav}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
