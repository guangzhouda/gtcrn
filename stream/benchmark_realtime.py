"""Benchmark GTCRN frame-by-frame streaming inference on CPU."""

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from gtcrn import GTCRN
from gtcrn_stream import StreamGTCRN
from modules.convert import convert_to_stream


SAMPLE_RATE = 16_000
N_FFT = 512
HOP_LENGTH = 256
FRAME_BUDGET_MS = HOP_LENGTH / SAMPLE_RATE * 1_000


def percentile(values: list[float], percent: float) -> float:
    return float(np.percentile(np.asarray(values), percent))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=Path, default=Path("test_wavs/mix.wav"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("onnx_models/model_trained_on_dns3.tar"))
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    offline_model = GTCRN().eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    offline_model.load_state_dict(checkpoint["model"])
    stream_model = StreamGTCRN().eval()
    convert_to_stream(stream_model, offline_model)

    audio, sample_rate = sf.read(args.wav, dtype="float32")
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz audio, got {sample_rate} Hz")
    if audio.ndim != 1:
        raise ValueError("benchmark input must be mono")

    window = torch.hann_window(N_FFT).sqrt()
    spectrum = torch.view_as_real(
        torch.stft(torch.from_numpy(audio), N_FFT, HOP_LENGTH, N_FFT,
                   window, return_complex=True)
    )[None]
    conv_cache = torch.zeros(2, 1, 16, 16, 33)
    tra_cache = torch.zeros(2, 3, 1, 1, 16)
    inter_cache = torch.zeros(2, 1, 33, 16)

    with torch.inference_mode():
        for index in range(min(args.warmup, spectrum.shape[2])):
            _, conv_cache, tra_cache, inter_cache = stream_model(
                spectrum[:, :, index:index + 1], conv_cache, tra_cache, inter_cache)

        conv_cache.zero_()
        tra_cache.zero_()
        inter_cache.zero_()
        elapsed_ms: list[float] = []
        outputs = []
        for index in range(spectrum.shape[2]):
            started = time.perf_counter()
            output, conv_cache, tra_cache, inter_cache = stream_model(
                spectrum[:, :, index:index + 1], conv_cache, tra_cache, inter_cache)
            elapsed_ms.append((time.perf_counter() - started) * 1_000)
            outputs.append(output)

        offline_output = offline_model(spectrum)

    stream_output = torch.cat(outputs, dim=2)
    max_error = torch.max(torch.abs(stream_output - offline_output)).item()
    mean_ms = statistics.fmean(elapsed_ms)
    deadline_misses = sum(value > FRAME_BUDGET_MS for value in elapsed_ms)

    print(f"audio_seconds: {len(audio) / SAMPLE_RATE:.3f}")
    print(f"frames: {len(elapsed_ms)}")
    print(f"threads: {args.threads}")
    print(f"frame_budget_ms: {FRAME_BUDGET_MS:.3f}")
    print(f"mean_ms: {mean_ms:.3f}")
    print(f"p50_ms: {percentile(elapsed_ms, 50):.3f}")
    print(f"p95_ms: {percentile(elapsed_ms, 95):.3f}")
    print(f"p99_ms: {percentile(elapsed_ms, 99):.3f}")
    print(f"max_ms: {max(elapsed_ms):.3f}")
    print(f"rtf: {mean_ms / FRAME_BUDGET_MS:.4f}")
    print(f"deadline_misses: {deadline_misses}/{len(elapsed_ms)}")
    print(f"stream_offline_max_error: {max_error:.8f}")


if __name__ == "__main__":
    main()
