#!/usr/bin/env python3
"""Generate per-frame and per-leaf-module StreamGTCRN reference tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
STREAM_DIR = REPO_ROOT / "stream"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(STREAM_DIR))

from gtcrn import GTCRN  # noqa: E402
from gtcrn_stream import StreamGTCRN  # noqa: E402
from modules.convert import convert_to_stream  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO_ROOT / "checkpoints/model_trained_on_dns3.tar")
    parser.add_argument("--wav", type=Path,
                        default=REPO_ROOT / "stream/test_wavs/mix.wav")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "native/out/golden.npz")
    return parser.parse_args()


def tensor_outputs(value: object) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [item for item in value if isinstance(item, torch.Tensor)]
    return []


def main() -> None:
    args = parse_args()
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch 1.11 compatibility.
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    offline = GTCRN().eval()
    offline.load_state_dict(checkpoint["model"])
    model = StreamGTCRN().eval()
    convert_to_stream(model, offline)

    audio, sample_rate = sf.read(args.wav, dtype="float32")
    if sample_rate != 16000 or audio.ndim != 1:
        raise ValueError("golden WAV must be mono 16 kHz")
    window = torch.hann_window(512).sqrt()
    spectrum = torch.view_as_real(torch.stft(
        torch.from_numpy(audio), 512, 256, 512, window, return_complex=True
    ))[None]

    captured: dict[str, np.ndarray] = {}
    current_frame = [0]
    leaf_types = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.GRU,
                  nn.LayerNorm, nn.PReLU, nn.BatchNorm2d)
    handles = []
    for name, module in model.named_modules():
        if not isinstance(module, leaf_types):
            continue

        def hook(_module: nn.Module, _inputs: object, output: object,
                 module_name: str = name) -> None:
            for index, tensor in enumerate(tensor_outputs(output)):
                key = f"frame_{current_frame[0]:03d}.{module_name}.out_{index}"
                captured[key] = tensor.detach().cpu().contiguous().numpy()

        handles.append(module.register_forward_hook(hook))

    conv = torch.zeros(2, 1, 16, 16, 33)
    tra = torch.zeros(2, 3, 1, 1, 16)
    inter = torch.zeros(2, 1, 33, 16)
    frame_count = min(args.frames, spectrum.shape[2])
    with torch.inference_mode():
        for frame in range(frame_count):
            current_frame[0] = frame
            model_input = spectrum[:, :, frame:frame + 1]
            captured[f"frame_{frame:03d}.input"] = model_input.numpy()
            output, conv, tra, inter = model(model_input, conv, tra, inter)
            captured[f"frame_{frame:03d}.output"] = output.numpy()
            captured[f"frame_{frame:03d}.conv_cache"] = conv.numpy().copy()
            captured[f"frame_{frame:03d}.tra_cache"] = tra.numpy().copy()
            captured[f"frame_{frame:03d}.inter_cache"] = inter.numpy().copy()

    for handle in handles:
        handle.remove()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **captured)
    metadata = {
        "format": "gtcrn-native-golden",
        "frames": frame_count,
        "tensor_count": len(captured),
        "source_wav": str(args.wav),
        "source_checkpoint": str(args.checkpoint),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"frames: {frame_count}")
    print(f"tensors: {len(captured)}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
