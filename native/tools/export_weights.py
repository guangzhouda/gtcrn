#!/usr/bin/env python3
"""Export StreamGTCRN weights for native inference.

The exporter writes a stable little-endian tensor blob plus a JSON manifest.
It intentionally uses the repository's existing offline->stream conversion so
the native runtime consumes exactly the streaming-layout weights used by
`stream/gtcrn_stream.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
STREAM_DIR = REPO_ROOT / "stream"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STREAM_DIR) not in sys.path:
    sys.path.insert(0, str(STREAM_DIR))

from gtcrn import GTCRN  # noqa: E402
from gtcrn_stream import StreamGTCRN  # noqa: E402
from modules.convert import convert_to_stream  # noqa: E402


ALIGNMENT_BYTES = 16
MANIFEST_NAME = "manifest.json"
WEIGHTS_NAME = "weights.bin"
INDEX_NAME = "weights_index.h"


@dataclass(frozen=True)
class TensorRecord:
    name: str
    array: np.ndarray
    source: str
    role: str
    layout: str
    op_path: str | None = None
    quantization: dict[str, Any] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export converted StreamGTCRN weights for C/C++ inference."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "model_trained_on_dns3.tar",
        help="PyTorch checkpoint containing a 'model' state_dict.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "native" / "out" / "gtcrn_weights",
        help="Directory for manifest.json and weights.bin.",
    )
    parser.add_argument(
        "--emit-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit all stream-model state_dict tensors except bookkeeping counters.",
    )
    parser.add_argument(
        "--emit-fused",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit Conv/BatchNorm fused tensors for native conv kernels.",
    )
    parser.add_argument(
        "--quantize",
        choices=("none", "int8-weight"),
        default="none",
        help="Optional weight-only symmetric int8 export. Activations still need runtime calibration.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Read the emitted files back and verify sha256, offsets, and sizes.",
    )
    return parser.parse_args()


def load_stream_model(checkpoint_path: Path) -> StreamGTCRN:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    offline_model = GTCRN().eval()
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch 1.11 used by the upstream repository.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("checkpoint must be a dict containing key 'model'")
    offline_model.load_state_dict(checkpoint["model"])

    stream_model = StreamGTCRN().eval()
    convert_to_stream(stream_model, offline_model)
    return stream_model


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().contiguous().numpy()
    if array.dtype.kind in ("f", "i", "u"):
        return array
    raise TypeError(f"unsupported tensor dtype: {array.dtype}")


def role_for_name(name: str) -> str:
    leaf = name.rsplit(".", 1)[-1]
    if ".erb_" in name:
        return "erb_linear"
    if ".att_gru." in name or "_rnn.rnn" in name:
        return "gru"
    if ".att_fc." in name or name.endswith("_fc.weight") or name.endswith("_fc.bias"):
        return "linear"
    if "_ln." in name:
        return "layer_norm"
    if ".bn." in name or "_bn" in name:
        return "batch_norm"
    if ".conv." in name or "Conv2d." in name or "ConvTranspose2d." in name:
        return "conv2d"
    if ".act.weight" in name or "_act.weight" in name:
        return "prelu"
    if leaf in {"weight", "bias"}:
        return "parameter"
    return "state"


def layout_for_name(name: str, array: np.ndarray) -> str:
    if array.ndim == 4:
        return "OIHW"
    if array.ndim == 2:
        if "weight_ih" in name or "weight_hh" in name:
            return "GRU_PYTORCH_GATE_ROW_MAJOR"
        return "row_major"
    if array.ndim == 1:
        return "vector"
    if array.ndim == 0:
        return "scalar"
    return "row_major"


def collect_raw_tensors(stream_model: StreamGTCRN) -> list[TensorRecord]:
    records: list[TensorRecord] = []
    for name, tensor in stream_model.state_dict().items():
        if name.endswith("num_batches_tracked"):
            continue
        array = tensor_to_numpy(tensor)
        if array.dtype.kind not in ("f",):
            continue
        parent_path = name.rsplit(".", 1)[0]
        try:
            parent = get_submodule(stream_model, parent_path)
        except (AttributeError, IndexError, KeyError):
            parent = None
        layout = (
            "IOHW_PYTORCH_CONV_TRANSPOSE"
            if isinstance(parent, nn.ConvTranspose2d) and array.ndim == 4
            else layout_for_name(name, array)
        )
        records.append(
            TensorRecord(
                name=f"raw.{name}",
                array=array.astype("<f4", copy=False),
                source=name,
                role=role_for_name(name),
                layout=layout,
            )
        )
    return records


def get_submodule(root: nn.Module, path: str) -> nn.Module:
    module: nn.Module = root
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def resolve_conv(module: nn.Module) -> nn.Module:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        return module
    for attr in ("Conv2d", "ConvTranspose2d"):
        if hasattr(module, attr):
            conv = getattr(module, attr)
            if isinstance(conv, nn.Conv2d):
                return conv
    raise TypeError(f"module is not a supported conv wrapper: {module.__class__.__name__}")


def fuse_conv_bn(conv_module: nn.Module, bn: nn.BatchNorm2d) -> tuple[np.ndarray, np.ndarray]:
    conv = resolve_conv(conv_module)
    weight = conv.weight.detach().cpu().float()
    if isinstance(conv, nn.ConvTranspose2d):
        # ConvTranspose2d stores [I, O/groups, H, W]. Native inference uses
        # zero insertion followed by Conv2d, whose layout is [O, I/groups,H,W].
        groups = conv.groups
        in_per_group = conv.in_channels // groups
        out_per_group = conv.out_channels // groups
        weight = (
            weight.reshape(groups, in_per_group, out_per_group, *weight.shape[2:])
            .permute(0, 2, 1, 3, 4)
            .reshape(conv.out_channels, in_per_group, *weight.shape[2:])
            .flip(-2, -1)
        )
    bias = (
        conv.bias.detach().cpu().float()
        if conv.bias is not None
        else torch.zeros(weight.shape[0], dtype=torch.float32)
    )
    gamma = bn.weight.detach().cpu().float()
    beta = bn.bias.detach().cpu().float()
    mean = bn.running_mean.detach().cpu().float()
    var = bn.running_var.detach().cpu().float()
    scale = gamma * torch.rsqrt(var + bn.eps)
    fused_weight = weight * scale.reshape([-1, 1, 1, 1])
    fused_bias = (bias - mean) * scale + beta
    return (
        fused_weight.contiguous().numpy().astype("<f4", copy=False),
        fused_bias.contiguous().numpy().astype("<f4", copy=False),
    )


def collect_fused_tensors(stream_model: StreamGTCRN) -> list[TensorRecord]:
    pairs = [
        ("encoder.en_convs.0.conv", "encoder.en_convs.0.bn"),
        ("encoder.en_convs.1.conv", "encoder.en_convs.1.bn"),
        ("encoder.en_convs.2.point_conv1", "encoder.en_convs.2.point_bn1"),
        ("encoder.en_convs.2.depth_conv", "encoder.en_convs.2.depth_bn"),
        ("encoder.en_convs.2.point_conv2", "encoder.en_convs.2.point_bn2"),
        ("encoder.en_convs.3.point_conv1", "encoder.en_convs.3.point_bn1"),
        ("encoder.en_convs.3.depth_conv", "encoder.en_convs.3.depth_bn"),
        ("encoder.en_convs.3.point_conv2", "encoder.en_convs.3.point_bn2"),
        ("encoder.en_convs.4.point_conv1", "encoder.en_convs.4.point_bn1"),
        ("encoder.en_convs.4.depth_conv", "encoder.en_convs.4.depth_bn"),
        ("encoder.en_convs.4.point_conv2", "encoder.en_convs.4.point_bn2"),
        ("decoder.de_convs.0.point_conv1", "decoder.de_convs.0.point_bn1"),
        ("decoder.de_convs.0.depth_conv", "decoder.de_convs.0.depth_bn"),
        ("decoder.de_convs.0.point_conv2", "decoder.de_convs.0.point_bn2"),
        ("decoder.de_convs.1.point_conv1", "decoder.de_convs.1.point_bn1"),
        ("decoder.de_convs.1.depth_conv", "decoder.de_convs.1.depth_bn"),
        ("decoder.de_convs.1.point_conv2", "decoder.de_convs.1.point_bn2"),
        ("decoder.de_convs.2.point_conv1", "decoder.de_convs.2.point_bn1"),
        ("decoder.de_convs.2.depth_conv", "decoder.de_convs.2.depth_bn"),
        ("decoder.de_convs.2.point_conv2", "decoder.de_convs.2.point_bn2"),
        ("decoder.de_convs.3.conv", "decoder.de_convs.3.bn"),
        ("decoder.de_convs.4.conv", "decoder.de_convs.4.bn"),
    ]

    records: list[TensorRecord] = []
    for conv_path, bn_path in pairs:
        weight, bias = fuse_conv_bn(get_submodule(stream_model, conv_path), get_submodule(stream_model, bn_path))
        records.append(
            TensorRecord(
                name=f"fused.{conv_path}.weight",
                array=weight,
                source=f"{conv_path} + {bn_path}",
                role="conv2d_fused_bn",
                layout="OIHW",
                op_path=conv_path,
            )
        )
        records.append(
            TensorRecord(
                name=f"fused.{conv_path}.bias",
                array=bias,
                source=f"{conv_path} + {bn_path}",
                role="conv2d_fused_bn_bias",
                layout="vector",
                op_path=conv_path,
            )
        )
    return records


def quantize_record(record: TensorRecord) -> TensorRecord:
    array = record.array.astype(np.float32, copy=False)
    leaf = record.name.rsplit(".", 1)[-1]
    quantizable_role = record.role in {
        "conv2d", "conv2d_fused_bn", "gru", "linear"
    }
    is_weight = leaf == "weight" or leaf.startswith("weight_")
    is_fixed_erb_matrix = record.name.startswith("raw.erb.")
    if array.ndim == 0 or not quantizable_role or not is_weight or is_fixed_erb_matrix:
        return record

    if (record.role.startswith("conv2d") and array.ndim == 4) or array.ndim == 2:
        axes = tuple(range(1, array.ndim))
        max_abs = np.max(np.abs(array), axis=axes, keepdims=True)
        scale = np.maximum(max_abs / 127.0, 1.0e-12).astype("<f4")
        quantized = np.clip(np.rint(array / scale), -127, 127).astype(np.int8)
        qmeta: dict[str, Any] = {
            "scheme": "symmetric_per_output_channel",
            "scale_dtype": "float32",
            "scale_shape": list(scale.reshape(-1).shape),
            "zero_point": 0,
        }
        return TensorRecord(
            name=record.name,
            array=quantized,
            source=record.source,
            role=record.role,
            layout=record.layout,
            op_path=record.op_path,
            quantization=qmeta | {"scale": scale.reshape(-1).tolist()},
        )

    max_abs = float(np.max(np.abs(array)))
    scale_value = max(max_abs / 127.0, 1.0e-12)
    quantized = np.clip(np.rint(array / scale_value), -127, 127).astype(np.int8)
    return TensorRecord(
        name=record.name,
        array=quantized,
        source=record.source,
        role=record.role,
        layout=record.layout,
        op_path=record.op_path,
        quantization={
            "scheme": "symmetric_per_tensor",
            "scale_dtype": "float32",
            "scale": scale_value,
            "zero_point": 0,
        },
    )


def quantize_records(records: list[TensorRecord]) -> list[TensorRecord]:
    exported: list[TensorRecord] = []
    for record in records:
        quantized = quantize_record(record)
        metadata = quantized.quantization
        if metadata is None:
            exported.append(quantized)
            continue
        scale_values = metadata["scale"]
        scale_array = np.asarray(scale_values, dtype="<f4").reshape(-1)
        scale_name = f"{quantized.name}.scale"
        exported.append(replace(
            quantized,
            quantization={
                key: value for key, value in metadata.items() if key != "scale"
            } | {"scale_tensor": scale_name},
        ))
        exported.append(TensorRecord(
            name=scale_name,
            array=scale_array,
            source=quantized.source,
            role="quantization_scale",
            layout="vector",
            op_path=quantized.op_path,
        ))
    return exported


def pad_to_alignment(handle: Any, offset: int) -> int:
    padding = (ALIGNMENT_BYTES - (offset % ALIGNMENT_BYTES)) % ALIGNMENT_BYTES
    if padding:
        handle.write(b"\x00" * padding)
    return offset + padding


def tensor_stats(array: np.ndarray) -> dict[str, float]:
    if array.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    view = array.astype(np.float32, copy=False)
    return {
        "min": float(np.min(view)),
        "max": float(np.max(view)),
        "mean": float(np.mean(view)),
        "std": float(np.std(view)),
    }


def build_erb_sparse(records: list[TensorRecord]) -> tuple[list[TensorRecord], dict[str, Any]]:
    erb_name = "raw.erb.erb_fc.weight"
    ierb_name = "raw.erb.ierb_fc.weight"
    erb_record = next(record for record in records if record.name == erb_name)
    ierb_record = next(record for record in records if record.name == ierb_name)
    erb = erb_record.array.astype(np.float32, copy=False)
    ierb = ierb_record.array.astype(np.float32, copy=False)
    if erb.shape != (64, 192) or not np.array_equal(ierb, erb.T):
        raise ValueError("unexpected ERB matrices; sparse transpose contract is invalid")
    rows, cols = np.nonzero(erb)
    row_ptr = np.zeros(erb.shape[0] + 1, dtype=np.uint16)
    np.add.at(row_ptr, rows + 1, 1)
    row_ptr = np.cumsum(row_ptr, dtype=np.uint16)
    sparse = {
        "row_ptr": row_ptr,
        "col_index": cols.astype(np.uint8),
        "values": erb[rows, cols].astype("<f4"),
    }
    filtered = [record for record in records if record.name not in {erb_name, ierb_name}]
    return filtered, sparse


def write_export(records: list[TensorRecord], out_dir: Path, checkpoint: Path,
                 quantize: str, erb_sparse: dict[str, Any]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / WEIGHTS_NAME
    manifest_path = out_dir / MANIFEST_NAME
    tensors: list[dict[str, Any]] = []

    sha = hashlib.sha256()
    offset = 0
    with weights_path.open("wb") as handle:
        for index, record in enumerate(records):
            offset = pad_to_alignment(handle, offset)
            data = np.ascontiguousarray(record.array)
            payload = data.tobytes(order="C")
            handle.write(payload)
            sha.update(payload)
            tensors.append(
                {
                    "index": index,
                    "name": record.name,
                    "source": record.source,
                    "role": record.role,
                    "layout": record.layout,
                    "shape": list(data.shape),
                    "dtype": str(data.dtype),
                    "offset_bytes": offset,
                    "nbytes": len(payload),
                    "op_path": record.op_path,
                    "stats": tensor_stats(data),
                    "quantization": record.quantization,
                }
            )
            offset += len(payload)

    manifest: dict[str, Any] = {
        "format": "gtcrn-native-weights",
        "format_version": 1,
        "model": "StreamGTCRN",
        "source_checkpoint": os.fspath(checkpoint),
        "weights_file": WEIGHTS_NAME,
        "weights_sha256_payload_only": sha.hexdigest(),
        "alignment_bytes": ALIGNMENT_BYTES,
        "byte_order": "little",
        "quantize": quantize,
        "runtime_contract": {
            "input_frame_shape": [1, 257, 1, 2],
            "conv_cache_shape": [2, 1, 16, 16, 33],
            "tra_cache_shape": [2, 3, 1, 1, 16],
            "inter_cache_shape": [2, 1, 33, 16],
            "stft": {"n_fft": 512, "hop_length": 256, "win_length": 512, "sample_rate": 16000},
            "gru_gate_order": "PyTorch GRU order: reset, update, new",
        },
        "notes": [
            "raw.* tensors mirror StreamGTCRN.state_dict() after convert_to_stream().",
            "fused.* tensors fold BatchNorm2d into preceding Conv2d-compatible kernels.",
            "int8-weight export stores learned weights; the portable runtime dynamically quantizes each activation vector.",
            "Fixed ERB and inverse-ERB matrices are emitted once as an exact sparse transpose table in weights_index.h.",
        ],
        "erb_sparse": {
            "rows": 64,
            "columns": 192,
            "nonzero": int(erb_sparse["values"].size),
            "storage_bytes": int(sum(value.nbytes for value in erb_sparse.values())),
        },
        "tensors": tensors,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_c_index(out_dir / INDEX_NAME, manifest, offset, erb_sparse)
    return manifest


def c_identifier(name: str) -> str:
    chars = [char.upper() if char.isalnum() else "_" for char in name]
    return "".join(chars)


def c_array(values: np.ndarray, formatter: Any, per_line: int = 12) -> list[str]:
    formatted = [formatter(value) for value in values.reshape(-1)]
    return ["    " + ", ".join(formatted[i:i + per_line]) + ","
            for i in range(0, len(formatted), per_line)]


def write_c_index(path: Path, manifest: dict[str, Any], blob_size: int,
                  erb_sparse: dict[str, Any]) -> None:
    lines = [
        "/* Generated by native/tools/export_weights.py. */",
        "#ifndef GTCRN_WEIGHTS_INDEX_H",
        "#define GTCRN_WEIGHTS_INDEX_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define GTCRN_WEIGHTS_BLOB_SIZE {blob_size}u",
        f"#define GTCRN_WEIGHTS_TENSOR_COUNT {len(manifest['tensors'])}u",
        "",
    ]
    for tensor in manifest["tensors"]:
        prefix = "GTCRN_TENSOR_" + c_identifier(tensor["name"])
        lines.append(f"#define {prefix}_OFFSET {tensor['offset_bytes']}u")
        lines.append(f"#define {prefix}_NBYTES {tensor['nbytes']}u")
    row_ptr = erb_sparse["row_ptr"]
    col_index = erb_sparse["col_index"]
    values = erb_sparse["values"]
    lines.extend([
        "",
        f"#define GTCRN_ERB_SPARSE_NNZ {values.size}u",
        "static const uint16_t gtcrn_erb_row_ptr[65] = {",
        *c_array(row_ptr, lambda value: f"{int(value)}u"),
        "};",
        f"static const uint8_t gtcrn_erb_col_index[{values.size}] = {{",
        *c_array(col_index, lambda value: f"{int(value)}u"),
        "};",
        f"static const float gtcrn_erb_values[{values.size}] = {{",
        *c_array(values, lambda value: f"{float(value).hex()}f", per_line=6),
        "};",
    ])
    lines.extend(["", "#endif", ""])
    path.write_text("\n".join(lines), encoding="ascii")


def verify_export(out_dir: Path, manifest: dict[str, Any]) -> None:
    weights_path = out_dir / manifest["weights_file"]
    blob = weights_path.read_bytes()
    sha = hashlib.sha256()
    previous_end = 0
    for tensor in manifest["tensors"]:
        offset = int(tensor["offset_bytes"])
        nbytes = int(tensor["nbytes"])
        if offset % int(manifest["alignment_bytes"]) != 0:
            raise AssertionError(f"misaligned tensor: {tensor['name']}")
        if offset < previous_end:
            raise AssertionError(f"overlapping tensor: {tensor['name']}")
        payload = blob[offset : offset + nbytes]
        if len(payload) != nbytes:
            raise AssertionError(f"truncated tensor: {tensor['name']}")
        sha.update(payload)
        previous_end = offset + nbytes
        itemsize = np.dtype(tensor["dtype"]).itemsize
        expected_elems = math.prod(tensor["shape"]) if tensor["shape"] else 1
        if expected_elems * itemsize != nbytes:
            raise AssertionError(f"size mismatch: {tensor['name']}")
    if sha.hexdigest() != manifest["weights_sha256_payload_only"]:
        raise AssertionError("weights sha256 mismatch")


def main() -> None:
    args = parse_args()
    stream_model = load_stream_model(args.checkpoint)

    records: list[TensorRecord] = []
    if args.emit_raw:
        records.extend(collect_raw_tensors(stream_model))
    if args.emit_fused:
        records.extend(collect_fused_tensors(stream_model))
    if not records:
        raise ValueError("nothing to export; enable --emit-raw and/or --emit-fused")

    records, erb_sparse = build_erb_sparse(records)

    if args.quantize == "int8-weight":
        records = quantize_records(records)

    manifest = write_export(records, args.out_dir, args.checkpoint, args.quantize, erb_sparse)
    if args.verify:
        verify_export(args.out_dir, manifest)

    total_bytes = sum(int(item["nbytes"]) for item in manifest["tensors"])
    print(f"exported_tensors: {len(manifest['tensors'])}")
    print(f"payload_bytes: {total_bytes}")
    print(f"manifest: {args.out_dir / MANIFEST_NAME}")
    print(f"weights: {args.out_dir / WEIGHTS_NAME}")
    print(f"index: {args.out_dir / INDEX_NAME}")
    if args.verify:
        print("verify: ok")


if __name__ == "__main__":
    main()
