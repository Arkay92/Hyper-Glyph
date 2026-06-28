"""Serialization helpers for compressed models."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .archive import load_compact_hwz, save_compact_hwz
from .codec import CompressedModel, CompressedTensor
from .compact_codec import CompactCompressedModel


def save_compressed(
    compressed_model: CompressedModel | CompactCompressedModel, path: str | Path
) -> None:
    """Save a compressed model to a .hwz zip archive."""
    if isinstance(compressed_model, CompactCompressedModel):
        save_compact_hwz(compressed_model, path)
        compressed_model.payload_breakdown["metadata_bytes"] = len(
            json.dumps(compressed_model.metadata, sort_keys=True).encode("utf-8")
        )
        compressed_model.payload_breakdown["archive_total_bytes"] = Path(path).stat().st_size
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        metadata = {
            "format_version": compressed_model.format_version,
            "tensors": {
                name: tensor_to_dict(tensor, include_prototype_matrix=False)
                for name, tensor in compressed_model.tensors.items()
            },
        }
        archive.writestr("metadata.json", json.dumps(metadata, indent=2))
        prototype_arrays = {}
        for name, tensor in compressed_model.tensors.items():
            prototype_arrays[f"{name}_prototypes"] = tensor.prototype_matrix
        if prototype_arrays:
            with archive.open("prototypes.npz", "w") as handle:
                np.savez(handle, **prototype_arrays)  # type: ignore[arg-type]


def load_compressed(path: str | Path) -> CompressedModel | CompactCompressedModel:
    """Load a compressed model from a .hwz archive."""
    archive_path = Path(path)
    with zipfile.ZipFile(archive_path, "r") as archive:
        metadata = json.loads(archive.read("metadata.json"))
        if metadata.get("mode") == "compact":
            compact_metadata, streams = load_compact_hwz(path)
            breakdown = {
                "metadata_bytes": len(json.dumps(compact_metadata, sort_keys=True).encode("utf-8")),
                "prototype_bytes": len(streams.get("prototypes", b"")),
                "assignment_bytes": len(streams.get("assignments", b"")),
                "scale_bytes": len(streams.get("scales", b""))
                + len(streams.get("zero_points", b"")),
                "residual_index_bytes": len(streams.get("residual_indices", b"")),
                "residual_value_bytes": len(streams.get("residual_values", b"")),
                "low_rank_bytes": len(streams.get("low_rank", b"")),
                "raw_value_bytes": len(streams.get("raw_values", b"")),
                "sparse_index_bytes": len(streams.get("sparse_indices", b"")),
                "sparse_value_bytes": len(streams.get("sparse_values", b"")),
                "archive_total_bytes": archive_path.stat().st_size,
            }
            return CompactCompressedModel(
                metadata=compact_metadata,
                streams=streams,
                archive_compression="zstd",
                payload_breakdown=breakdown,
            )
        prototype_arrays: Mapping[str, Any] = {}
        if "prototypes.npz" in archive.namelist():
            with archive.open("prototypes.npz") as handle:
                prototype_arrays = dict(np.load(handle))
        tensors: dict[str, CompressedTensor] = {}
        for name, value in metadata.get("tensors", {}).items():
            prototype_key = f"{name}_prototypes"
            tensors[name] = dict_to_tensor(value, prototype_arrays.get(prototype_key))
        return CompressedModel(
            tensors=tensors,
            payload=b"",
            format_version=metadata.get("format_version", "0.1"),
        )


def tensor_to_dict(
    tensor: CompressedTensor, include_prototype_matrix: bool = True
) -> dict[str, Any]:
    """Convert a compressed tensor to a JSON-safe dictionary."""
    payload: dict[str, Any] = {
        "name": tensor.name,
        "shape": list(tensor.shape),
        "block_size": tensor.block_size,
        "prototype_ids": tensor.prototype_ids,
        "scales": tensor.scales,
        "residuals": tensor.residuals,
        "seed": tensor.seed,
        "codec_config": tensor.codec_config,
    }
    if include_prototype_matrix:
        payload["prototype_matrix"] = tensor.prototype_matrix.tolist()
    return payload


def dict_to_tensor(
    payload: Mapping[str, Any], prototype_matrix: np.ndarray | None = None
) -> CompressedTensor:
    """Convert a JSON-safe dictionary back to a CompressedTensor."""
    matrix = prototype_matrix
    if matrix is None:
        matrix = np.asarray(payload.get("prototype_matrix", []), dtype=np.float32)
    return CompressedTensor(
        name=str(payload["name"]),
        shape=tuple(int(value) for value in payload["shape"]),
        block_size=int(payload["block_size"]),
        prototype_ids=[int(value) for value in payload["prototype_ids"]],
        scales=[float(value) for value in payload["scales"]],
        residuals=[dict(value) for value in payload["residuals"]],
        prototype_matrix=np.asarray(matrix, dtype=np.float32),
        seed=int(payload["seed"]),
        codec_config=dict(payload["codec_config"]),
    )
