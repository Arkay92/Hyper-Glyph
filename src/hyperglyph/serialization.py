"""Serialization helpers for compressed models."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .codec import CompressedModel, CompressedTensor


def save_compressed(compressed_model: CompressedModel, path: str | Path) -> None:
    """Save a compressed model to a .hwz zip archive."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        metadata = {
            "format_version": compressed_model.format_version,
            "tensors": {
                name: tensor_to_dict(tensor) for name, tensor in compressed_model.tensors.items()
            },
        }
        archive.writestr("metadata.json", json.dumps(metadata, indent=2))
        prototype_arrays = {}
        for name, tensor in compressed_model.tensors.items():
            prototype_arrays[f"{name}_prototypes"] = tensor.prototype_matrix
        if prototype_arrays:
            with archive.open("prototypes.npz", "w") as handle:
                np.savez(handle, **prototype_arrays)  # type: ignore[arg-type]


def load_compressed(path: str | Path) -> CompressedModel:
    """Load a compressed model from a .hwz archive."""
    archive_path = Path(path)
    with zipfile.ZipFile(archive_path, "r") as archive:
        metadata = json.loads(archive.read("metadata.json"))
        tensors: dict[str, CompressedTensor] = {}
        for name, value in metadata.get("tensors", {}).items():
            tensors[name] = dict_to_tensor(value)
        return CompressedModel(
            tensors=tensors, payload=b"", format_version=metadata.get("format_version", "0.1")
        )


def tensor_to_dict(tensor: CompressedTensor) -> dict[str, Any]:
    """Convert a compressed tensor to a JSON-safe dictionary."""
    return {
        "name": tensor.name,
        "shape": list(tensor.shape),
        "block_size": tensor.block_size,
        "prototype_ids": tensor.prototype_ids,
        "scales": tensor.scales,
        "residuals": tensor.residuals,
        "prototype_matrix": tensor.prototype_matrix.tolist(),
        "seed": tensor.seed,
        "codec_config": tensor.codec_config,
    }


def dict_to_tensor(payload: Mapping[str, Any]) -> CompressedTensor:
    """Convert a JSON-safe dictionary back to a CompressedTensor."""
    return CompressedTensor(
        name=str(payload["name"]),
        shape=tuple(int(value) for value in payload["shape"]),
        block_size=int(payload["block_size"]),
        prototype_ids=[int(value) for value in payload["prototype_ids"]],
        scales=[float(value) for value in payload["scales"]],
        residuals=[dict(value) for value in payload["residuals"]],
        prototype_matrix=np.asarray(payload["prototype_matrix"], dtype=np.float32),
        seed=int(payload["seed"]),
        codec_config=dict(payload["codec_config"]),
    )
