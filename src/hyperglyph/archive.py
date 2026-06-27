"""Compact .hwz archive helpers."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any


def _zstd_module() -> Any | None:
    try:
        import zstandard as zstd  # type: ignore
    except ImportError:
        return None
    return zstd


def write_binary_stream(data: bytes, use_zstd: bool = True, level: int = 9) -> tuple[bytes, str]:
    """Optionally compress a binary stream with zstd."""
    zstd = _zstd_module() if use_zstd else None
    if zstd is None:
        return data, "raw"
    return zstd.ZstdCompressor(level=level).compress(data), "zstd"


def read_binary_stream(data: bytes, compression: str) -> bytes:
    """Read a binary stream compressed by write_binary_stream."""
    if compression == "raw":
        return data
    if compression == "zstd":
        zstd = _zstd_module()
        if zstd is None:
            raise RuntimeError("zstandard is required to read this compact archive")
        return zstd.ZstdDecompressor().decompress(data)
    raise ValueError(f"unknown stream compression: {compression}")


def save_compact_hwz(model: Any, path: str | Path) -> None:
    """Save a compact compressed model."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    use_zstd = model.archive_compression == "zstd"
    metadata = dict(model.metadata)
    metadata["streams"] = {}
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in model.streams.items():
            encoded, compression = write_binary_stream(data, use_zstd, model.zstd_level)
            suffix = ".zst" if compression == "zstd" else ""
            filename = f"{name}.bin{suffix}"
            metadata["streams"][name] = {
                "file": filename,
                "compression": compression,
                "raw_bytes": len(data),
                "stored_bytes": len(encoded),
            }
            archive.writestr(filename, encoded)
        model.metadata = metadata
        archive.writestr("metadata.json", json.dumps(metadata, indent=2, sort_keys=True))


def load_compact_hwz(path: str | Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Load compact archive metadata and streams."""
    archive_path = Path(path)
    with zipfile.ZipFile(archive_path, "r") as archive:
        metadata = json.loads(archive.read("metadata.json"))
        streams: dict[str, bytes] = {}
        for name, info in metadata.get("streams", {}).items():
            encoded = archive.read(info["file"])
            streams[name] = read_binary_stream(encoded, info["compression"])
    return metadata, streams
