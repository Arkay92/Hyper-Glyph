import zipfile

import numpy as np

from hyperglyph import CompactHyperGlyphCodec, HyperGlyphConfig
from hyperglyph.archive import read_binary_stream, write_binary_stream
from hyperglyph.serialization import load_compressed, save_compressed


def test_save_load_compact_hwz(tmp_path) -> None:
    state = {"weight": np.arange(256, dtype=np.float32).reshape(16, 16)}
    codec = CompactHyperGlyphCodec(HyperGlyphConfig(mode="compact", min_tensor_size=4))
    compressed = codec.compress_state_dict(state)
    path = tmp_path / "model.hwz"
    save_compressed(compressed, path)
    loaded = load_compressed(path)
    restored = codec.decompress_state_dict(loaded)
    assert restored["weight"].shape == state["weight"].shape


def test_fallback_stream_roundtrip() -> None:
    data = b"abc" * 100
    encoded, compression = write_binary_stream(data, use_zstd=False)
    assert compression == "raw"
    assert read_binary_stream(encoded, compression) == data


def test_metadata_has_format_version_and_compact_mode(tmp_path) -> None:
    state = {"weight": np.arange(256, dtype=np.float32).reshape(16, 16)}
    codec = CompactHyperGlyphCodec(HyperGlyphConfig(mode="compact", min_tensor_size=4))
    compressed = codec.compress_state_dict(state)
    path = tmp_path / "model.hwz"
    save_compressed(compressed, path)
    with zipfile.ZipFile(path) as archive:
        metadata = archive.read("metadata.json")
    assert b'"format_version": "0.4"' in metadata
    assert b'"mode": "compact"' in metadata
