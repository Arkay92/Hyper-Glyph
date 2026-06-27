import zipfile

import numpy as np

from hyperglyph import HyperGlyphCodec, HyperGlyphConfig
from hyperglyph.serialization import load_compressed, save_compressed


def test_save_and_load_hwz(tmp_path) -> None:
    config = HyperGlyphConfig(block_size=8, n_prototypes=4, residual_k=2, min_tensor_size=4)
    codec = HyperGlyphCodec(config)
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    compressed = codec.compress_array("weight", data)
    path = tmp_path / "model.hwz"
    save_compressed(compressed_model_from_tensor(compressed), path)
    loaded = load_compressed(path)

    assert loaded.format_version == "0.2"
    assert "weight" in loaded.tensors
    assert loaded.tensors["weight"].name == "weight"


def test_metadata_format_version_exists(tmp_path) -> None:
    config = HyperGlyphConfig(block_size=8, n_prototypes=4, residual_k=2, min_tensor_size=4)
    codec = HyperGlyphCodec(config)
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    compressed = codec.compress_array("weight", data)
    path = tmp_path / "model.hwz"
    save_compressed(compressed_model_from_tensor(compressed), path)
    with zipfile.ZipFile(path) as archive:
        metadata = archive.read("metadata.json")
        assert b'"format_version"' in metadata


def compressed_model_from_tensor(compressed):
    from hyperglyph.codec import CompressedModel

    return CompressedModel(tensors={"weight": compressed}, payload=b"", format_version="0.2")
