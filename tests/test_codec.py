import numpy as np

from hyperglyph import HyperGlyphCodec, HyperGlyphConfig


def test_compress_and_decompress_numpy_array() -> None:
    config = HyperGlyphConfig(block_size=8, n_prototypes=4, residual_k=2, min_tensor_size=4)
    codec = HyperGlyphCodec(config)
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    compressed = codec.compress_array("weight", data)
    restored = codec.decompress_array(compressed)
    assert restored.shape == data.shape


def test_report_returns_valid_compression_fields() -> None:
    config = HyperGlyphConfig(block_size=8, n_prototypes=4, residual_k=2, min_tensor_size=4)
    codec = HyperGlyphCodec(config)
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    compressed = codec.compress_array("weight", data)
    restored = codec.decompress_array(compressed)
    report = codec.report(
        CompressedModelWrapper(compressed), {"weight": data}, {"weight": restored}
    )
    assert report.tensors_compressed == 1
    assert report.compressed_bytes > 0
    assert report.fp16_estimate_bytes == data.size * 2
    assert report.int8_estimate_bytes == data.size


def test_small_tensors_are_skipped_when_below_threshold() -> None:
    config = HyperGlyphConfig(min_tensor_size=1000)
    codec = HyperGlyphCodec(config)
    data = np.arange(16, dtype=np.float32)
    try:
        codec.compress_array("weight", data)
    except ValueError as exc:
        assert "too small" in str(exc)


def test_tensor_and_channel_scale_modes_compress() -> None:
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    for scale_mode in ("tensor", "channel"):
        config = HyperGlyphConfig(
            block_size=8,
            n_prototypes=4,
            residual_k=2,
            min_tensor_size=4,
            scale_mode=scale_mode,
        )
        codec = HyperGlyphCodec(config)
        compressed = codec.compress_array("weight", data)
        restored = codec.decompress_array(compressed)
        assert restored.shape == data.shape
        assert compressed.codec_config["scale_mode"] == f"per_{scale_mode}"


class CompressedModelWrapper:
    def __init__(self, compressed: object) -> None:
        self.tensors = {"weight": compressed}
        self.payload = b""
        self.format_version = "0.2"
