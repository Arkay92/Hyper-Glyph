import numpy as np

from hyperglyph import CompactHyperGlyphCodec, HyperGlyphConfig
from hyperglyph.serialization import save_compressed


def _state_dict() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(123)
    return {
        "attn.weight": rng.normal(0, 0.2, size=(64, 64)).astype(np.float32),
        "mlp.weight": rng.normal(0, 0.2, size=(64, 128)).astype(np.float32),
    }


def test_compact_compress_decompress_state_dict() -> None:
    codec = CompactHyperGlyphCodec(
        HyperGlyphConfig(mode="compact", scale_mode="per_channel", min_tensor_size=4)
    )
    compressed = codec.compress_state_dict(_state_dict())
    restored = codec.decompress_state_dict(compressed)
    assert set(restored) == {"attn.weight", "mlp.weight"}
    assert restored["attn.weight"].shape == (64, 64)


def test_actual_hwz_archive_smaller_than_fp32(tmp_path) -> None:
    state = _state_dict()
    codec = CompactHyperGlyphCodec(
        HyperGlyphConfig(mode="compact", scale_mode="per_channel", min_tensor_size=4)
    )
    compressed = codec.compress_state_dict(state)
    path = tmp_path / "model.hwz"
    save_compressed(compressed, path)
    fp32_bytes = sum(array.nbytes for array in state.values())
    assert path.stat().st_size < fp32_bytes


def test_compact_gets_better_than_4x_on_synthetic(tmp_path) -> None:
    state = _state_dict()
    codec = CompactHyperGlyphCodec(
        HyperGlyphConfig(mode="compact", scale_mode="per_tensor", min_tensor_size=4)
    )
    compressed = codec.compress_state_dict(state)
    path = tmp_path / "model.hwz"
    save_compressed(compressed, path)
    ratio = sum(array.nbytes for array in state.values()) / path.stat().st_size
    assert ratio >= 4.0


def test_compact_mse_is_finite_and_reasonable() -> None:
    state = _state_dict()
    codec = CompactHyperGlyphCodec(
        HyperGlyphConfig(mode="compact", scale_mode="per_channel", min_tensor_size=4)
    )
    compressed = codec.compress_state_dict(state)
    restored = codec.decompress_state_dict(compressed)
    report = codec.report(compressed, state, restored)
    assert np.isfinite(report.total_mse)
    assert report.total_mse < 1e-3
