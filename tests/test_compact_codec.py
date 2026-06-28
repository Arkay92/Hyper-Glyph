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
        HyperGlyphConfig(
            mode="compact",
            scale_mode="per_channel",
            min_tensor_size=4,
            compact_tensor_codec="packed_int4",
        )
    )
    compressed = codec.compress_state_dict(state)
    restored = codec.decompress_state_dict(compressed)
    report = codec.report(compressed, state, restored)
    assert np.isfinite(report.total_mse)
    assert report.total_mse < 1e-3


def test_codebook_reduces_assignment_bytes_vs_packed_int4() -> None:
    state = _state_dict()
    int4_codec = CompactHyperGlyphCodec(
        HyperGlyphConfig(mode="compact", compact_tensor_codec="packed_int4", min_tensor_size=4)
    )
    codebook_codec = CompactHyperGlyphCodec(
        HyperGlyphConfig(
            mode="compact",
            compact_tensor_codec="codebook",
            n_global_prototypes=16,
            block_size=16,
            min_tensor_size=4,
        )
    )
    int4_model = int4_codec.compress_state_dict(state)
    codebook_model = codebook_codec.compress_state_dict(state)

    assert (
        codebook_model.payload_breakdown["assignment_bytes"]
        < int4_model.payload_breakdown["assignment_bytes"]
    )
    assert codebook_model.metadata["config"]["compact_tensor_codec"] == "codebook"


def test_grouped_assignment_sharing_stores_fewer_assignments() -> None:
    state = _state_dict()
    ungrouped_codec = CompactHyperGlyphCodec(
        HyperGlyphConfig(
            mode="compact",
            compact_tensor_codec="codebook",
            assignment_group_size=1,
            min_tensor_size=4,
        )
    )
    grouped_codec = CompactHyperGlyphCodec(
        HyperGlyphConfig(
            mode="compact",
            compact_tensor_codec="codebook",
            assignment_group_size=4,
            min_tensor_size=4,
        )
    )
    ungrouped = ungrouped_codec.compress_state_dict(state)
    grouped = grouped_codec.compress_state_dict(state)
    ungrouped_count = sum(tensor["assignment_count"] for tensor in ungrouped.metadata["tensors"])
    grouped_count = sum(tensor["assignment_count"] for tensor in grouped.metadata["tensors"])
    assert grouped_count < ungrouped_count


def test_low_rank_codec_roundtrips_low_rank_matrix() -> None:
    rng = np.random.default_rng(7)
    left = rng.normal(size=(32, 4)).astype(np.float32)
    right = rng.normal(size=(4, 32)).astype(np.float32)
    state = {"proj.weight": left @ right}
    codec = CompactHyperGlyphCodec(
        HyperGlyphConfig(
            mode="compact",
            compact_tensor_codec="low_rank",
            low_rank_values=(4,),
            min_tensor_size=4,
        )
    )
    compressed = codec.compress_state_dict(state)
    restored = codec.decompress_state_dict(compressed)
    assert compressed.metadata["tensors"][0]["codec"].startswith("low_rank")
    assert restored["proj.weight"].shape == state["proj.weight"].shape


def test_auto_mode_keeps_packed_tensor_values_out_of_assignment_bytes() -> None:
    state = _state_dict()
    codec = CompactHyperGlyphCodec(
        HyperGlyphConfig(mode="compact", compact_tensor_codec="auto", min_tensor_size=4)
    )
    compressed = codec.compress_state_dict(state)
    assert compressed.payload_breakdown["raw_value_bytes"] > 0
    assert (
        compressed.payload_breakdown["assignment_bytes"]
        < compressed.payload_breakdown["raw_value_bytes"]
    )
