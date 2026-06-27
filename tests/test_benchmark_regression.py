import warnings

import numpy as np

from hyperglyph import CompactHyperGlyphCodec, HyperGlyphConfig
from hyperglyph.metrics import mse
from hyperglyph.quantization import dequantize_int4_packed, quantize_int4_packed
from hyperglyph.serialization import save_compressed


def _synthetic_gpt_state() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    state: dict[str, np.ndarray] = {
        "wte.weight": rng.normal(0, 0.02, size=(512, 64)).astype(np.float32)
    }
    for layer in range(4):
        state[f"layers.{layer}.attn.qkv.weight"] = rng.normal(0, 0.02, size=(64, 192)).astype(
            np.float32
        )
        state[f"layers.{layer}.attn.proj.weight"] = rng.normal(0, 0.02, size=(64, 64)).astype(
            np.float32
        )
        state[f"layers.{layer}.mlp.up.weight"] = rng.normal(0, 0.02, size=(64, 256)).astype(
            np.float32
        )
        state[f"layers.{layer}.mlp.down.weight"] = rng.normal(0, 0.02, size=(256, 64)).astype(
            np.float32
        )
        state[f"layers.{layer}.ln.bias"] = np.zeros(64, dtype=np.float32)
    return state


def test_compact_synthetic_gpt_regression(tmp_path) -> None:
    state = _synthetic_gpt_state()
    fp32_bytes = sum(array.nbytes for array in state.values())
    int4_restored = {
        name: dequantize_int4_packed(quantize_int4_packed(array))
        for name, array in state.items()
        if "bias" not in name
    }
    int4_mse = sum(mse(state[name], restored) for name, restored in int4_restored.items())

    codec = CompactHyperGlyphCodec(
        HyperGlyphConfig(mode="compact", scale_mode="per_tensor", min_tensor_size=256)
    )
    compressed = codec.compress_state_dict(state)
    path = tmp_path / "synthetic-gpt.hwz"
    save_compressed(compressed, path)
    restored = codec.decompress_state_dict(compressed)
    compact_mse = sum(mse(state[name], restored[name]) for name in restored)
    compact_ratio = fp32_bytes / path.stat().st_size

    assert compact_ratio >= 4.0
    if compact_ratio < 6.0:
        warnings.warn(f"compact ratio below stretch target: {compact_ratio:.2f}x", stacklevel=1)
    assert compact_mse <= 2.0 * int4_mse
