# Hyper Glyph

Hyper Glyph is an experimental package for compressing neural network weights with symbolic hyperdimensional prototypes, configurable prototype scales, and sparse residual repair.

## Installation

```bash
pip install hyperglyph
```

## Quickstart

```python
from hyperglyph import HyperGlyphCodec, HyperGlyphConfig

config = HyperGlyphConfig(block_size=16, n_prototypes=32, residual_k=4)
codec = HyperGlyphCodec(config)
```

v0.2 adds int8 residual quantization, per-block/per-tensor/per-channel scale
modes, markdown benchmark reports, and baseline comparisons against FP32, FP16
estimate, and INT8 estimate sizes.

v0.3 adds compact mode with byte-packed binary streams, packed int4 tensor
storage, payload breakdown reports, and a GPT-style benchmark script that
measures actual `.hwz` archive size.

v0.4 makes learned codebook mode the default compact tensor codec, adds uint4
assignment packing, RLE assignment streams, grouped assignment sharing, and
separate benchmark rows for codebook and packed-int4 compact modes.

v0.5 makes compact mode a per-tensor codec portfolio. It evaluates packed-int4,
codebook, low-rank, low-rank plus residual, sparse, and raw-int8 candidates, then
selects the best byte/error tradeoff.

## Notes

The codec is intended for research and experimentation rather than guaranteed production compression.
