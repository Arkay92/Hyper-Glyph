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

## Notes

The codec is intended for research and experimentation rather than guaranteed production compression.
