# Hyper Glyph

Hyper Glyph is an experimental package for compressing neural network weights with symbolic hyperdimensional prototypes and sparse residual repair.

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

## Notes

The codec is intended for research and experimentation rather than guaranteed production compression.
