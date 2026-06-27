# Roadmap

## Version 0.1.0 — Proof of concept

- src layout package
- HyperGlyphConfig
- HDC role vectors
- Block splitting
- Prototype learning
- Sparse residual repair
- .hwz serialization
- CLI
- tests
- CI
- PyPI publish workflow

## Version 0.2.0 — Better compression fidelity

- per-channel scale support
- residual quantization int8
- optional int4 residual packing
- entropy-coded residual indices
- improved compression report

## Version 0.3.0 — PyTorch integration

- compress_model(model)
- decompress_into_model(model, compressed)
- calibration pass
- layer include/exclude filters
