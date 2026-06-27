# Roadmap

## Version 0.1.0 - Proof of concept

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

## Version 0.2.0 - Residual quantization and reports

- per-block, per-tensor, and per-channel scale modes
- residual quantization int8
- markdown benchmark export
- baseline comparisons against FP32, FP16 estimate, and INT8 estimate
- example compressed .hwz artifact
- improved compression report

## Version 0.3.0 - Compact archive

- compact codec mode
- packed binary archive streams
- packed int4 tensor codec
- global prototype codebook helpers
- adaptive residual budget helpers
- GPT-style benchmark reports

## Version 0.4.0 - Codebook assignments

- learned compact codebook mode
- uint4 assignment packing
- RLE assignment encoding
- grouped/blockwise assignment sharing
- real open-model benchmark script path

## Future PyTorch integration

- compress_model(model)
- decompress_into_model(model, compressed)
- calibration pass
- layer include/exclude filters
- optional int4 residual packing
- entropy-coded residual indices
