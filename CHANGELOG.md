# Changelog

## 0.4.0

- Added learned compact codebook mode as the default compact tensor codec.
- Reduced GPT-style benchmark assignment bytes from 114,688 to 7,168 with uint4 assignment packing.
- Added assignment run-length encoding helpers and automatic assignment encoding selection.
- Added grouped assignment sharing configuration.
- Added compact benchmark rows for codebook and packed-int4 modes.
- Added v0.4 benchmark artifacts and payload breakdown reports.
- Added real open-model benchmark path through `--model`, using Hugging Face `transformers` when installed.

## 0.3.0

- Added compact codec mode with byte-packed binary streams.
- Added compact `.hwz` archive layout with binary assignments, scales, zero-points, and payload breakdown metadata.
- Added packed int4 tensor codec for quantization-class compression ratios.
- Added global prototype codebook helpers and packed assignment utilities.
- Added quantized prototype/int4/int8 helper functions.
- Added adaptive residual budget helpers with delta-varint index encoding.
- Added optional zstd stream compression through the `compression` extra.
- Added compact archive inspect payload breakdown.
- Added GPT-style benchmark script with markdown, JSON, and payload breakdown outputs.
- Added compact codec, archive, packing, quantization, global codebook, residual budget, and benchmark regression tests.

## 0.2.0

- Added int8 sparse residual quantization.
- Added prototype scale modes for per-block, per-tensor, and per-channel scaling.
- Added benchmark reports with FP32, FP16 estimate, INT8 estimate, and Hyper Glyph comparisons.
- Added markdown benchmark export through the Python API and CLI.
- Added an example compressed `.hwz` artifact and benchmark report.
- Improved `.hwz` serialization so prototype arrays are stored once in `prototypes.npz`.
- Preserved skipped tensors when restoring PyTorch state dicts with a reference state dict.

## 0.1.0

- Initial public release.
- Added NumPy compression path.
- Added optional PyTorch adapter.
- Added CLI and .hwz serialization.
