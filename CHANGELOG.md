# Changelog

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
