# Hyper Glyph

<p align="center">
  Hyperdimensional symbolic residual compression for neural network weights.
</p>

<p align="center">
  <img width="256" height="256" alt="Hyper Glyph Logo" src="https://github.com/Arkay92/Hyper-Glyph/raw/main/hyperglyph.png" />
</p>

<p align="center">
  <a href="https://github.com/Arkay92/Hyper-Glyph/actions/workflows/publish.yml"><img alt="Publish" src="https://github.com/Arkay92/Hyper-Glyph/actions/workflows/publish.yml/badge.svg" /></a>
  <a href="https://pypi.org/project/hyperglyph-codec/"><img alt="PyPI" src="https://img.shields.io/pypi/v/hyperglyph-codec.svg" /></a>
  <img alt="Python" src="https://img.shields.io/pypi/pyversions/hyperglyph-codec.svg" />
  <img alt="License" src="https://img.shields.io/pypi/l/hyperglyph-codec.svg" />
  <img alt="Downloads" src="https://img.shields.io/pypi/dm/hyperglyph-codec.svg" />
</p>

> **Package name:** `hyperglyph-codec`  
> **Project name:** Hyper Glyph  
> **Install:** `pip install hyperglyph-codec`  
> **Import:** `from hyperglyph import HyperGlyphCodec`

**Hyper Glyph** combines:

- **Block-level tensor compression** for NumPy arrays and neural network weights.
- **Symbolic prototype assignment** to represent repeated weight patterns compactly.
- **Sparse residual repair** to preserve reconstruction fidelity after prototype decoding.
- **Int8 residual quantization** to reduce sparse repair payload size.
- **Per-block, per-tensor, and per-channel prototype scales** for tuning reconstruction behavior.
- **Configurable compression controls** for block size, prototype count, residual size, and tensor filtering.
- **State dict compression** for model-like parameter dictionaries.
- **Optional PyTorch support** for loading, compressing, restoring, and benchmarking `.pt` state dicts.
- **`.hwz` serialization** for saving compressed models as portable archives.
- **Compression reports** with size ratio, tensor counts, and reconstruction error metrics.
- **Markdown benchmark export** with FP32, FP16, INT1 through INT8, and Hyper Glyph comparisons.
- **A small CLI** for compressing, decompressing, inspecting, and benchmarking model archives.
- **Typed Python API** designed for research, experimentation, and extension.

---

## Before / After

```python
import numpy as np

from hyperglyph import HyperGlyphCodec, HyperGlyphConfig

state_dict = {
    "layer.weight": np.arange(1024, dtype=np.float32).reshape(32, 32),
}

config = HyperGlyphConfig(
    block_size=16,
    n_prototypes=32,
    residual_k=4,
)

codec = HyperGlyphCodec(config)
compressed = codec.compress_state_dict(state_dict)
restored = codec.decompress_state_dict(compressed)
report = codec.report(compressed, state_dict, restored)

print(report)
```

Example output:

```text
CompressionReport(
    original_bytes=4096,
    compressed_bytes=...,
    compression_ratio=...,
    tensors_compressed=1,
    tensors_skipped=0,
    total_mse=...,
    total_mae=...,
    max_abs_error=...
)
```

That is the core job: encode large weight tensors as reusable symbolic
prototypes plus a small residual correction, then report the size and
reconstruction tradeoff.

Hyper Glyph v0.2.0 is an experimental research codec. It is intended for
testing ideas around hyperdimensional and symbolic weight compression rather
than guaranteed production compression.

Hyper Glyph v0.3.0 adds **compact mode**, a byte-packed archive path focused on
actual stored bytes rather than theoretical payload estimates. Hyper Glyph
v0.6.0 extends the benchmark harness across scalar quantization bit widths from
1 through 8 bits.

Sample v0.2.0 benchmark from `examples/artifacts/sample-v0.2-benchmark.md`:

| Representation | Bytes | Ratio vs FP32 | MSE | MAE | Max abs error |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 24576 | 1.00x | 0 | 0 | 0 |
| FP16 estimate | 12288 | 2.00x | - | - | - |
| INT8 estimate | 6144 | 4.00x | - | - | - |
| Hyper Glyph | 22032 | 1.12x | 0.00266153 | 0.0405458 | 0.197096 |

The matching compressed artifact is `examples/artifacts/sample-v0.2.hwz`
and is 26,318 bytes on disk in the current zip-based archive format.

### Hyper Glyph Compact Mode

Compact mode moves large payloads out of JSON and into binary streams inside the
`.hwz` archive. v0.6.0 keeps `compact_tensor_codec="auto"` as the default, so
each tensor can choose the best candidate for its byte/error tradeoff. It
includes:

- Global codebook and packed assignment utilities for symbolic prototype experiments.
- Packed affine scalar quantization baselines for INT1 through INT8.
- Uint4 assignment packing when the prototype count is 16 or lower.
- Run-length encoding for repeated assignment patterns.
- Grouped/blockwise assignment sharing through `assignment_group_size`.
- Packed int4 tensor storage when it beats prototype mode on byte/error tradeoffs.
- Low-rank int8 and low-rank plus sparse residual candidates.
- Sparse tensor and raw int8 candidates for small or structured tensors.
- Block-codebook plus sparse residual candidates.
- Float16 or float32 scale metadata, with per-tensor and per-channel scale modes.
- Adaptive sparse residual budget helpers with delta-varint index encoding.
- Optional zstd compression for binary streams via `hyperglyph-codec[compression]`.
- Payload breakdown reporting for metadata, assignments, scales, residuals, and archive size.

Measured on the deterministic synthetic GPT-style benchmark in
`examples/artifacts/v0.6/benchmark_report.md`:

| Method | Bytes | Ratio | MSE | Cosine |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 918528 | 1.00x | 0 | 1.00000004 |
| FP16 quantization | 459264 | 2.00x | 2.9432538e-10 | 0.99999992 |
| INT1 quantization | 28872 | 31.81x | 0.073967011 | 0.81135406 |
| INT2 quantization | 57576 | 15.95x | 0.0044553082 | 0.84468936 |
| INT3 quantization | 86280 | 10.65x | 0.00077001884 | 0.95792332 |
| INT4 quantization | 114984 | 7.99x | 0.00016693089 | 0.99027589 |
| INT5 quantization | 143688 | 6.39x | 3.9213638e-05 | 0.99768398 |
| INT6 quantization | 172392 | 5.33x | 9.4796408e-06 | 0.99943803 |
| INT7 quantization | 201096 | 4.57x | 2.3378236e-06 | 0.99986133 |
| INT8 quantization | 229800 | 4.00x | 5.783329e-07 | 0.99996566 |
| Hyper Glyph standard | 759603 | 1.21x | 0.00028803079 | 0.97888187 |
| Hyper Glyph compact codebook | 32226 | 28.50x | 0.0052734507 | 0.47573550 |
| Hyper Glyph compact auto | 123776 | 7.42x | 6.6366149e-05 | 0.99609527 |
| Hyper Glyph compact packed-int4 | 123155 | 7.46x | 6.6366149e-05 | 0.99517650 |

Payload breakdown for `Hyper Glyph compact auto`:

| Payload | Bytes |
| --- | ---: |
| metadata_bytes | 6896 |
| prototype_bytes | 0 |
| assignment_bytes | 0 |
| scale_bytes | 18464 |
| residual_index_bytes | 0 |
| residual_value_bytes | 0 |
| low_rank_bytes | 0 |
| raw_value_bytes | 114944 |
| sparse_index_bytes | 0 |
| sparse_value_bytes | 0 |
| archive_total_bytes | 123776 |

On this benchmark, codebook mode massively reduces archive size but has poor
error. Auto mode selects the quality-oriented packed-int4 candidate for these
random GPT-style tensors, beats plain INT4 MSE, and reports packed value bytes
separately from true codebook assignment bytes.

### Beyond Scalar Quantization

Standard INT4 and INT8 quantization round weights independently. The stronger
compression literature usually combines vector quantization, pruning, and
lossless coding to push practical storage toward the 1- to 2-bit regime without
accepting the error profile of naive rounding.

- **Extreme vector quantization:** learned dictionaries encode groups of weights
  as shared vectors or codewords instead of independent scalars. AQLM uses
  additive learned codebooks to keep language-model weight compression
  competitive below 3 bits per parameter. QuIP# uses randomized Hadamard mixing
  and high-dimensional lattice codebooks such as E8 for strong results at
  4 bits per weight and below.
- **Pruning and sparsity:** pruning removes redundant weights rather than only
  lowering their precision. Semi-structured 2:4 sparsity stores two non-zero
  weights per group of four and maps to supported neural cores. SparseGPT uses
  second-order approximations to remove unstructured weights while compensating
  for the dropped connections.
- **Entropy and lossless encoding:** quantized weights and assignments are not
  uniformly distributed, so Huffman, ANS, range coding, or zstd-style compression
  can reduce archive bytes further. EntroLLM-style entropy coding targets
  frequent patterns with shorter codes. Delta methods such as BitDelta and
  SVD-based deltas store only fine-tuning differences from a base model, often
  in 1- or 2-bit form.

Hyper Glyph v0.6.0 does not claim full AQLM, QuIP#, SparseGPT, or EntroLLM
implementations. It implements the practical building blocks needed to compare
against them: learned block codebooks, sparse residual repair, low-rank plus
residual candidates, grouped assignments, run-length and delta-varint streams,
optional zstd archive compression, and benchmark rows across every scalar bit
width from INT1 through INT8.

---

## Why Hyperdimensional Weight Compression?

Neural network weights often contain repeated local structure, approximate
patterns, and redundancy that can be represented more compactly than raw
floating-point tensors:

```text
Weight tensors
  -> Split into fixed-size blocks
  -> Learn reusable prototype blocks
  -> Assign each block to a prototype
  -> Store per-block scales
  -> Store sparse top-k residual corrections as int8 or float32

  -> Save compressed archive
  -> Restore approximate tensors
  -> Report compression and reconstruction metrics
```

`Hyper Glyph` is designed for experiments where you want to inspect that
tradeoff directly:

- **Large weight matrices** that can be split into repeated local blocks.
- **Prototype-based compression** where blocks share learned representatives.
- **Sparse residual repair** where only the largest reconstruction corrections are stored.
- **Scale modes** for per-block, per-tensor, or per-channel prototype scaling.
- **Approximate reconstruction** with measurable MSE, MAE, and max absolute error.
- **State dict workflows** that match common PyTorch model storage patterns.
- **Portable archive output** for saving and inspecting compressed runs.

---

## Why Not Just Quantization?

Quantization changes the numeric precision of individual weights. Hyper Glyph
uses a different representation: each tensor block is mapped to a learned
prototype, scaled, and repaired with a sparse residual.

That makes Hyper Glyph useful for experimenting with symbolic and
hyperdimensional compression ideas, not as a drop-in replacement for mature
quantization, pruning, or production model compression toolchains.

---

## Architecture

```text
Input weights
  - NumPy arrays
  - PyTorch state_dict values
    |
    v
Compression
  - tensor filtering
  - block splitting
  - prototype learning
  - prototype assignment
  - scale calculation
  - int8 or float32 sparse residual encoding
    |
    v
CompressedModel
  - compressed tensors
  - shapes
  - prototype ids
  - scales
  - residuals
  - prototype matrices
  - codec metadata
    |
    v
Decompression and report
  - reconstructed tensors
  - original/compressed byte estimates
  - compression ratio
  - MSE / MAE / max error
```

---

## Install

```bash
pip install hyperglyph-codec
```

For PyTorch state dict support:

```bash
pip install "hyperglyph-codec[torch]"
```

For compact archives with optional zstd compression:

```bash
pip install "hyperglyph-codec[torch,compression]"
```

For documentation dependencies:

```bash
pip install "hyperglyph-codec[docs]"
```

For development:

```bash
pip install -e ".[dev,torch,docs]"
pytest
python -m build
```

---

## Quick Start

### Compress a NumPy State Dict

```python
import numpy as np

from hyperglyph import HyperGlyphCodec, HyperGlyphConfig

state_dict = {
    "weight": np.arange(256, dtype=np.float32).reshape(16, 16),
}

config = HyperGlyphConfig(
    block_size=8,
    n_prototypes=8,
    residual_k=2,
)

codec = HyperGlyphCodec(config)
compressed = codec.compress_state_dict(state_dict)
restored = codec.decompress_state_dict(compressed)

print(restored["weight"].shape)
```

### Compress a PyTorch Model

```python
import torch

from hyperglyph import HyperGlyphCodec, HyperGlyphConfig

model = torch.nn.Sequential(
    torch.nn.Linear(32, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 8),
)

config = HyperGlyphConfig(
    block_size=16,
    n_prototypes=16,
    residual_k=4,
    residual_dtype="int8",
    scale_mode="block",
)

codec = HyperGlyphCodec(config)
compressed = codec.compress_state_dict(model.state_dict())
restored = codec.decompress_state_dict(compressed)

print(f"Compressed tensors: {len(compressed.tensors)}")
print(f"Restored keys: {sorted(restored)}")
```

### Save and Load `.hwz` Archives

```python
from hyperglyph import load_compressed, save_compressed

save_compressed(compressed, "model.hwz")
loaded = load_compressed("model.hwz")
restored = codec.decompress_state_dict(loaded)
```

### Generate a Report

```python
report = codec.report(
    compressed_model=compressed,
    original_state_dict=state_dict,
    restored_state_dict=restored,
)

print(report.compression_ratio)
print(report.total_mse)
print(report.max_abs_error)
```

---

## CLI

Compress a PyTorch state dict into a `.hwz` archive:

```bash
hyperglyph compress model.pt model.hwz
```

Use compact mode and a target ratio:

```bash
hyperglyph compress gpt2.pt gpt2.hwz --mode compact --target-ratio 8
hyperglyph inspect gpt2.hwz
```

Tune compression settings:

```bash
hyperglyph compress model.pt model.hwz \
  --block-size 16 \
  --hdc-dim 4096 \
  --n-buckets 16 \
  --n-prototypes 128 \
  --residual-k 8 \
  --residual-dtype int8 \
  --scale-mode per_channel \
  --min-tensor-size 256
```

Restore a compressed archive back to a PyTorch state dict:

```bash
hyperglyph decompress model.hwz restored.pt
```

Inspect archive metadata:

```bash
hyperglyph inspect model.hwz
```

Benchmark compression and reconstruction:

```bash
hyperglyph benchmark model.pt
```

Benchmark compact mode:

```bash
hyperglyph benchmark model.pt --mode compact
```

Export the benchmark as markdown:

```bash
hyperglyph benchmark model.pt --markdown-output benchmark.md
```

---

## Benchmark Example

A small practical benchmark is enough to see the current codec behavior:

```bash
hyperglyph benchmark model.pt
```

Example markdown output:

```text
| Representation | Bytes | Ratio vs FP32 | MSE | MAE | Max abs error |
| FP32 | 24576 | 1.00x | 0 | 0 | 0 |
| FP16 estimate | 12288 | 2.00x | - | - | - |
| INT8 estimate | 6144 | 4.00x | - | - | - |
| Hyper Glyph | 22032 | 1.12x | 0.00266153 | 0.0405458 | 0.197096 |
```

The current package focuses on transparent compression experiments rather than
claiming universal size reductions. Compare the restored model against your own
accuracy, latency, and reconstruction thresholds.

---

## Main Features

### 1. **Configurable Codec**

Tune the compression shape with one dataclass:

```python
from hyperglyph import HyperGlyphConfig

config = HyperGlyphConfig(
    hdc_dim=4096,
    block_size=16,
    n_buckets=16,
    n_prototypes=128,
    residual_k=8,
    residual_dtype="int8",
    scale_mode="channel",
    seed=42,
    min_tensor_size=256,
    compress_bias=False,
)
```

### 2. **Array Compression**

Compress and reconstruct a single NumPy array:

```python
import numpy as np

from hyperglyph import HyperGlyphCodec

codec = HyperGlyphCodec()
array = np.random.randn(64, 64).astype("float32")

compressed = codec.compress_array("layer.weight", array)
restored = codec.decompress_array(compressed)
```

### 3. **State Dict Compression**

Compress dictionary-style model weights:

```python
compressed = codec.compress_state_dict(state_dict)
restored = codec.decompress_state_dict(compressed)
```

By default, small tensors and bias tensors are skipped. Set `min_tensor_size`
and `compress_bias` in `HyperGlyphConfig` to change that behavior.

### 4. **Sparse Residual Repair**

Each block is reconstructed from a prototype and scale, then corrected with a
top-k sparse residual:

```text
block ~= prototype[prototype_id] * scale + sparse_residual
```

Increase `residual_k` for better reconstruction fidelity, or reduce it for a
smaller compressed representation.

Set `residual_dtype="int8"` to quantize sparse residual values. Use
`residual_dtype="float32"` when you want unquantized residual repairs.

### 5. **Serialization**

Save compressed models as `.hwz` zip archives:

```python
from hyperglyph import load_compressed, save_compressed

save_compressed(compressed, "model.hwz")
loaded = load_compressed("model.hwz")
```

### 6. **PyTorch Adapter**

Install the `torch` extra to convert PyTorch tensors into compressed Hyper Glyph
models:

```python
from hyperglyph import compress_state_dict, decompress_state_dict

compressed = compress_state_dict(model.state_dict())
restored = decompress_state_dict(compressed, reference_state_dict=model.state_dict())
```

### 7. **Compression Metrics**

Measure the compression and reconstruction tradeoff:

```python
report = codec.report(compressed, state_dict, restored)

print(report.original_bytes)
print(report.compressed_bytes)
print(report.compression_ratio)
print(report.total_mse)
print(report.total_mae)
print(report.max_abs_error)
```

---

## Configuration

```python
from hyperglyph import HyperGlyphConfig

config = HyperGlyphConfig(
    hdc_dim=4096,
    block_size=16,
    n_buckets=16,
    n_prototypes=128,
    residual_k=8,
    residual_dtype="int8",
    scale_mode="block",
    seed=42,
    min_tensor_size=256,
    compress_bias=False,
    dtype="float32",
    device="cpu",
)
```

Key settings:

- **`block_size`** controls how many flattened weights are grouped together.
- **`n_prototypes`** controls how many reusable block representatives are learned.
- **`residual_k`** controls how many residual correction values are stored per block.
- **`residual_dtype`** controls whether sparse residual values are stored as `int8` or `float32`.
- **`scale_mode`** controls whether prototype scales are calculated per `block`, per `tensor`, or per `channel`.
- **`min_tensor_size`** skips tensors too small to benefit from compression.
- **`compress_bias`** enables compression for bias tensors, which are skipped by default.
- **`seed`** makes prototype selection deterministic.

---

## Examples

```python
import numpy as np

from hyperglyph import HyperGlyphCodec, HyperGlyphConfig

state_dict = {
    "encoder.weight": np.random.randn(128, 128).astype("float32"),
    "decoder.weight": np.random.randn(128, 64).astype("float32"),
}

codec = HyperGlyphCodec(
    HyperGlyphConfig(
        block_size=16,
        n_prototypes=64,
        residual_k=8,
        residual_dtype="int8",
        scale_mode="channel",
    )
)

compressed = codec.compress_state_dict(state_dict)
restored = codec.decompress_state_dict(compressed)
report = codec.report(compressed, state_dict, restored)

print(report)
```

```bash
hyperglyph compress model.pt model.hwz
hyperglyph inspect model.hwz
hyperglyph benchmark model.pt
```

---

## Project Structure

```text
src/hyperglyph/
  __init__.py             # Public API
  archive.py              # Compact .hwz binary stream archive helpers
  benchmark.py            # Benchmark report helpers
  blocks.py               # Tensor flattening, block splitting, shape restore
  cli.py                  # Command-line interface
  codec.py                # HyperGlyphCodec and compressed dataclasses
  compact_codec.py        # CompactHyperGlyphCodec
  config.py               # HyperGlyphConfig
  exceptions.py           # Package exceptions
  global_codebook.py      # Global prototype collection and assignment helpers
  hdc.py                  # Hyperdimensional vector helpers
  metrics.py              # Size and reconstruction metrics
  packing.py              # uint4/int4/varint/delta packing helpers
  prototypes.py           # Prototype learning and assignment
  quantization.py         # int8/int4 quantization helpers
  residual.py             # Sparse residual encoding and repair
  residual_budget.py      # Adaptive residual budget helpers
  serialization.py        # .hwz save/load helpers
  torch_adapter.py        # Optional PyTorch integration
  py.typed                # Typing marker
tests/
  test_*.py               # Unit tests
docs/
  algorithm.md            # Algorithm overview
  api.md                  # API notes
  cli.md                  # CLI examples
  index.md                # Documentation home
  roadmap.md              # Planned work
examples/
  compress_mlp.py         # PyTorch MLP compression example
  compress_state_dict.py  # NumPy state dict compression example
  mnist_demo.py           # MNIST-oriented demo
  benchmark_hyperglyph_vs_quant.py # FP32/FP16/INT1-INT8/Hyper Glyph benchmark
  artifacts/
    sample-v0.2.hwz       # Example compressed archive
    sample-v0.2-benchmark.md # Markdown benchmark report
hyperglyph.png            # Project logo
pyproject.toml            # Package metadata and dependencies
CHANGELOG.md              # Release history
CONTRIBUTING.md           # Contribution guide
LICENSE                   # MIT license
```

---

## Development

```bash
# Install with dev, PyTorch, and docs extras
pip install -e ".[dev,torch,docs]"

# Run tests
pytest

# Run linting
ruff check .

# Type-check package code
mypy

# Build package
python -m build
```

---

## License

MIT

---

## Contributing

Contributions are welcome. Open an issue or pull request with the model shape,
codec configuration, expected compression behavior, reconstruction metrics, and
any accuracy checks you used.

---

## Citation

If you use Hyper Glyph in research, please cite:

```bibtex
@software{HyperGlyph2026,
  title={Hyper Glyph: Hyperdimensional Symbolic Residual Compression for Neural Network Weights},
  author={Robert McMenemy},
  year={2026},
  version={0.6.0},
}
```
