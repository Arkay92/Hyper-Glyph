# CLI

## compress

Compress a PyTorch state dict into a .hwz archive.

```bash
hyperglyph compress model.pt model.hwz \
  --residual-dtype int8 \
  --scale-mode per_channel
```

Compact mode:

```bash
hyperglyph compress model.pt model.hwz --mode compact --target-ratio 8
hyperglyph inspect model.hwz
```

## decompress

Restore a compressed archive back to a PyTorch state dict.

```bash
hyperglyph decompress model.hwz restored.pt
```

## inspect

Print the metadata for a compressed archive.

```bash
hyperglyph inspect model.hwz
```

## benchmark

Benchmark compression and reconstruction for a state dict.

```bash
hyperglyph benchmark model.pt
```

Write a markdown benchmark report with FP32, FP16 estimate, INT8 estimate, and
Hyper Glyph comparisons:

```bash
hyperglyph benchmark model.pt --markdown-output benchmark.md
```

Benchmark compact mode:

```bash
hyperglyph benchmark model.pt --mode compact
```
