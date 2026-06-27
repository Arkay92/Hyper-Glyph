# CLI

## compress

Compress a PyTorch state dict into a .hwz archive.

```bash
hyperglyph compress model.pt model.hwz
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
