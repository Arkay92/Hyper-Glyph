# API

## HyperGlyphConfig

Configuration for the codec.

- `residual_dtype`: `int8` or `float32` sparse residual storage.
- `scale_mode`: `per_block`, `per_tensor`, or `per_channel` scaling.
- `mode`: `standard` or `compact`.

## HyperGlyphCodec

- compress_array(name, array)
- decompress_array(compressed)
- compress_state_dict(state_dict)
- decompress_state_dict(compressed_model)
- report(compressed_model, original_state_dict, restored_state_dict)

## Benchmark helpers

- benchmark_state_dict(state_dict, codec=None)
- BenchmarkReport.to_markdown()

## Compact codec

- CompactHyperGlyphCodec
- compact binary `.hwz` streams
- payload breakdown metadata

## Serialization helpers

- save_compressed(compressed_model, path)
- load_compressed(path)
