# API

## HyperGlyphConfig

Configuration for the codec.

- `residual_dtype`: `int8` or `float32` sparse residual storage.
- `scale_mode`: `block`, `tensor`, or `channel` prototype scaling.

## HyperGlyphCodec

- compress_array(name, array)
- decompress_array(compressed)
- compress_state_dict(state_dict)
- decompress_state_dict(compressed_model)
- report(compressed_model, original_state_dict, restored_state_dict)

## Benchmark helpers

- benchmark_state_dict(state_dict, codec=None)
- BenchmarkReport.to_markdown()

## Serialization helpers

- save_compressed(compressed_model, path)
- load_compressed(path)
