# API

## HyperGlyphConfig

Configuration for the codec.

## HyperGlyphCodec

- compress_array(name, array)
- decompress_array(compressed)
- compress_state_dict(state_dict)
- decompress_state_dict(compressed_model)
- report(compressed_model, original_state_dict, restored_state_dict)

## Serialization helpers

- save_compressed(compressed_model, path)
- load_compressed(path)
