import numpy as np

from hyperglyph import HyperGlyphCodec, HyperGlyphConfig

state_dict = {"weight": np.arange(256, dtype=np.float32).reshape(16, 16)}
config = HyperGlyphConfig(block_size=8, n_prototypes=8, residual_k=2)
codec = HyperGlyphCodec(config)
compressed = codec.compress_state_dict(state_dict)
restored = codec.decompress_state_dict(compressed)
print(restored["weight"].shape)
