import torch

from hyperglyph import HyperGlyphCodec, HyperGlyphConfig

model = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 8))
config = HyperGlyphConfig(block_size=16, n_prototypes=16, residual_k=4)
codec = HyperGlyphCodec(config)
compressed = codec.compress_state_dict(model.state_dict())
restored = codec.decompress_state_dict(compressed)
print(f"Compressed tensors: {len(compressed.tensors)}")
print(f"Restored keys: {sorted(restored)}")
