import pytest

from hyperglyph import HyperGlyphConfig
from hyperglyph.torch_adapter import compress_state_dict, decompress_state_dict, is_torch_available

pytestmark = pytest.mark.skipif(not is_torch_available(), reason="torch is not installed")


def test_compress_and_decompress_simple_state_dict() -> None:
    import torch

    model = torch.nn.Linear(4, 3)
    config = HyperGlyphConfig(block_size=4, n_prototypes=4, residual_k=2, min_tensor_size=4)
    compressed = compress_state_dict(model.state_dict(), config=config)
    restored = decompress_state_dict(compressed, reference_state_dict=model.state_dict())

    assert set(restored) == set(model.state_dict())
    for name, tensor in model.state_dict().items():
        assert restored[name].shape == tensor.shape
