"""Optional PyTorch adapter."""

from __future__ import annotations

from typing import Any, Mapping

from .config import HyperGlyphConfig
from .exceptions import OptionalDependencyError


def is_torch_available() -> bool:
    """Check whether torch is installed."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def tensor_to_numpy(tensor: Any) -> Any:
    """Convert a torch tensor to a NumPy array."""
    if not is_torch_available():
        raise OptionalDependencyError("torch is required for tensor_to_numpy")
    import torch

    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def numpy_to_tensor(array: Any, reference_tensor: Any | None = None) -> Any:
    """Convert a NumPy array back to a torch tensor."""
    if not is_torch_available():
        raise OptionalDependencyError("torch is required for numpy_to_tensor")
    import torch

    if reference_tensor is not None and isinstance(reference_tensor, torch.Tensor):
        return torch.tensor(array, dtype=reference_tensor.dtype, device=reference_tensor.device)
    return torch.tensor(array)


def compress_state_dict(
    state_dict: Mapping[str, Any], config: HyperGlyphConfig | None = None
) -> Any:
    """Compress a torch state_dict."""
    if not is_torch_available():
        raise OptionalDependencyError("torch is required for compress_state_dict")
    from .codec import HyperGlyphCodec

    codec = HyperGlyphCodec(config)
    numpy_state_dict = {name: tensor_to_numpy(value) for name, value in state_dict.items()}
    return codec.compress_state_dict(numpy_state_dict)


def decompress_state_dict(
    compressed_model: Any, reference_state_dict: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Decompress a compressed model into a dictionary of numpy arrays or torch tensors."""
    if not is_torch_available():
        raise OptionalDependencyError("torch is required for decompress_state_dict")
    from .codec import HyperGlyphCodec

    codec = HyperGlyphCodec()
    restored = codec.decompress_state_dict(compressed_model)
    if reference_state_dict is None:
        return restored
    merged: dict[str, Any] = dict(reference_state_dict)
    for name, value in restored.items():
        merged[name] = numpy_to_tensor(value, reference_state_dict[name])
    return merged
