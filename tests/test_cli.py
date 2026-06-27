import numpy as np

from hyperglyph import HyperGlyphCodec, HyperGlyphConfig
from hyperglyph.cli import main
from hyperglyph.codec import CompressedModel
from hyperglyph.serialization import save_compressed


def test_inspect_command_exits_successfully(tmp_path) -> None:
    config = HyperGlyphConfig(block_size=8, n_prototypes=4, residual_k=2, min_tensor_size=4)
    codec = HyperGlyphCodec(config)
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    compressed = codec.compress_array("weight", data)

    path = tmp_path / "model.hwz"
    save_compressed(
        CompressedModel(tensors={"weight": compressed}, payload=b"", format_version="0.2"),
        path,
    )

    exit_code = main(["inspect", str(path)])
    assert exit_code == 0
