import numpy as np

from hyperglyph import HyperGlyphCodec, HyperGlyphConfig, benchmark_state_dict


def test_benchmark_report_exports_markdown_with_baselines() -> None:
    state_dict = {"weight": np.arange(256, dtype=np.float32).reshape(16, 16)}
    codec = HyperGlyphCodec(
        HyperGlyphConfig(block_size=8, n_prototypes=8, residual_k=2, min_tensor_size=4)
    )

    report = benchmark_state_dict(state_dict, codec)
    markdown = report.to_markdown()

    assert "FP32" in markdown
    assert "FP16 estimate" in markdown
    assert "INT8 estimate" in markdown
    assert "Hyper Glyph" in markdown
    assert report.compression.compressed_bytes > 0

