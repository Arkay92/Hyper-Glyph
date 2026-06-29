import numpy as np

from hyperglyph import HyperGlyphConfig
from hyperglyph.evaluation import (
    OPEN_MODEL_SUITE,
    ablation_configs,
    available_strong_quantization_libraries,
    run_ablation_study,
    tensor_error_analysis,
    tensor_error_markdown,
)


def test_tensor_error_analysis_sorts_by_mse() -> None:
    original = {
        "small_error": np.array([1.0, 2.0], dtype=np.float32),
        "large_error": np.array([1.0, 2.0], dtype=np.float32),
    }
    restored = {
        "small_error": np.array([1.0, 2.1], dtype=np.float32),
        "large_error": np.array([3.0, 2.0], dtype=np.float32),
    }

    rows = tensor_error_analysis(original, restored)
    markdown = tensor_error_markdown(rows, limit=1)

    assert rows[0].name == "large_error"
    assert "| large_error |" in markdown
    assert len(rows) == 2


def test_ablation_configs_cover_requested_dimensions() -> None:
    names = {name for name, _ in ablation_configs(HyperGlyphConfig(min_tensor_size=4))}

    assert {"residuals_off", "residuals_budget"}.issubset(names)
    assert {"codebook_8", "codebook_16", "codebook_32"}.issubset(names)
    assert {"block_8", "block_16", "block_32"}.issubset(names)
    assert {"scale_per_tensor", "scale_per_channel", "scale_per_block"}.issubset(names)


def test_run_ablation_study_can_run_selected_rows() -> None:
    state = {"weight": np.linspace(-1, 1, 64, dtype=np.float32).reshape(8, 8)}
    results = run_ablation_study(
        state,
        HyperGlyphConfig(min_tensor_size=4, auto_max_svd_elements=128),
        names={"residuals_off", "block_8"},
    )

    assert {result.name for result in results} == {"residuals_off", "block_8"}
    assert all(result.compressed_bytes > 0 for result in results)


def test_stronger_quantization_library_detection_is_structured() -> None:
    statuses = available_strong_quantization_libraries()

    assert {status.name for status in statuses} >= {"GPTQ", "AWQ", "SmoothQuant"}
    assert all(isinstance(status.available, bool) for status in statuses)


def test_open_model_suite_includes_requested_models() -> None:
    assert "distilgpt2" in OPEN_MODEL_SUITE
    assert "EleutherAI/gpt-neo-125m" in OPEN_MODEL_SUITE
    assert "facebook/opt-125m" in OPEN_MODEL_SUITE
    assert "EleutherAI/pythia-70m" in OPEN_MODEL_SUITE
    assert "TinyLlama/TinyLlama-1.1B-Chat-v1.0" in OPEN_MODEL_SUITE
