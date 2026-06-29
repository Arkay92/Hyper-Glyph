"""Evaluation helpers for model compression experiments."""

from __future__ import annotations

import copy
import importlib.util
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .codec import HyperGlyphCodec
from .config import HyperGlyphConfig
from .metrics import cosine_weight_similarity, mae, max_abs_error, mse

OPEN_MODEL_SUITE: tuple[str, ...] = (
    "distilgpt2",
    "EleutherAI/gpt-neo-125m",
    "facebook/opt-125m",
    "EleutherAI/pythia-70m",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
)

STRONG_QUANTIZATION_LIBRARIES: tuple[tuple[str, str], ...] = (
    ("GPTQ", "auto_gptq"),
    ("AWQ", "awq"),
    ("SmoothQuant", "smoothquant"),
    ("bitsandbytes", "bitsandbytes"),
    ("TorchAO", "torchao"),
    ("Optimum Quanto", "optimum.quanto"),
)


@dataclass(slots=True)
class TensorError:
    """Tensor-wise reconstruction error row."""

    name: str
    shape: tuple[int, ...]
    fp32_bytes: int
    mse: float
    mae: float
    max_abs_error: float
    cosine: float


@dataclass(slots=True)
class QuantizationLibraryStatus:
    """Availability of an external advanced quantization baseline."""

    name: str
    module: str
    available: bool


@dataclass(slots=True)
class AblationResult:
    """One ablation benchmark row."""

    name: str
    compressed_bytes: int
    ratio_vs_fp32: float
    mse: float
    mae: float
    max_abs_error: float
    tensors_compressed: int


@dataclass(slots=True)
class InferenceComparison:
    """Logit-level comparison after loading restored weights."""

    mse: float
    mae: float
    max_abs_error: float


@dataclass(slots=True)
class PerplexityResult:
    """Language-model perplexity result."""

    model_name: str
    dataset_name: str
    dataset_config: str | None
    split: str
    samples: int
    tokens: int
    fp32_perplexity: float
    decompressed_perplexity: float | None = None

    @property
    def perplexity_delta(self) -> float | None:
        if self.decompressed_perplexity is None:
            return None
        return self.decompressed_perplexity - self.fp32_perplexity


def tensor_error_analysis(
    original: Mapping[str, Any], restored: Mapping[str, Any]
) -> list[TensorError]:
    """Return layer/tensor-wise reconstruction errors sorted by MSE descending."""
    rows: list[TensorError] = []
    for name, original_value in original.items():
        if name not in restored:
            continue
        original_array = np.asarray(original_value, dtype=np.float32)
        restored_array = np.asarray(restored[name], dtype=np.float32)
        rows.append(
            TensorError(
                name=name,
                shape=tuple(original_array.shape),
                fp32_bytes=original_array.nbytes,
                mse=mse(original_array, restored_array),
                mae=mae(original_array, restored_array),
                max_abs_error=max_abs_error(original_array, restored_array),
                cosine=cosine_weight_similarity(original_array, restored_array),
            )
        )
    return sorted(rows, key=lambda row: row.mse, reverse=True)


def tensor_error_markdown(rows: list[TensorError], limit: int | None = None) -> str:
    """Render tensor-wise reconstruction errors as markdown."""
    selected = rows[:limit] if limit is not None else rows
    lines = [
        "| Tensor | Shape | FP32 Bytes | MSE | MAE | Max Abs Error | Cosine |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in selected:
        lines.append(
            f"| {row.name} | {list(row.shape)} | {row.fp32_bytes} | "
            f"{row.mse:.8g} | {row.mae:.8g} | {row.max_abs_error:.8g} | {row.cosine:.8f} |"
        )
    return "\n".join(lines) + "\n"


def available_strong_quantization_libraries() -> list[QuantizationLibraryStatus]:
    """Detect optional GPTQ/AWQ/SmoothQuant-style comparison libraries."""
    return [
        QuantizationLibraryStatus(
            name=name,
            module=module,
            available=_module_available(module),
        )
        for name, module in STRONG_QUANTIZATION_LIBRARIES
    ]


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def ablation_configs(base: HyperGlyphConfig | None = None) -> list[tuple[str, HyperGlyphConfig]]:
    """Build a compact ablation grid for residuals, codebooks, blocks, and scales."""
    source = base or HyperGlyphConfig(mode="compact")

    def make_config(
        *,
        residual_mode: str = source.residual_mode,
        n_global_prototypes: int = source.n_global_prototypes,
        block_size: int = source.block_size,
        scale_mode: str = source.scale_mode,
    ) -> HyperGlyphConfig:
        return HyperGlyphConfig(
            mode=source.mode,
            min_tensor_size=source.min_tensor_size,
            target_ratio=source.target_ratio,
            compact_tensor_codec=source.compact_tensor_codec,
            seed=source.seed,
            residual_mode=residual_mode,
            n_global_prototypes=n_global_prototypes,
            block_size=block_size,
            scale_mode=scale_mode,
            auto_max_codebook_blocks=source.auto_max_codebook_blocks,
            auto_max_svd_elements=source.auto_max_svd_elements,
        )

    return [
        ("residuals_off", make_config(residual_mode="none")),
        ("residuals_budget", make_config(residual_mode="budget")),
        ("codebook_8", make_config(n_global_prototypes=8)),
        ("codebook_16", make_config(n_global_prototypes=16)),
        ("codebook_32", make_config(n_global_prototypes=32)),
        ("block_8", make_config(block_size=8)),
        ("block_16", make_config(block_size=16)),
        ("block_32", make_config(block_size=32)),
        ("scale_per_tensor", make_config(scale_mode="per_tensor")),
        ("scale_per_channel", make_config(scale_mode="per_channel")),
        ("scale_per_block", make_config(scale_mode="per_block")),
    ]


def run_ablation_study(
    state_dict: Mapping[str, Any],
    base: HyperGlyphConfig | None = None,
    names: set[str] | None = None,
) -> list[AblationResult]:
    """Compress a state dict across the default ablation grid."""
    results: list[AblationResult] = []
    for name, config in ablation_configs(base):
        if names is not None and name not in names:
            continue
        codec = HyperGlyphCodec(config)
        compressed = codec.compress_state_dict(state_dict)
        restored = codec.decompress_state_dict(compressed)
        report = codec.report(compressed, state_dict, restored)
        results.append(
            AblationResult(
                name=name,
                compressed_bytes=report.compressed_bytes,
                ratio_vs_fp32=report.compression_ratio,
                mse=report.total_mse,
                mae=report.total_mae,
                max_abs_error=report.max_abs_error,
                tensors_compressed=report.tensors_compressed,
            )
        )
    return results


def compare_inference_after_decompression(
    model: Any,
    restored_state_dict: Mapping[str, Any],
    input_ids: Any,
) -> InferenceComparison:
    """Compare model logits before and after loading decompressed weights."""
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("PyTorch is required for inference comparison") from exc

    restored_model = copy.deepcopy(model)
    converted = {
        name: torch.as_tensor(value, dtype=param.dtype, device=param.device)
        for name, param in restored_model.state_dict().items()
        if name in restored_state_dict
        for value in [restored_state_dict[name]]
    }
    restored_model.load_state_dict(converted, strict=False)
    model.eval()
    restored_model.eval()
    with torch.no_grad():
        original_logits = _model_logits(model(input_ids))
        restored_logits = _model_logits(restored_model(input_ids))
    delta = (original_logits - restored_logits).detach().cpu().float().numpy()
    return InferenceComparison(
        mse=float(np.mean(delta**2)),
        mae=float(np.mean(np.abs(delta))),
        max_abs_error=float(np.max(np.abs(delta))) if delta.size else 0.0,
    )


def evaluate_perplexity(
    model_name: str,
    dataset_name: str = "wikitext",
    dataset_config: str | None = "wikitext-2-raw-v1",
    split: str = "test",
    max_samples: int = 64,
    sequence_length: int = 512,
    compression_config: HyperGlyphConfig | None = None,
) -> PerplexityResult:
    """Evaluate FP32 and optional decompressed perplexity on WikiText-2/C4 subsets."""
    try:
        import torch  # type: ignore
        from datasets import load_dataset  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "Perplexity evaluation requires torch, transformers, and datasets"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    texts = _dataset_texts(dataset, max_samples)
    fp32_nll, tokens = _negative_log_likelihood(model, tokenizer, texts, sequence_length, torch)
    decompressed_perplexity: float | None = None
    if compression_config is not None:
        state = {name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()}
        codec = HyperGlyphCodec(compression_config)
        restored_state = codec.decompress_state_dict(codec.compress_state_dict(state))
        restored_model = copy.deepcopy(model)
        restored_model.load_state_dict(
            {
                name: torch.as_tensor(value, dtype=param.dtype)
                for name, param in restored_model.state_dict().items()
                if name in restored_state
                for value in [restored_state[name]]
            },
            strict=False,
        )
        restored_nll, _ = _negative_log_likelihood(
            restored_model, tokenizer, texts, sequence_length, torch
        )
        decompressed_perplexity = math.exp(restored_nll)
    return PerplexityResult(
        model_name=model_name,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        samples=len(texts),
        tokens=tokens,
        fp32_perplexity=math.exp(fp32_nll),
        decompressed_perplexity=decompressed_perplexity,
    )


def _dataset_texts(dataset: Any, max_samples: int) -> list[str]:
    texts: list[str] = []
    for item in dataset:
        text = str(item.get("text") or item.get("content") or "")
        if text.strip():
            texts.append(text)
        if len(texts) >= max_samples:
            break
    return texts


def _negative_log_likelihood(
    model: Any, tokenizer: Any, texts: list[str], sequence_length: int, torch: Any
) -> tuple[float, int]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=sequence_length,
            )
            input_ids = encoded["input_ids"]
            if input_ids.numel() < 2:
                continue
            outputs = model(input_ids, labels=input_ids)
            token_count = int(input_ids.numel())
            total_loss += float(outputs.loss) * token_count
            total_tokens += token_count
    if total_tokens == 0:
        raise ValueError("no tokens available for perplexity evaluation")
    return total_loss / total_tokens, total_tokens


def _model_logits(output: Any) -> Any:
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, tuple):
        return output[0]
    return output
