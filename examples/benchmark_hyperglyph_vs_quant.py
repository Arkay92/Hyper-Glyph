from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from hyperglyph import CompactHyperGlyphCodec, HyperGlyphCodec, HyperGlyphConfig
from hyperglyph.metrics import cosine_weight_similarity, mae, max_abs_error, mse
from hyperglyph.quantization import (
    dequantize_int4_packed,
    dequantize_int8,
    quantize_int4_packed,
    quantize_int8,
)
from hyperglyph.serialization import save_compressed


def synthetic_gpt_state() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    state: dict[str, np.ndarray] = {
        "wte.weight": rng.normal(0, 0.02, size=(512, 64)).astype(np.float32)
    }
    for layer in range(4):
        state[f"layers.{layer}.attn.qkv.weight"] = rng.normal(0, 0.02, size=(64, 192)).astype(
            np.float32
        )
        state[f"layers.{layer}.attn.proj.weight"] = rng.normal(0, 0.02, size=(64, 64)).astype(
            np.float32
        )
        state[f"layers.{layer}.mlp.up.weight"] = rng.normal(0, 0.02, size=(64, 256)).astype(
            np.float32
        )
        state[f"layers.{layer}.mlp.down.weight"] = rng.normal(0, 0.02, size=(256, 64)).astype(
            np.float32
        )
        state[f"layers.{layer}.ln.bias"] = np.zeros(64, dtype=np.float32)
    return state


def load_hf_state(model_name: str) -> dict[str, np.ndarray]:
    try:
        from transformers import AutoModelForCausalLM  # type: ignore
    except ImportError as exc:
        raise SystemExit("transformers is required for --model benchmarks") from exc
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return {name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()}


def row(
    method: str,
    size_bytes: int,
    fp32_bytes: int,
    original: dict[str, np.ndarray],
    restored: dict[str, np.ndarray],
    seconds: float,
) -> dict[str, Any]:
    names = [name for name in restored if name in original]
    return {
        "method": method,
        "bytes": size_bytes,
        "ratio": fp32_bytes / size_bytes if size_bytes else float("inf"),
        "mse": sum(mse(original[name], restored[name]) for name in names),
        "mae": sum(mae(original[name], restored[name]) for name in names),
        "max_abs_error": max(max_abs_error(original[name], restored[name]) for name in names),
        "cosine": float(
            np.mean([cosine_weight_similarity(original[name], restored[name]) for name in names])
        ),
        "seconds": seconds,
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Method | Bytes | Ratio vs FP32 | MSE | MAE | Max Abs Error | Cosine | Seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in rows:
        lines.append(
            f"| {item['method']} | {item['bytes']} | {item['ratio']:.2f}x | "
            f"{item['mse']:.8g} | {item['mae']:.8g} | {item['max_abs_error']:.8g} | "
            f"{item['cosine']:.8f} | {item['seconds']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Hugging Face causal LM name")
    parser.add_argument("--synthetic-gpt", action="store_true")
    parser.add_argument("--target-ratio", type=float, default=6.0)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    state = synthetic_gpt_state()
    if args.model and not args.synthetic_gpt:
        state = load_hf_state(args.model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fp32_bytes = sum(array.nbytes for array in state.values())
    rows: list[dict[str, Any]] = []
    rows.append(row("FP32", fp32_bytes, fp32_bytes, state, state, 0.0))

    start = time.perf_counter()
    fp16 = {name: array.astype(np.float16).astype(np.float32) for name, array in state.items()}
    rows.append(
        row(
            "FP16 quantization",
            sum(array.size * 2 for array in state.values()),
            fp32_bytes,
            state,
            fp16,
            time.perf_counter() - start,
        )
    )

    start = time.perf_counter()
    int8 = {name: dequantize_int8(quantize_int8(array)) for name, array in state.items()}
    rows.append(
        row(
            "INT8 quantization",
            sum(array.size for array in state.values()),
            fp32_bytes,
            state,
            int8,
            time.perf_counter() - start,
        )
    )

    start = time.perf_counter()
    int4 = {
        name: dequantize_int4_packed(quantize_int4_packed(array)) for name, array in state.items()
    }
    rows.append(
        row(
            "INT4 quantization",
            sum((array.size + 1) // 2 + 4 for array in state.values()),
            fp32_bytes,
            state,
            int4,
            time.perf_counter() - start,
        )
    )

    start = time.perf_counter()
    standard = HyperGlyphCodec(HyperGlyphConfig(mode="standard", min_tensor_size=256))
    standard_model = standard.compress_state_dict(state)
    standard_restored = standard.decompress_state_dict(standard_model)
    standard_path = output_dir / "benchmark_standard.hwz"
    save_compressed(standard_model, standard_path)
    rows.append(
        row(
            "Hyper Glyph standard",
            standard_path.stat().st_size,
            fp32_bytes,
            state,
            standard_restored,
            time.perf_counter() - start,
        )
    )

    start = time.perf_counter()
    compact = CompactHyperGlyphCodec(
        HyperGlyphConfig(
            mode="compact",
            scale_mode="per_channel",
            scale_dtype="float32",
            target_ratio=args.target_ratio,
            min_tensor_size=256,
        )
    )
    compact_model = compact.compress_state_dict(state)
    compact_restored = compact.decompress_state_dict(compact_model)
    compact_path = output_dir / "benchmark_compact.hwz"
    save_compressed(compact_model, compact_path)
    rows.append(
        row(
            "Hyper Glyph compact",
            compact_path.stat().st_size,
            fp32_bytes,
            state,
            compact_restored,
            time.perf_counter() - start,
        )
    )

    report_md = markdown_table(rows)
    (output_dir / "benchmark_report.md").write_text(report_md, encoding="utf-8")
    (output_dir / "benchmark_report.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    breakdown = compact_model.payload_breakdown
    breakdown_md = "\n".join(
        ["| Payload | Bytes |", "| --- | ---: |"]
        + [f"| {key} | {value} |" for key, value in breakdown.items()]
    )
    (output_dir / "benchmark_payload_breakdown.md").write_text(
        breakdown_md + "\n", encoding="utf-8"
    )
    print(report_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
