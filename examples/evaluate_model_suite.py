from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from hyperglyph import HyperGlyphConfig
from hyperglyph.evaluation import (
    OPEN_MODEL_SUITE,
    available_strong_quantization_libraries,
    evaluate_perplexity,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=list(OPEN_MODEL_SUITE))
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--decompressed", action="store_true")
    parser.add_argument("--output", default="model_suite_eval.json")
    args = parser.parse_args()

    config = None
    if args.decompressed:
        config = HyperGlyphConfig(mode="compact", compact_tensor_codec="auto")

    rows = []
    for model_name in args.models:
        result = evaluate_perplexity(
            model_name=model_name,
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            split=args.split,
            max_samples=args.max_samples,
            sequence_length=args.sequence_length,
            compression_config=config,
        )
        rows.append(asdict(result))

    payload = {
        "models": rows,
        "strong_quantization_libraries": [
            asdict(status) for status in available_strong_quantization_libraries()
        ],
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
