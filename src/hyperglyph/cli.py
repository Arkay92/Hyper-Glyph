"""Command-line interface for Hyper Glyph."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - optional dependency path
    torch = None

from .codec import HyperGlyphCodec
from .config import HyperGlyphConfig
from .serialization import load_compressed, save_compressed


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(prog="hyperglyph", description="Hyper Glyph compression CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compress_parser = subparsers.add_parser("compress")
    compress_parser.add_argument("input", help="Input torch state dict file (.pt)")
    compress_parser.add_argument("output", help="Output .hwz file")
    compress_parser.add_argument("--block-size", type=int, default=16)
    compress_parser.add_argument("--hdc-dim", type=int, default=4096)
    compress_parser.add_argument("--n-buckets", type=int, default=16)
    compress_parser.add_argument("--n-prototypes", type=int, default=128)
    compress_parser.add_argument("--residual-k", type=int, default=8)
    compress_parser.add_argument("--seed", type=int, default=42)
    compress_parser.add_argument("--compress-bias", action="store_true")
    compress_parser.add_argument("--min-tensor-size", type=int, default=256)

    decompress_parser = subparsers.add_parser("decompress")
    decompress_parser.add_argument("input", help="Input .hwz file")
    decompress_parser.add_argument("output", help="Output torch state dict file (.pt)")

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("input", help="Input .hwz file")

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("input", help="Input torch state dict file (.pt)")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "compress":
        if torch is None:
            raise SystemExit("PyTorch is required for the compress/decompress CLI commands")
        state_dict = torch.load(args.input, map_location="cpu")
        config = HyperGlyphConfig(
            hdc_dim=args.hdc_dim,
            block_size=args.block_size,
            n_buckets=args.n_buckets,
            n_prototypes=args.n_prototypes,
            residual_k=args.residual_k,
            seed=args.seed,
            compress_bias=args.compress_bias,
            min_tensor_size=args.min_tensor_size,
        )
        codec = HyperGlyphCodec(config)
        compressed = codec.compress_state_dict(state_dict)
        save_compressed(compressed, args.output)
        print(f"Saved compressed model to {args.output}")
        return 0

    if args.command == "decompress":
        if torch is None:
            raise SystemExit("PyTorch is required for the compress/decompress CLI commands")
        compressed = load_compressed(args.input)
        restored = HyperGlyphCodec().decompress_state_dict(compressed)
        torch.save(restored, args.output)
        print(f"Restored state dict to {args.output}")
        return 0

    if args.command == "inspect":
        compressed = load_compressed(args.input)
        print(
            json.dumps(
                {
                    "format_version": compressed.format_version,
                    "tensor_count": len(compressed.tensors),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "benchmark":
        if torch is None:
            raise SystemExit("PyTorch is required for benchmark CLI commands")
        state_dict = torch.load(args.input, map_location="cpu")
        codec = HyperGlyphCodec()
        compressed = codec.compress_state_dict(state_dict)
        restored = codec.decompress_state_dict(compressed)
        report = codec.report(compressed, state_dict, restored)
        print(report)
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
