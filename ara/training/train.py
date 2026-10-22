from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .corpus import prepare
from .model import NgramError, evaluate_checkpoint, train_model
from .tokenizer import TokenizerError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ara.training.train", description="Train Ara's reproducible CPU teaching baseline.")
    parser.add_argument("--data", help="Prepared dataset directory (created with prepare if missing)")
    parser.add_argument("--out", default="runs/baseline", help="Run output directory")
    parser.add_argument("--vocab-size", type=int, default=320)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--resume", action="store_true", help="Resume from a valid checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate an existing run")
    parser.add_argument("--export-finetune", action="store_true", help="Write the fine-tuning bundle without running it")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out).expanduser().resolve()
    data = Path(args.data).expanduser().resolve() if args.data else out / "data"
    try:
        if args.export_finetune:
            from .finetune import export_bundle

            if not (data / "chat.jsonl").exists():
                manifest = prepare(data)
                print(json.dumps({"prepared": manifest}, indent=2))
            report = export_bundle(data, out / "finetune", args.base_model)
            print(json.dumps(report, indent=2))
            return 0
        if args.eval_only:
            report = evaluate_checkpoint(out / "model.json", out / "tokenizer.json", data / "val.txt")
            print(json.dumps(report, indent=2))
            return 0
        if not (data / "train.txt").exists():
            manifest = prepare(data)
            print(json.dumps({"prepared": manifest}, indent=2), file=sys.stderr)
        report = train_model(data, out, vocab_size=args.vocab_size, order=args.order, resume=args.resume)
        print(json.dumps(report, indent=2))
        return 0
    except (ValueError, NgramError, TokenizerError, OSError) as exc:
        print(f"training: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
