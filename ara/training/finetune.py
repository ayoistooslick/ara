"""Fine-tuning and export helpers.

No fine-tune is executed here. This module writes a reproducible dataset bundle,
checks which tools a machine actually has, and refuses to pretend a GPU, model, or
quantizer exists. Commands are templates; verify them against your installed
transformers/TRL or llama.cpp version.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def environment() -> dict[str, Any]:
    packages = {name: importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "datasets", "peft", "trl", "bitsandbytes")}
    binaries = {name: shutil.which(name) for name in ("llama-finetune", "llama-quantize", "llama-cli", "huggingface-cli")}
    try:
        import torch  # type: ignore

        cuda = bool(torch.cuda.is_available())
        device = torch.cuda.get_device_name(0) if cuda else "CPU"
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3 if cuda else None
    except Exception:
        cuda, device, vram = False, "unknown", None
    return {
        "python_packages": packages,
        "binaries": {name: path or "" for name, path in binaries.items()},
        "cuda": cuda,
        "device": device,
        "vram_gib": round(vram, 1) if vram else None,
        "can_finetune_with_transformers": all(packages[name] for name in ("torch", "transformers", "datasets", "peft", "trl")),
        "can_quantize_with_llama_cpp": bool(binaries["llama-quantize"]),
        "note": "Availability is detected, not assumed. Ara has not run a fine-tune or quantization in this environment.",
    }


def export_bundle(data_dir: str | Path, out_dir: str | Path, base_model: str = DEFAULT_BASE_MODEL) -> dict[str, Any]:
    source = Path(data_dir)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    chat = source / "chat.jsonl"
    if not chat.exists():
        raise ValueError("chat.jsonl is missing; prepare the dataset first")
    shutil.copy2(chat, target / "train_chat.jsonl")
    env = environment()
    plan = {
        "version": 1,
        "base_model": base_model,
        "dataset": "train_chat.jsonl",
        "format": "chat messages (system/user/assistant)",
        "status": "NOT RUN",
        "environment": env,
        "notes": [
            "Fine-tuning is not required for Ara to work; the app uses a pretrained model through the adapter.",
            "Do not describe this fine-tune as training Ara from scratch.",
            "Review the dataset licence and the base model's licence before distributing weights.",
            "The curated dataset is small; expect memorisation and limited generalisation.",
        ],
        "transformers_command_template": (
            "python -c \"from datasets import load_dataset; from trl import SFTTrainer; "
            "from transformers import AutoModelForCausalLM, AutoTokenizer; "
            "# Build SFTTrainer with the local train_chat.jsonl and your chosen hyperparameters. \""
        ),
        "llama_cpp_note": "Use the finetune/quantize tools shipped with your installed llama.cpp version; flags change between releases. Verify with --help first.",
    }
    (target / "finetune_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    (target / "README.md").write_text(
        "# Fine-tuning bundle\n\n"
        "This directory contains instruction data derived from Ara's curated Nigerian knowledge.\n"
        "No training has been run. "

        "`finetune_plan.json` records the detected environment and explicitly marks the run as NOT RUN.\n\n"
        "A small CPU baseline (tokenizer + counted n-gram) is produced by `python -m ara.training.train`.\n"
        "That baseline is not a chat model and must not be described as one.\n\n"
        "Before distributing any fine-tuned weights, record the base model, licence, data version, hyperparameters,\n"
        "evaluation results, and hardware in a model card. Do not claim capabilities you have not measured.\n",
        encoding="utf-8",
    )
    return {"out_dir": str(target), "environment": env, "base_model": base_model, "run": False}


def quantization_guidance(gguf_path: str | None = None) -> dict[str, Any]:
    env = environment()
    return {
        "available": env["can_quantize_with_llama_cpp"],
        "binary": env["binaries"].get("llama-quantize", ""),
        "command_template": f"llama-quantize {gguf_path or 'model-f16.gguf'} model-q4_k_m.gguf Q4_K_M",
        "note": "Quantization requires a compatible GGUF and llama.cpp build. It was not run by Ara.",
    }
