"""Dataset preparation from Ara's curated Nigerian knowledge.

No private messages, scraped accounts, or unlicensed social-media dumps are used.
Instruction pairs are derived from the documented corpus itself, so their
provenance is auditable and their limitations are the corpus's limitations.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ..nigeria import BUILTIN_DIR, KnowledgeBase


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def documents(kb: KnowledgeBase | None = None) -> list[dict[str, str]]:
    kb = kb or KnowledgeBase.load()
    docs: list[dict[str, str]] = []
    for entry in kb.of_kind("expressions"):
        examples = " ".join(f"{example.get('text', '')} -> {example.get('reading', '')}" for example in entry.examples)
        text = (
            f"Nigerian expression: {entry.term}\n"
            f"Meaning: {entry.summary}\n"
            f"Literal: {entry.literal}\n"
            f"Usage: {entry.usage}\n"
            f"Register: {entry.register}. Regions: {', '.join(entry.regions)}. Languages: {', '.join(entry.languages)}.\n"
            f"Examples: {examples}\n"
            f"Misreadings: {' '.join(entry.misreadings)}\n"
            f"Confidence: {entry.confidence:.2f}\n"
        )
        docs.append({"id": entry.id, "kind": "expression", "text": text})
    for entry in kb.of_kind("history"):
        text = (
            f"Nigerian history: {entry.title} ({entry.period})\n"
            f"{entry.summary}\nContext: {entry.context}\nCaution: {entry.caution}\n"
        )
        docs.append({"id": entry.id, "kind": "history", "text": text})
    for entry in kb.of_kind("culture"):
        text = (
            f"Nigerian everyday context: {entry.title}\n{entry.summary}\n"
            f"Details: {entry.context or ''}\nCaution: {entry.caution}\n"
        )
        docs.append({"id": entry.id, "kind": "culture", "text": text})
    seen: set[str] = set()
    unique = []
    for doc in docs:
        digest = _sha256(doc["text"])
        if digest in seen:
            continue
        seen.add(digest)
        doc["sha256"] = digest
        unique.append(doc)
    return unique


def instruction_pairs(kb: KnowledgeBase | None = None) -> list[dict[str, Any]]:
    kb = kb or KnowledgeBase.load()
    pairs: list[dict[str, Any]] = []
    for entry in kb.of_kind("expressions"):
        question = f"What does “{entry.term}” mean in Nigerian usage?"
        answer = (
            f"“{entry.term}” is {entry.register}. {entry.summary} "
            f"{('Literal note: ' + entry.literal + ' ') if entry.literal else ''}"
            f"Regions: {', '.join(entry.regions)}. Languages: {', '.join(entry.languages)}. "
            f"Confidence {entry.confidence:.2f}. {('Watch out: ' + ' '.join(entry.misreadings[:2])) if entry.misreadings else ''}"
        ).strip()
        pairs.append({
            "id": f"expr-{entry.id}",
            "messages": [
                {"role": "system", "content": "You are Ara, a Nigerian-context assistant. Cite uncertainty and avoid overgeneralising."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            "source": f"knowledge/expressions.json#{entry.id}",
            "license": "same as Ara repository",
        })
    for entry in kb.of_kind("history"):
        pairs.append({
            "id": f"history-{entry.id}",
            "messages": [
                {"role": "system", "content": "You are Ara, a Nigerian-context assistant. Separate documented facts from contested claims."},
                {"role": "user", "content": f"Explain {entry.title}."},
                {"role": "assistant", "content": f"{entry.summary} Context: {entry.context} Caution: {entry.caution}"},
            ],
            "source": f"knowledge/history.json#{entry.id}",
            "license": "same as Ara repository",
        })
    for entry in kb.of_kind("culture"):
        pairs.append({
            "id": f"culture-{entry.id}",
            "messages": [
                {"role": "system", "content": "You are Ara, a Nigerian-context assistant. Describe patterns without turning them into rules."},
                {"role": "user", "content": f"Tell me about {entry.title} in Nigeria."},
                {"role": "assistant", "content": f"{entry.summary} {entry.context} Caution: {entry.caution}"},
            ],
            "source": f"knowledge/culture.json#{entry.id}",
            "license": "same as Ara repository",
        })
    return pairs


def prepare(out_dir: str | Path, kb: KnowledgeBase | None = None, seed: int = 7) -> dict[str, Any]:
    """Write a deterministic train/validation split, chat JSONL, and manifest."""
    import random

    kb = kb or KnowledgeBase.load()
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    docs = documents(kb)
    if len(docs) < 4:
        raise ValueError("Not enough documents to prepare a split")
    ordered = sorted(docs, key=lambda doc: doc["id"])
    random.Random(seed).shuffle(ordered)
    split = max(1, int(len(ordered) * 0.1))
    validation, training = ordered[:split], ordered[split:]
    train_text = "\n\n".join(doc["text"] for doc in training) + "\n"
    val_text = "\n\n".join(doc["text"] for doc in validation) + "\n"
    (target / "corpus.txt").write_text("\n\n".join(doc["text"] for doc in ordered) + "\n", encoding="utf-8")
    (target / "train.txt").write_text(train_text, encoding="utf-8")
    (target / "val.txt").write_text(val_text, encoding="utf-8")
    pairs = instruction_pairs(kb)
    with (target / "chat.jsonl").open("w", encoding="utf-8") as stream:
        for pair in pairs:
            stream.write(json.dumps(pair, ensure_ascii=False) + "\n")
    manifest = {
        "version": 1,
        "source": "Ara curated Nigerian knowledge (expressions, history, culture)",
        "license": "Same licence as the Ara repository",
        "privacy": "No private messages, private accounts, restricted groups, or personal data.",
        "documents": len(docs),
        "train_documents": len(training),
        "validation_documents": len(validation),
        "instruction_pairs": len(pairs),
        "seed": seed,
        "sha256": {
            "corpus.txt": _sha256((target / "corpus.txt").read_text(encoding="utf-8")),
            "train.txt": _sha256(train_text),
            "val.txt": _sha256(val_text),
        },
        "counts": {"train_bytes": len(train_text.encode()), "val_bytes": len(val_text.encode())},
        "limitations": "Small curated corpus. It covers selected usage and history, not all Nigerian languages or communities.",
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def read_chat(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def corpus_stats(text: str) -> dict[str, int]:
    return {
        "characters": len(text),
        "words": len(re.findall(r"\S+", text)),
        "lines": len(text.splitlines()),
        "sha256": _sha256(text)[:16],
    }
