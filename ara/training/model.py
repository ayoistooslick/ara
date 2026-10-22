"""Counted n-gram language-model baseline.

This is real training from initialization over the supplied corpus, but it is a
small statistical baseline - not a transformer and not Ara's chat model. The model
card and metadata say so explicitly.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .tokenizer import ByteBPETokenizer, TokenizerError

BOS = -1
EOS = -2


class NgramError(RuntimeError):
    pass


class CountedNgramModel:
    def __init__(self, order: int = 3, alpha: float = 0.1):
        if not 1 <= order <= 5:
            raise ValueError("order must be between 1 and 5")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.order = order
        self.alpha = alpha
        self.unigrams: Counter[int] = Counter()
        self.ngrams: dict[int, dict[tuple[int, ...], Counter[int]]] = {n: defaultdict(Counter) for n in range(2, order + 1)}
        self.documents_seen = 0
        self.tokens_seen = 0
        self.vocab_size = 256

    # ------------------------------------------------------------------ training
    def observe(self, tokens: list[int]) -> None:
        if not tokens:
            return
        padded = [BOS] * (self.order - 1) + tokens + [EOS]
        for index, token in enumerate(padded):
            if token >= 0:
                self.unigrams[token] += 1
                self.tokens_seen += 1
            for n in range(2, self.order + 1):
                if index + 1 < len(padded):
                    context = tuple(padded[max(0, index - n + 2):index + 1])
                    if len(context) == n - 1:
                        self.ngrams[n][context][padded[index + 1]] += 1
        self.documents_seen += 1

    def train(self, documents: Iterable[list[int]], checkpoint: Any = None, every: int = 20) -> None:
        for index, tokens in enumerate(documents, 1):
            self.observe(tokens)
            if checkpoint and index % every == 0:
                checkpoint(self)
        if checkpoint:
            checkpoint(self)

    # ------------------------------------------------------------------ scoring
    def _prob(self, context: tuple[int, ...], token: int, vocab_size: int) -> float:
        context = context[-(self.order - 1):] if self.order > 1 else ()
        # Interpolate the available orders, weighting longer context more heavily.
        weighted = 0.0
        total_weight = 0.0
        for n in range(1, self.order + 1):
            weight = n
            total_weight += weight
            if n == 1:
                counts = self.unigrams
                denominator = sum(self.unigrams.values()) + self.alpha * vocab_size
                weighted += weight * ((counts.get(token, 0) + self.alpha) / denominator if denominator else 0.0)
            else:
                table = self.ngrams[n]
                key = context[-(n - 1):]
                counts = table.get(key) if len(key) == n - 1 else None
                if not counts:
                    continue
                denominator = sum(counts.values()) + self.alpha * vocab_size
                weighted += weight * ((counts.get(token, 0) + self.alpha) / denominator)
        return weighted / total_weight if total_weight else 0.0

    def perplexity(self, tokens: list[int], vocab_size: int) -> float:
        if len(tokens) < 3:
            return float("nan")
        total = 0.0
        count = 0
        padded = [BOS] * (self.order - 1) + tokens
        for index, token in enumerate(padded):
            if index < self.order - 1:
                continue
            context = tuple(padded[max(0, index - self.order + 1):index])
            probability = self._prob(context, token, vocab_size)
            total += -math.log(max(probability, 1e-12))
            count += 1
        return math.exp(total / count) if count else float("nan")

    def next_tokens(self, context: list[int], limit: int = 8) -> list[tuple[int, float, int]]:
        candidates = self.unigrams.keys()
        scored = [(token, self._prob(tuple(context), token, self.vocab_size), count) for token, count in self.unigrams.items()]
        scored.sort(key=lambda item: (-item[1], -item[2]))
        return scored[:limit]

    def generate(self, tokenizer: ByteBPETokenizer, prompt: str, max_tokens: int = 30, seed: int = 7) -> str:
        rng = random.Random(seed)
        tokens = tokenizer.encode(prompt)
        output = list(tokens)
        for _ in range(max_tokens):
            context = output[-(self.order - 1):] if self.order > 1 else []
            candidates = self.next_tokens(context, limit=12)
            if not candidates:
                break
            token = rng.choices([item[0] for item in candidates], weights=[max(item[1], 1e-9) for item in candidates], k=1)[0]
            if token == EOS:
                break
            output.append(token)
        return tokenizer.decode(output)

    # ---------------------------------------------------------------- persistence
    def to_dict(self) -> dict[str, Any]:
        def encode_ngrams(table: dict[tuple[int, ...], Counter[int]]) -> dict[str, dict[str, int]]:
            return {",".join(str(value) for value in key): {str(token): count for token, count in counts.items()} for key, counts in table.items()}
        return {
            "type": "counted-ngram-baseline",
            "version": 1,
            "order": self.order,
            "alpha": self.alpha,
            "documents_seen": self.documents_seen,
            "tokens_seen": self.tokens_seen,
            "vocab_size": self.vocab_size,
            "unigrams": {str(token): count for token, count in self.unigrams.items()},
            "ngrams": {str(n): encode_ngrams(table) for n, table in self.ngrams.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CountedNgramModel":
        if payload.get("type") != "counted-ngram-baseline":
            raise NgramError("Not a counted n-gram baseline checkpoint")
        model = cls(int(payload["order"]), float(payload["alpha"]))
        model.documents_seen = int(payload.get("documents_seen", 0))
        model.tokens_seen = int(payload.get("tokens_seen", 0))
        model.vocab_size = int(payload.get("vocab_size", 256))
        model.unigrams = Counter({int(key): int(value) for key, value in payload.get("unigrams", {}).items()})
        for n_text, table in payload.get("ngrams", {}).items():
            n = int(n_text)
            target = defaultdict(Counter)
            for key_text, counts in table.items():
                key = tuple(int(value) for value in key_text.split(",") if value != "")
                target[key] = Counter({int(token): int(count) for token, count in counts.items()})
            model.ngrams[n] = target
        return model

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(self.to_dict(), ensure_ascii=False).encode("utf-8")
        target.write_bytes(data)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "CountedNgramModel":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise NgramError(f"Cannot read checkpoint: {exc}") from exc
        return cls.from_dict(payload)


def load_documents(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    docs = [part.strip() for part in text.split("\n\n") if part.strip()]
    return docs


def train_model(
    data_dir: str | Path,
    out_dir: str | Path,
    vocab_size: int = 320,
    order: int = 3,
    resume: bool = False,
) -> dict[str, Any]:
    from .tokenizer import train_bpe

    data = Path(data_dir)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    train_path = data / "train.txt"
    val_path = data / "val.txt"
    if not train_path.exists():
        raise NgramError("Prepared train.txt is missing; run prepare first")
    train_docs = load_documents(train_path)
    val_docs = load_documents(val_path) if val_path.exists() else []
    tokenizer_path = target / "tokenizer.json"
    if tokenizer_path.exists():
        tokenizer = ByteBPETokenizer.load(tokenizer_path)
    else:
        tokenizer, log = train_bpe(train_path.read_text(encoding="utf-8"), vocab_size=vocab_size)
        tokenizer.save(tokenizer_path)
        (target / "tokenizer_training.json").write_text(json.dumps({"log": log}, indent=2) + "\n", encoding="utf-8")
    checkpoint_path = target / "checkpoint.json"
    if resume and checkpoint_path.exists():
        model = CountedNgramModel.load(checkpoint_path)
        model.vocab_size = tokenizer.vocab_size
    else:
        model = CountedNgramModel(order=order)
        model.vocab_size = tokenizer.vocab_size
        model.documents_seen = 0

    def checkpoint(current: CountedNgramModel) -> None:
        current.save(checkpoint_path)

    pending = train_docs[model.documents_seen:] if resume else train_docs
    encoded = [tokenizer.encode(doc) for doc in pending]
    model.train(encoded, checkpoint=checkpoint, every=20)
    checkpoint(model)
    val_tokens = [token for doc in val_docs for token in tokenizer.encode(doc)]
    perplexity = model.perplexity(val_tokens, tokenizer.vocab_size) if val_tokens else float("nan")
    sample_prompt = "Nigerian expression:"
    sample = model.generate(tokenizer, sample_prompt, max_tokens=30)
    metrics = {
        "perplexity": None if math.isnan(perplexity) else round(perplexity, 3),
        "validation_tokens": len(val_tokens),
        "train_tokens": model.tokens_seen,
        "train_documents": model.documents_seen,
        "vocab_size": tokenizer.vocab_size,
        "order": model.order,
        "type": "counted n-gram baseline",
        "sample": sample,
        "note": "Perplexity is measured on a small held-out split of the curated corpus. It is not a benchmark against other models.",
    }
    model.save(target / "model.json")
    (target / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    tokenizer_hash = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    config = {
        "type": "counted-ngram-baseline",
        "order": order,
        "vocab_size": vocab_size,
        "actual_vocab_size": tokenizer.vocab_size,
        "alpha": model.alpha,
        "data_dir": str(data.resolve()),
        "tokenizer_sha256": tokenizer_hash,
        "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "training": "from initialization (counted parameters)",
        "fine_tuned": False,
        "chat_model": False,
        "hardware": "CPU",
    }
    (target / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    write_model_card(target, config, metrics)
    return {"config": config, "metrics": metrics, "out_dir": str(target)}


def write_model_card(target: Path, config: dict[str, Any], metrics: dict[str, Any]) -> Path:
    card = f"""# Ara teaching baseline: counted {config['order']}-gram model

**What this is:** a small counted n-gram language model trained from initialization on Ara's curated Nigerian-context corpus, over a byte-level BPE tokenizer. It exists to make dataset preparation, tokenizer training, checkpoints, evaluation, and export reproducible on a CPU.

**What this is not:** it is not Ara's chat model, not a transformer, not fine-tuned, and not evidence that Ara was trained from scratch. Ara's conversations use a separate pretrained model through a replaceable adapter.

- Type: {config['type']}
- Training: {config['training']}
- Fine-tuned: {config['fine_tuned']}
- Order: {config['order']}
- Tokenizer: byte-level BPE, {config['actual_vocab_size']} tokens
- Validation perplexity: {metrics['perplexity']}
- Training tokens: {metrics['train_tokens']}
- Validation tokens: {metrics['validation_tokens']}
- Hardware: {config['hardware']}
- Created: {config['created_at']}
- Tokenizer SHA-256: {config['tokenizer_sha256']}

## Data
Derived from `ara/data/nigeria/{{expressions,history,culture}}.json`, a curated seed corpus. No private messages, private accounts, restricted groups, or personal data were used. The corpus is small and does not represent every Nigerian language or community. Instruction pairs in `chat.jsonl` are derived from the same notes.

## Limitations
- A counted n-gram baseline cannot hold a conversation reliably.
- Perplexity on this small, stylistically uniform corpus is not comparable to general benchmarks.
- Regional and generational coverage is incomplete.
- No quantization is produced here. For a fine-tuned transformer, quantization via llama.cpp is a separate, hardware-dependent step.

## Reproduction
```
python -m ara.training.train --data {config['data_dir']} --out {target} --vocab-size {config['vocab_size']} --order {config['order']}
```
"""
    path = target / "MODEL_CARD.md"
    path.write_text(card, encoding="utf-8")
    return path


def evaluate_checkpoint(model_path: str | Path, tokenizer_path: str | Path, val_path: str | Path) -> dict[str, Any]:
    try:
        model = CountedNgramModel.load(model_path)
        tokenizer = ByteBPETokenizer.load(tokenizer_path)
    except (NgramError, TokenizerError) as exc:
        raise NgramError(str(exc)) from exc
    docs = load_documents(val_path)
    tokens = [token for doc in docs for token in tokenizer.encode(doc)]
    perplexity = model.perplexity(tokens, tokenizer.vocab_size) if tokens else float("nan")
    return {
        "perplexity": None if math.isnan(perplexity) else round(perplexity, 3),
        "documents": len(docs),
        "tokens": len(tokens),
        "vocab_size": tokenizer.vocab_size,
        "model_type": "counted n-gram baseline",
    }
