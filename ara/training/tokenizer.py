"""A small, dependency-free byte-level BPE tokenizer.

Byte-level base means every UTF-8 input is representable, including Pidgin,
Yoruba diacritics, Igbo dots, Hausa hooked letters, and emoji. Training is real
merge learning over the supplied corpus, not a hard-coded vocabulary.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable


class TokenizerError(RuntimeError):
    pass


class ByteBPETokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None, metadata: dict | None = None):
        self.merges: list[tuple[int, int]] = list(merges or [])
        self.metadata = metadata or {}
        self.vocab: dict[int, bytes] = {index: bytes([index]) for index in range(256)}
        for index, (left, right) in enumerate(self.merges, start=256):
            if left not in self.vocab or right not in self.vocab:
                raise TokenizerError("Merge references an unknown token")
            self.vocab[index] = self.vocab[left] + self.vocab[right]
        self._ranks = {pair: rank for rank, pair in enumerate(self.merges)}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    def _merge_once(self, tokens: list[int]) -> list[int]:
        if not tokens:
            return tokens
        output: list[int] = []
        index = 0
        while index < len(tokens):
            if index + 1 < len(tokens):
                pair = (tokens[index], tokens[index + 1])
                rank = self._ranks.get(pair)
                if rank is not None:
                    output.append(256 + rank)
                    index += 2
                    continue
            output.append(tokens[index])
            index += 1
        return output

    def encode(self, text: str) -> list[int]:
        tokens = list(text.encode("utf-8"))
        while True:
            merged = self._merge_once(tokens)
            if merged == tokens:
                return tokens
            tokens = merged

    def decode(self, tokens: Iterable[int]) -> str:
        data = b"".join(self.vocab.get(token, b"") for token in tokens)
        return data.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "byte-bpe",
            "version": 1,
            "merges": [[left, right] for left, right in self.merges],
            "metadata": self.metadata,
        }
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        target = Path(path)
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TokenizerError(f"Cannot read tokenizer {target}: {exc}") from exc
        if payload.get("type") != "byte-bpe" or not isinstance(payload.get("merges"), list):
            raise TokenizerError("Not an Ara byte-BPE tokenizer")
        merges = []
        for item in payload["merges"]:
            if not isinstance(item, list) or len(item) != 2 or not all(isinstance(value, int) for value in item):
                raise TokenizerError("Invalid merge entry")
            merges.append((item[0], item[1]))
        return cls(merges, payload.get("metadata", {}))


def train_bpe(text: str, vocab_size: int = 320, min_frequency: int = 2) -> tuple[ByteBPETokenizer, list[dict]]:
    """Learn merges from the corpus. Returns the tokenizer and a merge log."""
    if vocab_size < 256:
        raise ValueError("vocab_size must be at least 256 for byte-level tokens")
    if not text:
        raise ValueError("Cannot train a tokenizer on empty text")
    tokens = list(text.encode("utf-8"))
    merges: list[tuple[int, int]] = []
    log: list[dict] = []
    target_merges = vocab_size - 256
    for step in range(target_merges):
        pairs = Counter(zip(tokens, tokens[1:]))
        if not pairs:
            break
        pair, count = max(pairs.items(), key=lambda item: (item[1], -item[0][0], -item[0][1]))
        if count < min_frequency:
            break
        left, right = pair
        merges.append(pair)
        new_id = 256 + step
        output: list[int] = []
        index = 0
        while index < len(tokens):
            if index + 1 < len(tokens) and tokens[index] == left and tokens[index + 1] == right:
                output.append(new_id)
                index += 2
            else:
                output.append(tokens[index])
                index += 1
        tokens = output
        log.append({"step": step + 1, "pair": [left, right], "count": count, "tokens_after": len(tokens)})
    tokenizer = ByteBPETokenizer(merges, {"trained_vocab_size": 256 + len(merges), "corpus_bytes": len(text.encode("utf-8")), "min_frequency": min_frequency})
    return tokenizer, log
