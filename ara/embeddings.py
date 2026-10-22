"""Embedding providers for memory retrieval.

Ara ships a deterministic feature-hashing embedder so hybrid retrieval works with
no downloads. It captures token and character overlap, not learned meaning; the
application labels it accordingly. A local OpenAI-compatible embedding server can
be configured for genuine semantic vectors.
"""
from __future__ import annotations

import hashlib
import json
import math
import urllib.error
import urllib.request
from typing import Iterable, Sequence

from .config import Config


class EmbeddingError(RuntimeError):
    pass


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) ** 2 for value in vector))
    if not norm:
        return [0.0 for _ in vector]
    return [float(value) / norm for value in vector]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


class Embedder:
    name = "base"
    kind = "unknown"
    dim = 0

    def embed(self, text: str) -> list[float]:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.name} ({self.kind}, {self.dim} dimensions)"


class HashingEmbedder(Embedder):
    """Feature hashing over word unigrams/bigrams and character trigrams.

    Deterministic across processes and machines. Vectors are sparse-ish but stored
    densely. This is a lexical-similarity model, not a trained semantic encoder.
    """

    kind = "hashed-lexical"
    dim = 384

    def __init__(self, dim: int = 384):
        if dim % 2:
            raise ValueError("dim must be even for signed hashing")
        self.dim = dim
        self.name = "ara-hash-v1"

    def _features(self, text: str) -> Iterable[tuple[str, float]]:
        clean = " ".join(text.casefold().split())
        tokens = [token.strip(".,!?;:'\"()[]{}") for token in clean.split()]
        tokens = [token for token in tokens if token]
        for token in tokens:
            yield "w:" + token, 1.0
        for left, right in zip(tokens, tokens[1:]):
            yield f"b:{left}_{right}", 0.7
        compact = clean.replace(" ", "_")
        for index in range(max(0, len(compact) - 2)):
            yield "c:" + compact[index:index + 3], 0.35

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        counts: dict[str, int] = {}
        for feature, weight in self._features(text):
            counts[feature] = counts.get(feature, 0) + 1
        for feature, count in counts.items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "big")
            index = raw % self.dim
            sign = 1.0 if (raw >> 63) & 1 else -1.0
            weight = 1.0 + math.log(count)
            # The weight from the last occurrence is adequate for a bag of features.
            vector[index] += sign * weight
        return _normalize(vector)


class HttpEmbedder(Embedder):
    """Client for a local OpenAI-compatible /embeddings endpoint."""

    kind = "server-semantic"

    def __init__(self, url: str, model: str, timeout: int = 30):
        if not url.startswith(("http://", "https://")):
            raise EmbeddingError("Embedding URL must be http(s)")
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.name = f"{model}@{self.url}"

    def embed(self, text: str) -> list[float]:
        body = json.dumps({"model": self.model, "input": text}).encode("utf-8")
        request = urllib.request.Request(
            self.url + "/embeddings",
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read(4_000_000).decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise EmbeddingError(f"Embedding server returned HTTP {exc.code}") from exc
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise EmbeddingError(f"Cannot reach the embedding server: {exc}") from exc
        try:
            vector = payload["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise EmbeddingError("Embedding server returned an unexpected payload") from exc
        if not isinstance(vector, list) or not vector or not all(isinstance(v, (int, float)) for v in vector):
            raise EmbeddingError("Embedding server returned no numeric vector")
        self.dim = len(vector)
        return _normalize([float(value) for value in vector])


def get_embedder(config: Config) -> Embedder:
    if config.embedding_url:
        return HttpEmbedder(config.embedding_url, config.embedding_model, timeout=min(config.request_timeout, 60))
    return HashingEmbedder()


def backfill_vectors(store, embedder: Embedder, memory_ids: Sequence[int] | None = None) -> int:
    """Compute and store vectors for active memories that do not have one."""
    stored = store.memory_vectors(embedder.name)
    count = 0
    for memory in store.list_memories(limit=10_000):
        if memory_ids is not None and memory["id"] not in memory_ids:
            continue
        if memory["id"] in stored:
            continue
        try:
            vector = embedder.embed(memory["content"])
        except EmbeddingError:
            raise
        if vector:
            store.set_memory_vector(memory["id"], embedder.name, vector)
            count += 1
    return count


def retrieval_report(store, embedder: Embedder) -> dict[str, object]:
    return {
        "embedder": embedder.describe(),
        "vectors": len(store.memory_vectors(embedder.name)),
        "missing": store.count_missing_vectors(embedder.name),
        "semantic": embedder.kind == "server-semantic",
    }
