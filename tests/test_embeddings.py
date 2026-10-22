from __future__ import annotations

import json
import math
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ara.config import Config
from ara.engine import ConversationEngine
from ara.embeddings import (
    EmbeddingError,
    HashingEmbedder,
    HttpEmbedder,
    backfill_vectors,
    cosine,
    get_embedder,
    retrieval_report,
)
from ara.store import Store


class EmbeddingHandler(BaseHTTPRequestHandler):
    calls = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        EmbeddingHandler.calls.append(payload)
        text = payload.get("input", "")
        vector = [1.0 if word in text.casefold() else 0.0 for word in ("tea", "coffee", "night", "morning")]
        body = json.dumps({"data": [{"embedding": vector}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.config = Config(home=self.home, allowed_roots=(self.home,))
        self.embedder = HashingEmbedder()
        self.store = Store(self.config.db_path)

    def tearDown(self):
        self.temp.cleanup()

    def test_hash_embedding_is_deterministic_and_normalised(self):
        first = self.embedder.embed("Nigerian Pidgin and plain English")
        second = HashingEmbedder().embed("Nigerian Pidgin and plain English")
        self.assertEqual(first, second)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in first)), 1.0, places=5)
        other = self.embedder.embed("completely unrelated zebra calculus")
        self.assertLess(cosine(first, other), cosine(first, self.embedder.embed("Nigerian English and Pidgin")))
        self.assertEqual(self.embedder.kind, "hashed-lexical")

    def test_default_embedder_is_labelled_non_semantic(self):
        embedder = get_embedder(self.config)
        self.assertIsInstance(embedder, HashingEmbedder)
        report = retrieval_report(self.store, embedder)
        self.assertFalse(report["semantic"])
        self.assertIn("lexical", report["embedder"])

    def test_backfill_and_hybrid_retrieval(self):
        memory_id, _ = self.store.add_memory("preference", "User prefers concise morning answers", confidence=0.9)
        self.store.add_memory("fact", "User works night shifts", confidence=0.9)
        self.assertGreater(self.store.count_missing_vectors(), 0)
        count = backfill_vectors(self.store, self.embedder)
        self.assertEqual(count, 2)
        self.assertEqual(self.store.count_missing_vectors(self.embedder.name), 0)
        results = self.store.search_memories_hybrid("morning answers", embedder=self.embedder.embed, limit=2)
        self.assertEqual(results[0]["id"], memory_id)
        self.assertEqual(results[0]["retrieval"], "hybrid")
        self.assertIn("retrieval_score", results[0])
        self.assertEqual(len(self.store.memory_vectors(self.embedder.name)), 2)

    def test_hybrid_falls_back_to_lexical_when_embedder_fails(self):
        self.store.add_memory("fact", "User likes tea", confidence=0.9)

        def broken(text):
            raise RuntimeError("server down")

        results = self.store.search_memories_hybrid("likes tea", embedder=broken, limit=3)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["retrieval"], "lexical")

    def test_vectors_from_another_embedder_are_not_scored(self):
        memory_id, _ = self.store.add_memory("preference", "User prefers concise morning answers", confidence=0.9)
        self.store.set_memory_vector(memory_id, "old-model@old-url", [1.0, 0.0])
        results = self.store.search_memories_hybrid(
            "morning answers", embedder=self.embedder.embed, limit=3, model=self.embedder.name
        )
        self.assertEqual(results[0]["id"], memory_id)
        self.assertEqual(results[0]["retrieval"], "lexical")

    def test_prompt_labels_hashed_retrieval_without_claiming_semantic(self):
        self.store.add_memory("fact", "User likes tea in the morning", confidence=0.9)
        backfill_vectors(self.store, self.embedder)

        class _Model:
            def status(self):
                return False, "stub"

            def generate(self, messages):
                return "ok"

        engine = ConversationEngine(self.config, self.store, _Model(), conversation_id="label-test")
        system = engine._system_message("tea in the morning", "chat")
        self.assertIn("retrieval lexical+hashed", system)
        self.assertNotIn("retrieval hybrid", system)

    def test_forgetting_removes_vectors(self):
        memory_id, _ = self.store.add_memory("fact", "A vector should disappear")
        self.store.set_memory_vector(memory_id, self.embedder.name, self.embedder.embed("A vector should disappear"))
        self.assertTrue(self.store.forget_memory(memory_id))
        self.assertEqual(self.store.memory_vectors(self.embedder.name), {})

    def test_server_embedder_protocol_and_normalisation(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), EmbeddingHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            EmbeddingHandler.calls = []
            config = Config(home=self.home, allowed_roots=(self.home,), embedding_url=f"http://127.0.0.1:{server.server_port}/v1", embedding_model="stub-embed")
            embedder = get_embedder(config)
            self.assertIsInstance(embedder, HttpEmbedder)
            vector = embedder.embed("tea in the morning")
            self.assertAlmostEqual(math.sqrt(sum(value * value for value in vector)), 1.0, places=5)
            self.assertEqual(EmbeddingHandler.calls[0]["model"], "stub-embed")
            self.assertEqual(len(vector), 4)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_server_embedder_reports_failure(self):
        embedder = HttpEmbedder("http://127.0.0.1:19998/v1", "missing", timeout=1)
        with self.assertRaises(EmbeddingError):
            embedder.embed("hello")


if __name__ == "__main__":
    unittest.main()
