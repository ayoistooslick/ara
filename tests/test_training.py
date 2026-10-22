from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ara.training.corpus import documents, instruction_pairs, prepare
from ara.training.finetune import export_bundle, quantization_guidance
from ara.training.model import CountedNgramModel, evaluate_checkpoint, train_model
from ara.training.tokenizer import ByteBPETokenizer, train_bpe


class TokenizerTests(unittest.TestCase):
    def test_byte_level_roundtrip_for_nigerian_text(self):
        text = "Sannu da aiki. Ẹ káàárọ̀. Ndewo, kedu ka ị mere? Yaya dai? Emoji: 🇳🇬 🐸"
        tokenizer, log = train_bpe(text * 4, vocab_size=280, min_frequency=2)
        self.assertGreater(tokenizer.vocab_size, 256)
        self.assertLessEqual(tokenizer.vocab_size, 280)
        encoded = tokenizer.encode(text)
        self.assertEqual(tokenizer.decode(encoded), text)
        self.assertLess(len(encoded), len(text.encode("utf-8")))
        self.assertTrue(log)

    def test_save_load_preserves_encoding(self):
        tokenizer, _ = train_bpe("Ara is a Nigerian frog. " * 10, vocab_size=270)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenizer.json"
            tokenizer.save(path)
            loaded = ByteBPETokenizer.load(path)
            for text in ("Ara is a Nigerian frog.", "Wahala no dey finish", "Ẹ káàárọ̀"):
                self.assertEqual(loaded.encode(text), tokenizer.encode(text))


class TrainingTests(unittest.TestCase):
    def test_dataset_preparation_is_reproducible_and_documented(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = prepare(Path(tmp) / "one", seed=7)
            second = prepare(Path(tmp) / "two", seed=7)
            self.assertEqual(first["sha256"], second["sha256"])
            self.assertGreater(first["documents"], 100)
            self.assertGreater(first["instruction_pairs"], 100)
            self.assertIn("No private messages", first["privacy"])
            self.assertIn("limitations", first)
            chats = (Path(tmp) / "one" / "chat.jsonl").read_text().splitlines()
            self.assertEqual(len(chats), first["instruction_pairs"])
            row = json.loads(chats[0])
            self.assertEqual([message["role"] for message in row["messages"]], ["system", "user", "assistant"])
            self.assertTrue(row["source"].startswith("knowledge/"))

    def test_documents_have_provenance_and_no_private_material(self):
        docs = documents()
        self.assertGreater(len(docs), 100)
        self.assertTrue(all(doc["sha256"] for doc in docs))
        joined = " ".join(doc["text"] for doc in docs)
        self.assertNotIn("private message", joined.casefold())
        self.assertNotRegex(joined, r"\b\d{11}\b")
        self.assertNotRegex(joined, r"[\w.+-]+@[\w-]+\.[\w.]+")

    def test_instruction_pairs_are_derived_not_invented(self):
        pairs = instruction_pairs()
        self.assertTrue(any("sannu" in pair["messages"][1]["content"].casefold() for pair in pairs))
        self.assertTrue(any("June 12" in pair["messages"][1]["content"] or "june-12" in pair["id"] for pair in pairs))

    def test_training_checkpoint_metrics_and_model_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            run = Path(tmp) / "run"
            prepare(data)
            report = train_model(data, run, vocab_size=280, order=3)
            self.assertGreater(report["metrics"]["train_tokens"], 1000)
            self.assertIsNotNone(report["metrics"]["perplexity"])
            for name in ("model.json", "checkpoint.json", "tokenizer.json", "config.json", "metrics.json", "MODEL_CARD.md"):
                self.assertTrue((run / name).exists(), name)
            card = (run / "MODEL_CARD.md").read_text()
            self.assertIn("not Ara's chat model", card)
            self.assertIn("not evidence that Ara was trained from scratch", card)
            self.assertIn("counted", card.casefold())
            config = json.loads((run / "config.json").read_text())
            self.assertFalse(config["fine_tuned"])
            self.assertFalse(config["chat_model"])
            self.assertEqual(config["type"], "counted-ngram-baseline")
            evaluation = evaluate_checkpoint(run / "model.json", run / "tokenizer.json", data / "val.txt")
            self.assertGreater(evaluation["tokens"], 0)
            self.assertEqual(evaluation["model_type"], "counted n-gram baseline")

    def test_resume_does_not_duplicate_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            run = Path(tmp) / "run"
            prepare(data)
            train_model(data, run, vocab_size=270, order=3)
            before = CountedNgramModel.load(run / "checkpoint.json").tokens_seen
            train_model(data, run, vocab_size=270, order=3, resume=True)
            after = CountedNgramModel.load(run / "checkpoint.json").tokens_seen
            self.assertEqual(before, after)

    def test_model_generates_without_claiming_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            run = Path(tmp) / "run"
            prepare(data)
            train_model(data, run, vocab_size=270, order=2)
            model = CountedNgramModel.load(run / "model.json")
            tokenizer = ByteBPETokenizer.load(run / "tokenizer.json")
            text = model.generate(tokenizer, "Nigerian ", max_tokens=10)
            self.assertIsInstance(text, str)
            self.assertGreaterEqual(len(text), len("Nigerian "))

    def test_finetune_export_is_not_reported_as_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            prepare(data)
            report = export_bundle(data, Path(tmp) / "finetune", "Qwen/Qwen2.5-1.5B-Instruct")
            self.assertFalse(report["run"])
            plan = json.loads((Path(tmp) / "finetune" / "finetune_plan.json").read_text())
            self.assertEqual(plan["status"], "NOT RUN")
            self.assertIn("not required", " ".join(plan["notes"]).casefold())
            self.assertTrue((Path(tmp) / "finetune" / "README.md").exists())
            quant = quantization_guidance()
            self.assertIn("not run", quant["note"].casefold())

    def test_training_cli_is_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            out = Path(tmp) / "run"
            env = dict(os.environ, PYTHONPATH=str(ROOT))
            result = subprocess.run(
                [sys.executable, "-m", "ara.training.train", "--data", str(data), "--out", str(out), "--vocab-size", "270"],
                cwd=ROOT, env=env, text=True, capture_output=True, timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["metrics"]["order"], 3)
            self.assertIn("not a benchmark", payload["metrics"]["note"])


if __name__ == "__main__":
    unittest.main()
