from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ara.nigeria import BUILTIN_DIR, KnowledgeBase, evaluate
from ara.nigeria_pipeline import ingest_jsonl, validate_corpus, write_manifest


class NigeriaKnowledgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = KnowledgeBase.load()

    def test_corpus_loads_with_explicit_provenance(self):
        self.assertGreaterEqual(len(self.kb.entries), 100)
        stats = self.kb.stats()
        self.assertGreater(stats["by_kind"]["expressions"], 50)
        self.assertGreater(stats["by_kind"]["history"], 15)
        self.assertGreater(stats["by_kind"]["culture"], 15)
        for kind in ("expressions", "history", "culture"):
            metadata = self.kb.metadata[kind]
            self.assertTrue(metadata.get("license"))
            self.assertTrue(metadata.get("provenance"))

    def test_validation_passes_and_manifest_hashes(self):
        report = validate_corpus()
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(report["counts"], {"expressions": 65, "history": 23, "culture": 23})
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            for name in ("expressions.json", "history.json", "culture.json", "eval.json"):
                shutil.copy2(BUILTIN_DIR / name, target / name)
            manifest_path = write_manifest(target, target / "manifest.json")
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["counts"]["expressions"], 65)
            self.assertEqual(len(manifest["sha256"]), 64)
            self.assertGreater(manifest["evaluation_cases"], 30)

    def test_figurative_expression_is_not_read_literally(self):
        result = self.kb.interpret("I don chop breakfast this morning.")
        ids = [entry.id for entry in result.detected]
        self.assertIn("chop-breakfast", ids)
        self.assertEqual(result.register, "pidgin")
        joined = " ".join(result.guidance).casefold()
        self.assertIn("dump", joined)
        self.assertFalse(result.clarify)

    def test_genuine_ambiguity_requests_clarification(self):
        bare = self.kb.interpret("Na wa o.")
        self.assertTrue(bare.clarify)
        self.assertIn("ambiguous", " ".join(bare.reasons).casefold())
        with_cue = self.kb.interpret("Na wa o, this price don double.")
        self.assertFalse(with_cue.clarify)

    def test_register_detection(self):
        self.assertEqual(self.kb.register("Dear Sir, kindly find attached the application."), "formal")
        self.assertEqual(self.kb.register("Abeg, wetin dey happen for there?"), "pidgin")
        self.assertEqual(self.kb.register("The meeting starts at nine."), "standard")
        self.assertEqual(self.kb.register("Sannu da aiki."), "nigerian-english")

    def test_regional_languages_are_named_not_flattened(self):
        result = self.kb.interpret("Sannu da aiki.")
        entry = next(item for item in result.detected if item.id == "sannu")
        self.assertIn("Hausa", entry.languages)
        self.assertIn("North", entry.regions)
        biko = self.kb.interpret("Biko, help me.")
        entry = next(item for item in biko.detected if item.id == "biko")
        self.assertIn("Igbo", entry.languages)
        self.assertNotIn("abeg", [item.id for item in biko.detected])

    def test_insult_is_flagged_with_context(self):
        result = self.kb.interpret("You dey crase? Werey!")
        self.assertIn("werey", [entry.id for entry in result.detected])
        self.assertIn("insult", " ".join(result.guidance).casefold())

    def test_stereotype_is_flagged_not_reinforced(self):
        result = self.kb.interpret("All Nigerians speak Pidgin, abi?")
        self.assertTrue(result.stereotype)
        self.assertIn("generalisation", " ".join(result.guidance))
        result = self.kb.interpret("Every Igbo person is good at business.")
        self.assertTrue(result.stereotype)

    def test_context_prompt_is_caveated_and_has_confidence(self):
        context = self.kb.context_for("I don chop breakfast")
        self.assertIn("fallible reference notes", context)
        self.assertIn("Confidence", context)
        self.assertIn("chop", context.casefold())
        self.assertIn("Do not assume", context)

    def test_context_requires_a_meaningful_match_not_a_shared_word(self):
        # A foreign headline that shares filler words with corpus entries must not
        # receive random historical context.
        unrelated = self.kb.history_context(
            "Trump backs Republican candidate's attack ad featuring rival's mother in Ohio"
        )
        self.assertEqual(unrelated, "")
        relevant = self.kb.history_context("Tinubu declared winner of the 2023 presidential election")
        self.assertIn("2023", relevant)
        self.assertEqual(self.kb.search("the of and", limit=3), [])

    def test_evaluation_measures_retrieval_and_interpretation(self):
        report = evaluate(self.kb)
        self.assertGreaterEqual(report["retrieval"]["cases"], 30)
        self.assertGreaterEqual(report["retrieval"]["accuracy"], 0.95)
        self.assertEqual(report["interpretation"]["accuracy"], 1.0)
        self.assertFalse(report["model"]["run"])
        self.assertIn("not", report["model"]["note"].casefold())

    def test_evaluation_model_scoring_is_keyword_based_and_labelled(self):
        report = evaluate(self.kb, answer_fn=lambda question: "The election was annulled and Abiola was involved in 1993.")
        self.assertTrue(report["model"]["run"])
        self.assertIn("not human", report["model"]["note"].casefold())

    def test_guarded_ingestion_rejects_private_and_personal_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "incoming.jsonl"
            rows = [
                {"id": "good-one", "term": "testterm", "meaning": "A test meaning that is long enough.", "register": "informal", "regions": ["Lagos"], "languages": ["Nigerian Pidgin"]},
                {"id": "private-one", "term": "private", "meaning": "Taken from a private WhatsApp group chat.", "register": "informal", "regions": ["Lagos"], "languages": ["Nigerian Pidgin"], "private": True},
                {"id": "pii", "term": "phone", "meaning": "Call 08031234567 for the meaning of this expression.", "register": "informal", "regions": ["Lagos"], "languages": ["Nigerian Pidgin"]},
            ]
            source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            with self.assertRaises(ValueError):
                ingest_jsonl(source, "expressions", Path(tmp) / "out", "", "community review")
            report = ingest_jsonl(source, "expressions", Path(tmp) / "out", "CC BY-SA 4.0", "explicit contributor consent")
            self.assertEqual(report["accepted"], 1)
            self.assertEqual(report["added"], 1)
            self.assertEqual([item["reason"] for item in report["rejected"]], ["private or consent-unclear material", "looks like personal data"])
            stored = json.loads((Path(tmp) / "out" / "expressions.json").read_text())
            self.assertEqual(stored["entries"][0]["provenance"]["license"], "CC BY-SA 4.0")
            # Importing the same record again is idempotent.
            second = ingest_jsonl(source, "expressions", Path(tmp) / "out", "CC BY-SA 4.0", "explicit contributor consent")
            self.assertEqual(second["added"], 0)

    def test_prompts_do_not_claim_universal_nigerianness(self):
        context = self.kb.context_for("Tell me about Nigerian humour")
        self.assertIn("region, generation, class", context)
        self.assertNotIn("all Nigerians", context.casefold())


if __name__ == "__main__":
    unittest.main()
