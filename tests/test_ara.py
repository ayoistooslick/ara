from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import tempfile
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ara.config import Config, load_config
from ara.engine import ConversationEngine, evaluate_options, identify_learning, select_goal
from ara.model import LocalChatModel, ModelError
from ara.store import Store
from ara.tools import ToolError, ToolRegistry, calculate


class FakeModel:
    """A test double; this is not evidence of real model inference."""
    def __init__(self, answer="test reply"):
        self.answer = answer
        self.received = None
    def generate(self, messages):
        self.received = messages
        return self.answer


class AraTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.home = self.root / "ara-home"
        self.home.mkdir()
        self.config = Config(home=self.home, allowed_roots=(self.root,))
        self.store = Store(self.config.db_path)
    def tearDown(self):
        self.temp.cleanup()

    def test_config_defaults_and_toml(self):
        self.assertEqual(load_config(self.home).model_url, "http://127.0.0.1:8080/v1")
        (self.home / "config.toml").write_text('[model]\nname="local-qwen"\nmax_tokens=123\n[tools]\nallowed_roots=["/tmp"]\n')
        cfg = load_config(self.home)
        self.assertEqual(cfg.model_name, "local-qwen")
        self.assertEqual(cfg.max_tokens, 123)
        self.assertEqual(cfg.allowed_roots, (Path("/tmp"),))

    def test_cli_help_and_memory_commands(self):
        env = dict(os.environ, ARA_HOME=str(self.home), PYTHONPATH=str(ROOT))
        result = subprocess.run([sys.executable, "-m", "ara", "--help"], cwd=ROOT, env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, "-m", "ara", "memory", "add", "user likes tea", "--kind", "preference"], cwd=ROOT, env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Stored memory", result.stdout)
        output = self.root / "export.json"
        result = subprocess.run([sys.executable, "-m", "ara", "memory", "export", str(output)], cwd=ROOT, env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(output.read_text())[0]["content"], "user likes tea")

    def test_memory_persists_and_retrieves(self):
        self.store.add_memory("preference", "User prefers concise answers", confidence=.9)
        reopened = Store(self.config.db_path)
        rows = reopened.search_memories("concise answers", 3)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["content"], "User prefers concise answers")

    def test_memory_dedup_conflicts_correction_and_forget(self):
        mid, _ = self.store.add_memory("preference", "User prefers tea", confidence=.8)
        same, conflicts = self.store.add_memory("preference", "User prefers tea.", confidence=.8)
        self.assertEqual(mid, same)
        self.assertEqual(conflicts, [])
        _, candidates = self.store.add_memory("preference", "User prefers coffee", confidence=.8)
        self.assertTrue(candidates)
        self.assertTrue(self.store.correct_memory(mid, "User prefers herbal tea"))
        self.assertEqual(self.store.get_memory(mid)["source"], "user-correction")
        self.assertTrue(self.store.forget_memory(mid))
        self.assertEqual(self.store.search_memories("herbal tea"), [])
        with self.assertRaises(ValueError): self.store.add_memory("fact", "my password is AraCredentialFixture12345")

    def test_conservative_learning(self):
        self.assertEqual(identify_learning("I prefer mornings"), ("preference", "User prefers mornings"))
        self.assertEqual(identify_learning("Remember that I work nights"), ("fact", "I work nights"))
        self.assertIsNone(identify_learning("The weather is nice"))

    def test_learning_is_persisted_before_conversation_and_context_retrieved(self):
        model = FakeModel()
        engine = ConversationEngine(self.config, self.store, model, "test-session")
        answer, learning = engine.respond("I prefer concise answers")
        self.assertEqual(answer, "test reply")
        self.assertIsNotNone(learning.saved_id)
        system = model.received[0]["content"]
        self.assertIn("User prefers concise answers", system)
        self.assertEqual(len(self.store.recent_messages("test-session")), 2)

    def test_active_conversation_resumes_across_engine_instances(self):
        first = ConversationEngine(self.config, self.store, FakeModel("first reply"))
        first_id = first.conversation_id
        first.respond("Keep this conversation on this machine")
        reopened = Store(self.config.db_path)
        second_model = FakeModel("second reply")
        second = ConversationEngine(self.config, reopened, second_model)
        self.assertEqual(second.conversation_id, first_id)
        second.respond("Continue")
        self.assertIn("Keep this conversation on this machine", [m["content"] for m in second_model.received[1:]])
        self.assertNotIn("__active_conversation_id", reopened.get_identity())

    def test_credentials_are_redacted_before_storage_and_model_context(self):
        model = FakeModel()
        engine = ConversationEngine(self.config, self.store, model, "secret-test")
        _, learning = engine.respond("Remember my password is AraCredentialFixture12345 and keep it")
        self.assertTrue(learning.sensitive_rejected)
        self.assertNotIn("AraCredentialFixture12345", str(self.store.recent_messages("secret-test")))
        self.assertNotIn("AraCredentialFixture12345", str(model.received))
        self.assertIn("redacted", str(self.store.recent_messages("secret-test")))

    def test_identity_persists(self):
        self.store.set_identity("communication", "brief")
        self.assertEqual(Store(self.config.db_path).get_identity()["communication"], "brief")

    def test_goal_and_decision(self):
        a = self.store.add_goal("lower", priority=2)
        b = self.store.add_goal("higher", priority=5)
        self.assertEqual(select_goal(self.store.goals())["id"], b)
        self.assertTrue(self.store.set_goal_status(a, "done"))
        chosen, rationale = evaluate_options(["A", "B"], {"A": 1.0, "B": 3.0})
        self.assertEqual(chosen, "B")
        self.store.record_decision("test", ["A", "B"], chosen, rationale)
        self.assertEqual(self.store.decisions()[0]["selected"], "B")

    def test_safe_calculator(self):
        self.assertEqual(calculate("(17 + 5) * 3"), 66)
        with self.assertRaises(ToolError): calculate("__import__('os').system('echo bad')")
        with self.assertRaises(ToolError): calculate("2**1000")

    def test_tool_permission_roots_and_audit(self):
        tools = ToolRegistry(self.config, self.store)
        self.assertEqual(tools.run("calculate", {"expression": "6*7"}), "42")
        safe = self.root / "safe.txt"
        with self.assertRaises(ToolError): tools.run("write_file", {"path": str(safe), "content": "hello"})
        self.assertFalse(safe.exists())
        self.assertIn("Wrote", tools.run("write_file", {"path": str(safe), "content": "hello"}, approved=True))
        self.assertEqual(tools.run("read_file", {"path": str(safe)}), "hello")
        with self.assertRaises(ToolError): tools.run("write_file", {"path": str(safe), "content": "replace"}, approved=True)
        with self.assertRaises(ToolError): tools.run("read_file", {"path": "/etc/passwd"})
        self.assertGreaterEqual(self.store.counts()["messages"], 0)
        with self.store.connect() as db: self.assertGreaterEqual(db.execute("SELECT COUNT(*) FROM tool_runs").fetchone()[0], 4)
        tools.run("write_file", {"path": str(self.root / "credentials.txt"), "content": "password is this is private"}, approved=True)
        with self.store.connect() as db:
            audit = " ".join(row[0] + row[1] for row in db.execute("SELECT input_json,output FROM tool_runs"))
        self.assertNotIn("this is private", audit)

    def test_skills_version_and_removal(self):
        steps = [{"action": "inspect files", "verification": "review list"}]
        self.store.save_skill("sort-project", "Organize a project", steps)
        self.store.save_skill("sort-project", "Updated organization", steps, "Only when asked")
        skill = self.store.skills()[0]
        self.assertEqual(skill["version"], 2)
        self.assertTrue(self.store.remove_skill("sort-project"))
        self.assertEqual(self.store.skills(), [])

    def test_model_rejects_remote_by_default_and_handles_offline(self):
        cfg = Config(home=self.home, model_url="https://example.org/v1")
        with self.assertRaises(ModelError): LocalChatModel(cfg)
        local = LocalChatModel(self.config)
        ok, message = local.status()
        self.assertFalse(ok)
        self.assertIn("No compatible model server", message)
        with self.assertRaises(ModelError): local.generate([{"role": "user", "content": "hello"}])

    def test_model_adapter_local_http_protocol(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps({"data": [{"id": "test-local-model"}]}).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                body = json.dumps({"choices": [{"message": {"content": "stub response"}}]}).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            def log_message(self, *_): pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            cfg = Config(home=self.home, model_url=f"http://127.0.0.1:{server.server_port}/v1")
            model = LocalChatModel(cfg)
            self.assertTrue(model.status()[0])
            self.assertEqual(model.generate([{"role": "user", "content": "hello"}]), "stub response")
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=2)

    def test_backup_integrity_and_restoreable_contents(self):
        self.store.add_memory("fact", "backup test")
        backup = self.root / "backup.sqlite3"
        self.store.backup_to(backup)
        self.assertTrue(Store.check_backup(backup))
        restored = Store(backup)
        self.assertEqual(restored.list_memories()[0]["content"], "backup test")
        self.assertFalse(Store.check_backup(self.root / "missing.sqlite3"))
        empty_sqlite = self.root / "empty.sqlite3"
        import sqlite3
        sqlite3.connect(empty_sqlite).close()
        self.assertFalse(Store.check_backup(empty_sqlite))

    def test_skill_json_failure_is_not_hidden(self):
        with self.assertRaises(ValueError): self.store.save_skill("x", "bad", [{"not-action": "none"}])


if __name__ == "__main__":
    unittest.main()
