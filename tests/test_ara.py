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
        # Use a port that is not running a model server
        offline_config = Config(home=self.home, model_url="http://127.0.0.1:19999/v1")
        local = LocalChatModel(offline_config)
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


class TestModelErrorHandling(AraTests):
    """Test model error paths without needing a real server."""

    def test_model_rejects_invalid_url(self):
        with self.assertRaises(ModelError):
            LocalChatModel(Config(home=self.home, model_url="not-a-url"))

    def test_model_rejects_remote_by_default(self):
        with self.assertRaises(ModelError):
            LocalChatModel(Config(home=self.home, model_url="http://192.168.1.1:8080/v1"))

    def test_model_accepts_remote_when_allowed(self):
        model = LocalChatModel(Config(home=self.home, model_url="http://192.168.1.1:8080/v1", allow_remote_model=True))
        self.assertIsInstance(model, LocalChatModel)

    def test_model_rejects_https_localhost(self):
        with self.assertRaises(ModelError):
            LocalChatModel(Config(home=self.home, model_url="https://127.0.0.1:8080/v1"))


class TestConversationContinuity(AraTests):
    """Test multi-turn conversation with a fake model to verify context flow."""

    def test_multi_turn_context_accumulating(self):
        model = FakeModel("reply")
        engine = ConversationEngine(self.config, self.store, model, "ctx-test")
        engine.respond("First user message")
        engine.respond("Second user message")
        engine.respond("Third user message")
        history = self.store.recent_messages("ctx-test", 10)
        self.assertEqual(len(history), 6)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["content"], "First user message")

    def test_system_message_includes_identity_and_memories(self):
        self.store.set_identity("temperament", "test-temperament")
        self.store.add_memory("fact", "User likes tea", confidence=0.9)
        model = FakeModel("ok")
        engine = ConversationEngine(self.config, self.store, model, "sys-test")
        engine.respond("Do I like tea?")
        system = model.received[0]["content"]
        self.assertIn("test-temperament", system)
        self.assertIn("User likes tea", system)

    def test_correction_surfaces_in_system_message(self):
        self.store.add_memory("fact", "User works days", confidence=0.8)
        self.store.add_memory("correction", "User actually works nights", confidence=1.0)
        model = FakeModel("ok")
        engine = ConversationEngine(self.config, self.store, model, "corr-test")
        engine.respond("When does the user work?")
        system = model.received[0]["content"]
        self.assertIn("User actually works nights", system)


class TestMemoryLearningApplication(AraTests):
    """Test that learned information is retrieved and applied in later turns."""

    def test_preference_is_retrieved_and_included_in_context(self):
        self.store.add_memory("preference", "User prefers concise answers", confidence=0.9)
        model = FakeModel("reply")
        engine = ConversationEngine(self.config, self.store, model, "pref-test")
        engine.respond("I prefer concise answers in our chat")
        system = model.received[0]["content"]
        self.assertIn("User prefers concise answers", system)

    def test_forgotten_memory_not_included(self):
        mid, _ = self.store.add_memory("fact", "Secret detail", confidence=0.8)
        self.store.forget_memory(mid)
        model = FakeModel("reply")
        engine = ConversationEngine(self.config, self.store, model, "forget-test")
        engine.respond("Tell me the secret")
        system = model.received[0]["content"]
        self.assertNotIn("Secret detail", system)

    def test_sensitivity_rejection_prevents_memory_storage(self):
        model = FakeModel("reply")
        engine = ConversationEngine(self.config, self.store, model, "sens-test")
        _, learning = engine.respond("My password is SecretPass123 and remember it")
        self.assertTrue(learning.sensitive_rejected)
        self.assertIsNone(learning.saved_id)


class TestIdentityPersistence(AraTests):
    """Test that identity fields survive restarts."""

    def test_identity_fields_persist(self):
        self.store.set_identity("temperament", "warm and direct")
        self.store.set_identity("values", "privacy first")
        reopened = Store(self.config.db_path)
        ident = reopened.get_identity()
        self.assertEqual(ident["temperament"], "warm and direct")
        self.assertEqual(ident["values"], "privacy first")

    def test_default_identity_fields_exist(self):
        ident = self.store.get_identity()
        self.assertIn("temperament", ident)
        self.assertIn("values", ident)
        self.assertIn("communication", ident)


class TestGoalPersistence(AraTests):
    """Test that goals survive restarts and can be tracked."""

    def test_goals_persist_and_can_be_closed(self):
        gid = self.store.add_goal("Learn Python", "Study every evening", priority=4)
        reopened = Store(self.config.db_path)
        goals = reopened.goals()
        self.assertEqual(len(goals), 1)
        self.assertEqual(goals[0]["title"], "Learn Python")
        self.assertTrue(reopened.set_goal_status(gid, "done"))

    def test_select_goal_returns_highest_priority(self):
        low = self.store.add_goal("Low", priority=1)
        high = self.store.add_goal("High", priority=5)
        selected = select_goal(self.store.goals())
        self.assertEqual(selected["id"], high)


class TestActionSelection(AraTests):
    """Test the decision and action selection mechanisms."""

    def test_evaluate_options_selects_highest_score(self):
        chosen, rationale = evaluate_options(["A", "B", "C"], {"A": 1.0, "B": 5.0, "C": 3.0})
        self.assertEqual(chosen, "B")
        self.assertIn("5", rationale)

    def test_select_goal_returns_none_for_empty_list(self):
        self.assertIsNone(select_goal([]))


class TestOutcomeReplanning(AraTests):
    """Test the bounded action loop's replanning behavior."""

    def test_execute_next_runs_pending_step(self):
        from ara.tools import ToolRegistry
        from ara.engine import BoundedActionLoop, PlannedAction
        loop = BoundedActionLoop(self.config, self.store, FakeModel("done"), ToolRegistry(self.config, self.store))
        step = PlannedAction("calc", "calculate", {"expression": "1+1"}, "test", "read-only")
        loop.plan_for("test goal", [step])
        executed = loop.execute_next()
        self.assertEqual(executed.status, "ok")
        self.assertEqual(executed.result, "2")

    def test_execute_next_blocks_unapproved_writes(self):
        from ara.tools import ToolRegistry
        from ara.engine import BoundedActionLoop, PlannedAction
        loop = BoundedActionLoop(self.config, self.store, FakeModel("done"), ToolRegistry(self.config, self.store))
        step = PlannedAction("write", "write_file", {"path": str(self.root / "x.txt"), "content": "x"}, "test", "destructive")
        loop.plan_for("test", [step])
        executed = loop.execute_next(approved=False)
        self.assertEqual(executed.status, "blocked")
        self.assertIn("approval", executed.error)

    def test_needs_replan_after_failure(self):
        from ara.tools import ToolRegistry
        from ara.engine import BoundedActionLoop, PlannedAction
        loop = BoundedActionLoop(self.config, self.store, FakeModel("done"), ToolRegistry(self.config, self.store))
        step1 = PlannedAction("bad", "calculate", {"expression": "bad"}, "test", "read-only")
        step2 = PlannedAction("good", "calculate", {"expression": "1+1"}, "test", "read-only")
        loop.plan_for("test", [step1, step2])
        loop.execute_next()  # step1 fails
        self.assertTrue(loop.needs_replan())

    def test_plan_done_when_all_complete(self):
        from ara.tools import ToolRegistry
        from ara.engine import BoundedActionLoop, PlannedAction
        loop = BoundedActionLoop(self.config, self.store, FakeModel("done"), ToolRegistry(self.config, self.store))
        step = PlannedAction("calc", "calculate", {"expression": "1+1"}, "test", "read-only")
        loop.plan_for("test", [step])
        loop.execute_next()
        self.assertTrue(loop.plan_done())


class TestSkillExecution(AraTests):
    """Test that skills can be saved and executed."""

    def test_execute_simple_skill(self):
        steps = [{"action": "calculate", "expression": "6 * 7"}]
        self.store.save_skill("multiply-demo", "A simple calculation", steps)
        from ara.tools import ToolRegistry
        result = ToolRegistry(self.config, self.store).run("run_skill", {"skill_name": "multiply-demo"}, approved=True)
        self.assertIn("42", result)
        self.assertIn("Procedure completed", result)

    def test_execute_skill_with_file_operations(self):
        steps = [
            {"action": "write a file", "path": str(self.root / "from-skill.txt"), "content": "written by skill", "permission": "reversible"},
            {"action": "read a file", "path": str(self.root / "from-skill.txt")},
        ]
        self.store.save_skill("write-and-read", "Write then read", steps, "Only with approval")
        from ara.tools import ToolRegistry
        result = ToolRegistry(self.config, self.store).run("run_skill", {"skill_name": "write-and-read"}, approved=True)
        self.assertIn("written by skill", result)
        self.assertIn("Procedure completed", result)

    def test_skill_execution_fails_on_unknown_action(self):
        steps = [{"action": "teleport to mars"}]
        self.store.save_skill("impossible", "A useless skill", steps)
        from ara.tools import ToolRegistry
        with self.assertRaises(ToolError) as ctx:
            ToolRegistry(self.config, self.store).run("run_skill", {"skill_name": "impossible"}, approved=True)
        self.assertIn("Unknown skill action", str(ctx.exception))

    def test_skill_stops_on_failure_with_stop_policy(self):
        steps = [
            {"action": "calculate", "expression": "1 / 0"},
            {"action": "calculate", "expression": "1 + 1"},
        ]
        self.store.save_skill("failing", "Will fail", steps)
        from ara.tools import ToolRegistry
        result = ToolRegistry(self.config, self.store).run("run_skill", {"skill_name": "failing"}, approved=True)
        self.assertIn("Procedure stopped", result)
        self.assertNotIn("Procedure completed", result)

    def test_skill_requires_approval(self):
        steps = [{"action": "calculate", "expression": "1+1"}]
        self.store.save_skill("needs-approval", "Needs ok", steps)
        from ara.tools import ToolRegistry
        with self.assertRaises(ToolError) as ctx:
            ToolRegistry(self.config, self.store).run("run_skill", {"skill_name": "needs-approval"}, approved=False)
        self.assertIn("needs explicit approval", str(ctx.exception))


class TestPermissionBoundaries(AraTests):
    """Test that tool permissions are enforced correctly."""

    def test_read_only_operation_works_without_approval(self):
        registry = ToolRegistry(self.config, self.store)
        self.assertEqual(registry.run("calculate", {"expression": "10 + 5"}), "15")

    def test_write_file_without_approval_fails(self):
        registry = ToolRegistry(self.config, self.store)
        target = self.root / "unapproved.txt"
        with self.assertRaises(ToolError) as ctx:
            registry.run("write_file", {"path": str(target), "content": "x"})
        self.assertIn("needs explicit approval", str(ctx.exception))
        self.assertFalse(target.exists())

    def test_audit_log_redacts_file_contents(self):
        registry = ToolRegistry(self.config, self.store)
        target = self.root / "secret.txt"
        registry.run("write_file", {"path": str(target), "content": "password=secret123"}, approved=True)
        registry.run("read_file", {"path": str(target)})
        with self.store.connect() as db:
            rows = db.execute("SELECT input_json, output FROM tool_runs").fetchall()
            audit = " ".join(str(r[0]) + str(r[1]) for r in rows)
        self.assertNotIn("secret123", audit)


class TestBackupAndRestore(AraTests):
    """Test backup integrity and restoration."""

    def test_backup_contains_all_tables(self):
        self.store.add_memory("fact", "backup test content")
        self.store.add_goal("Test goal", priority=3)
        backup_path = self.root / "full-backup.sqlite3"
        self.store.backup_to(backup_path)
        self.assertTrue(Store.check_backup(backup_path))
        restored = Store(backup_path)
        self.assertEqual(restored.list_memories()[0]["content"], "backup test content")

    def test_backup_rejects_same_path(self):
        with self.assertRaises(ValueError):
            self.store.backup_to(self.config.db_path)

    def test_check_backup_fails_for_corrupt_file(self):
        corrupt = self.root / "corrupt.sqlite3"
        corrupt.write_bytes(b"not a database")
        self.assertFalse(Store.check_backup(corrupt))


class TestOfflineOperation(AraTests):
    """Test that Ara works without network access for local operations."""

    def test_memory_commands_work_offline(self):
        result = subprocess.run(
            [sys.executable, "-m", "ara", "memory", "add", "offline fact", "--kind", "fact"],
            cwd=ROOT, env=dict(os.environ, ARA_HOME=str(self.home), PYTHONPATH=str(ROOT)),
            text=True, capture_output=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("Stored memory", result.stdout)

    def test_goal_commands_work_offline(self):
        result = subprocess.run(
            [sys.executable, "-m", "ara", "goal", "add", "Offline goal", "--priority", "3"],
            cwd=ROOT, env=dict(os.environ, ARA_HOME=str(self.home), PYTHONPATH=str(ROOT)),
            text=True, capture_output=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("Added goal", result.stdout)

    def test_tool_calculate_works_offline(self):
        result = subprocess.run(
            [sys.executable, "-m", "ara", "tool", "calculate", "10*2"],
            cwd=ROOT, env=dict(os.environ, ARA_HOME=str(self.home), PYTHONPATH=str(ROOT)),
            text=True, capture_output=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("20", result.stdout)


class TestConfigurationFailures(AraTests):
    """Test that bad configuration is caught early."""

    def test_invalid_toml_reports_error(self):
        (self.home / "config.toml").write_text("[model\nbad syntax")
        with self.assertRaises(ValueError) as ctx:
            load_config(self.home)
        self.assertIn("Cannot read", str(ctx.exception))

    def test_negative_max_tokens_rejected(self):
        (self.home / "config.toml").write_text("[model\nmax_tokens = -1\n")
        with self.assertRaises(ValueError):
            load_config(self.home)


class TestDataIntegrityAcrossRestarts(AraTests):
    """Test that data survives store re-openings."""

    def test_messages_persist(self):
        self.store.add_message("conv1", "user", "Hello")
        self.store.add_message("conv1", "assistant", "Hi there")
        reopened = Store(self.config.db_path)
        msgs = reopened.recent_messages("conv1", 10)
        self.assertEqual(len(msgs), 2)

    def test_memories_persist(self):
        self.store.add_memory("fact", "Persistent fact", confidence=0.9)
        reopened = Store(self.config.db_path)
        rows = reopened.list_memories()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["content"], "Persistent fact")

    def test_active_conversation_resumes(self):
        first = Store(self.config.db_path)
        first_id = first.active_conversation_id()
        second = Store(self.config.db_path)
        self.assertEqual(second.active_conversation_id(), first_id)




if __name__ == "__main__":
    unittest.main()
