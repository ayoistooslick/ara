from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ara.store import SCHEMA_VERSION, Store

V1_SCHEMA = """
CREATE TABLE schema_version (version INTEGER NOT NULL);
INSERT INTO schema_version(version) VALUES (1);
CREATE TABLE messages (
  id INTEGER PRIMARY KEY, conversation_id TEXT NOT NULL, role TEXT NOT NULL,
  content TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE memories (
  id INTEGER PRIMARY KEY, kind TEXT NOT NULL, content TEXT NOT NULL,
  source TEXT NOT NULL, confidence REAL NOT NULL, status TEXT NOT NULL DEFAULT 'active',
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL, access_count INTEGER NOT NULL DEFAULT 0,
  evidence TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE identity (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE goals (
  id INTEGER PRIMARY KEY, title TEXT NOT NULL, details TEXT NOT NULL DEFAULT '',
  priority INTEGER NOT NULL DEFAULT 3, status TEXT NOT NULL DEFAULT 'active',
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE decisions (
  id INTEGER PRIMARY KEY, context TEXT NOT NULL, options TEXT NOT NULL,
  selected TEXT NOT NULL, rationale TEXT NOT NULL, outcome TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL
);
CREATE TABLE skills (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, version INTEGER NOT NULL,
  description TEXT NOT NULL, procedure_json TEXT NOT NULL, conditions TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL, UNIQUE(name, version)
);
CREATE TABLE tool_runs (
  id INTEGER PRIMARY KEY, tool_name TEXT NOT NULL, input_json TEXT NOT NULL,
  status TEXT NOT NULL, output TEXT NOT NULL, created_at TEXT NOT NULL
);
INSERT INTO memories(kind,content,source,confidence,status,created_at,updated_at)
VALUES('fact','A memory from before the upgrade','user',0.9,'active','2026-01-01T00:00:00+00:00','2026-01-01T00:00:00+00:00');
INSERT INTO identity(key,value,updated_at) VALUES('temperament','old identity','2026-01-01T00:00:00+00:00');
INSERT INTO messages(conversation_id,role,content,created_at) VALUES('old','user','old message','2026-01-01T00:00:00+00:00');
"""


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.db = self.home / "ara.sqlite3"
        with sqlite3.connect(self.db) as db:
            db.executescript(V1_SCHEMA)

    def tearDown(self):
        self.temp.cleanup()

    def test_v1_database_upgrades_without_losing_data(self):
        self.assertTrue(Store.check_backup(self.db))
        store = Store(self.db)
        self.assertIsNotNone(store.last_migration_backup)
        self.assertTrue(store.last_migration_backup.exists())
        with store.connect() as db:
            version = db.execute("SELECT version FROM schema_version").fetchone()[0]
            columns = {row[1] for row in db.execute("PRAGMA table_info(memories)")}
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertEqual(version, SCHEMA_VERSION)
        self.assertTrue({"topic", "language", "region", "source_url"} <= columns)
        self.assertTrue({"memory_vectors", "news", "web_sessions", "web_events"} <= tables)
        memories = store.list_memories()
        self.assertEqual(memories[0]["content"], "A memory from before the upgrade")
        self.assertEqual(store.get_identity()["temperament"], "old identity")
        self.assertEqual(store.recent_messages("old"), [{"role": "user", "content": "old message"}])
        self.assertTrue(Store.check_backup(store.last_migration_backup))

    def test_reopening_does_not_migrate_again(self):
        Store(self.db)
        reopened = Store(self.db)
        self.assertIsNone(reopened.last_migration_backup)
        with reopened.connect() as db:
            self.assertEqual(db.execute("SELECT version FROM schema_version").fetchone()[0], SCHEMA_VERSION)

    def test_newer_schema_is_refused(self):
        with sqlite3.connect(self.db) as db:
            db.execute("UPDATE schema_version SET version=99")
        with self.assertRaises(RuntimeError):
            Store(self.db)

    def test_corrupt_database_reports_a_clear_error(self):
        corrupt = self.home / "corrupt.sqlite3"
        corrupt.write_bytes(b"not a database")
        with self.assertRaises(RuntimeError):
            Store(corrupt)


if __name__ == "__main__":
    unittest.main()
