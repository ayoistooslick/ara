from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from .safety import contains_credential, redact_sensitive


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    """Small local SQLite store. Every public write is committed atomically."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=10000")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def initialize(self) -> None:
        try:
            with self.connect() as db:
                db.execute("PRAGMA journal_mode=WAL")
                db.executescript("""
                CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY, conversation_id TEXT NOT NULL, role TEXT NOT NULL,
                    content TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_conversation ON messages(conversation_id, id);
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY, kind TEXT NOT NULL, content TEXT NOT NULL,
                    source TEXT NOT NULL, confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
                    status TEXT NOT NULL DEFAULT 'active', created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0, evidence TEXT NOT NULL DEFAULT '[]'
                );
                CREATE INDEX IF NOT EXISTS memories_kind_status ON memories(kind, status);
                CREATE TABLE IF NOT EXISTS identity (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS goals (
                    id INTEGER PRIMARY KEY, title TEXT NOT NULL, details TEXT NOT NULL DEFAULT '',
                    priority INTEGER NOT NULL DEFAULT 3, status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY, context TEXT NOT NULL, options TEXT NOT NULL,
                    selected TEXT NOT NULL, rationale TEXT NOT NULL, outcome TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS skills (
                    id INTEGER PRIMARY KEY, name TEXT NOT NULL, version INTEGER NOT NULL,
                    description TEXT NOT NULL, procedure_json TEXT NOT NULL,
                    conditions TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
                    UNIQUE(name, version)
                );
                CREATE TABLE IF NOT EXISTS tool_runs (
                    id INTEGER PRIMARY KEY, tool_name TEXT NOT NULL, input_json TEXT NOT NULL,
                    status TEXT NOT NULL, output TEXT NOT NULL, created_at TEXT NOT NULL
                );
                """)
                row = db.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
                if row is None:
                    db.execute("INSERT INTO schema_version(version) VALUES (1)")
                elif row["version"] > 1:
                    raise RuntimeError(f"Database schema {row['version']} is newer than this Ara version")
        except sqlite3.DatabaseError as exc:
            raise RuntimeError(f"Ara could not initialize its local database at {self.path}: {exc}") from exc

    def add_message(self, conversation_id: str, role: str, content: str) -> int:
        content = redact_sensitive(content)
        with self.connect() as db:
            cur = db.execute("INSERT INTO messages(conversation_id,role,content,created_at) VALUES(?,?,?,?)", (conversation_id, role, content, now()))
            return int(cur.lastrowid)

    def recent_messages(self, conversation_id: str, limit: int = 12) -> list[dict[str, str]]:
        with self.connect() as db:
            rows = db.execute("SELECT role,content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT ?", (conversation_id, limit)).fetchall()
            return [dict(row) for row in reversed(rows)]

    def conversations(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT conversation_id,COUNT(*) AS message_count,MAX(created_at) AS updated_at,MAX(id) AS last_id FROM messages GROUP BY conversation_id ORDER BY last_id DESC").fetchall()
            return [dict(row) for row in rows]

    def add_memory(self, kind: str, content: str, source: str = "user", confidence: float = 0.8) -> tuple[int, list[dict[str, Any]]]:
        content = content.strip()
        if not content:
            raise ValueError("Memory content cannot be empty")
        if contains_credential(content):
            raise ValueError("Ara will not store passwords, API keys, tokens, or private keys in ordinary memory")
        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        stamp = now()
        words = {w.casefold() for w in content.split() if len(w) > 3}
        conflicts: list[dict[str, Any]] = []
        with self.connect() as db:
            existing = db.execute("SELECT id,kind,content,confidence FROM memories WHERE status='active' AND kind=?", (kind,)).fetchall()
            for row in existing:
                if row["content"].casefold().strip(" .!?\t") == content.casefold().strip(" .!?\t"):
                    db.execute("UPDATE memories SET access_count=access_count+1,updated_at=? WHERE id=?", (stamp, row["id"]))
                    return int(row["id"]), []
                prior = {w.casefold() for w in row["content"].split() if len(w) > 3}
                overlap = len(words & prior) / max(1, len(words | prior))
                if overlap >= 0.45:
                    conflicts.append(dict(row))
            cur = db.execute("INSERT INTO memories(kind,content,source,confidence,created_at,updated_at,evidence) VALUES(?,?,?,?,?,?,?)", (kind, content, source, confidence, stamp, stamp, json.dumps([source])))
            return int(cur.lastrowid), conflicts

    def search_memories(self, query: str, limit: int = 8, kind: str | None = None) -> list[dict[str, Any]]:
        terms = {t.casefold().strip(".,!?;:'\"()[]{}") for t in query.split() if len(t.strip(".,!?;:'\"()[]{}")) > 2}
        if not terms:
            return []
        with self.connect() as db:
            sql = "SELECT * FROM memories WHERE status='active'"
            params: list[Any] = []
            if kind:
                sql += " AND kind=?"
                params.append(kind)
            rows = db.execute(sql, params).fetchall()
            scored = []
            for row in rows:
                tokens = {t.casefold().strip(".,!?;:'\"()[]{}") for t in row["content"].split()}
                overlap = len(terms & tokens)
                if overlap:
                    score = overlap / (len(terms) ** 0.5) + min(float(row["confidence"]), 1.0) * 0.15
                    scored.append((score, dict(row)))
            scored.sort(key=lambda item: (item[0], item[1]["updated_at"]), reverse=True)
            result = [row for _, row in scored[:limit]]
            ids = [row["id"] for row in result]
            if ids:
                db.executemany("UPDATE memories SET access_count=access_count+1 WHERE id=?", [(i,) for i in ids])
            return result

    def list_memories(self, limit: int = 200) -> list[dict[str, Any]]:
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM memories WHERE status='active' ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()]

    def get_memory(self, memory_id: int) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
            return dict(row) if row else None

    def correct_memory(self, memory_id: int, content: str, confidence: float = 1.0) -> bool:
        if contains_credential(content):
            raise ValueError("Ara will not store credential values in ordinary memory")
        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        with self.connect() as db:
            cur = db.execute("UPDATE memories SET content=?,confidence=?,source='user-correction',updated_at=? WHERE id=? AND status='active'", (content.strip(), confidence, now(), memory_id))
            return cur.rowcount > 0

    def forget_memory(self, memory_id: int) -> bool:
        with self.connect() as db:
            cur = db.execute("UPDATE memories SET status='forgotten',updated_at=? WHERE id=? AND status='active'", (now(), memory_id))
            return cur.rowcount > 0

    def set_identity(self, key: str, value: str) -> None:
        if contains_credential(value):
            raise ValueError("Ara will not store credential values in identity settings")
        with self.connect() as db:
            db.execute("INSERT INTO identity(key,value,updated_at) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at", (key, value, now()))

    def get_identity(self) -> dict[str, str]:
        defaults = {"temperament": "steady, curious, candid, and considerate", "values": "honesty, user ownership, privacy, and safe boundaries", "communication": "natural and context-aware; avoid filler and false certainty"}
        with self.connect() as db:
            for row in db.execute("SELECT key,value FROM identity"):
                if not row["key"].startswith("__"):
                    defaults[row["key"]] = row["value"]
        return defaults

    def active_conversation_id(self) -> str:
        with self.connect() as db:
            row = db.execute("SELECT value FROM identity WHERE key='__active_conversation_id'").fetchone()
            if row:
                return str(row["value"])
            value = str(uuid.uuid4())
            db.execute("INSERT INTO identity(key,value,updated_at) VALUES('__active_conversation_id',?,?)", (value, now()))
            return value

    def set_active_conversation_id(self, conversation_id: str) -> None:
        self.set_identity("__active_conversation_id", conversation_id)

    def add_goal(self, title: str, details: str = "", priority: int = 3) -> int:
        if contains_credential(title) or contains_credential(details):
            raise ValueError("Ara will not store credential values in goals")
        if not 1 <= priority <= 5:
            raise ValueError("Priority must be from 1 to 5")
        with self.connect() as db:
            cur = db.execute("INSERT INTO goals(title,details,priority,created_at,updated_at) VALUES(?,?,?,?,?)", (title.strip(), details, priority, now(), now()))
            return int(cur.lastrowid)

    def goals(self, status: str = "active") -> list[dict[str, Any]]:
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM goals WHERE status=? ORDER BY priority DESC,updated_at", (status,)).fetchall()]

    def set_goal_status(self, goal_id: int, status: str) -> bool:
        if status not in {"active", "done", "paused", "cancelled"}:
            raise ValueError("Invalid goal status")
        with self.connect() as db:
            cur = db.execute("UPDATE goals SET status=?,updated_at=? WHERE id=?", (status, now(), goal_id))
            return cur.rowcount > 0

    def record_decision(self, context: str, options: list[str], selected: str, rationale: str) -> int:
        if any(contains_credential(item) for item in [context, selected, rationale, *options]):
            raise ValueError("Ara will not store credential values in decisions")
        with self.connect() as db:
            cur = db.execute("INSERT INTO decisions(context,options,selected,rationale,created_at) VALUES(?,?,?,?,?)", (context, json.dumps(options), selected, rationale, now()))
            return int(cur.lastrowid)

    def decisions(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]

    def save_skill(self, name: str, description: str, procedure: list[dict[str, str]], conditions: str = "") -> int:
        if not name.strip() or not description.strip() or not procedure:
            raise ValueError("A skill needs a name, description, and at least one documented step")
        for step in procedure:
            if not isinstance(step, dict) or not step.get("action"):
                raise ValueError("Each skill step must be an object with an action")
        skill_text = [name, description, conditions, *[item for step in procedure for item in step.values() if isinstance(item, str)]]
        if any(contains_credential(value) for value in skill_text):
            raise ValueError("Ara will not store credential values in skills")
        with self.connect() as db:
            row = db.execute("SELECT COALESCE(MAX(version),0) AS v FROM skills WHERE name=?", (name.strip(),)).fetchone()
            version = int(row["v"]) + 1
            cur = db.execute("INSERT INTO skills(name,version,description,procedure_json,conditions,created_at) VALUES(?,?,?,?,?,?)", (name.strip(), version, description.strip(), json.dumps(procedure), conditions, now()))
            return int(cur.lastrowid)

    def skills(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT s.* FROM skills s JOIN (SELECT name,MAX(version) version FROM skills GROUP BY name) latest USING(name,version) ORDER BY name").fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["procedure"] = json.loads(item.pop("procedure_json"))
                result.append(item)
            return result

    def remove_skill(self, name: str) -> bool:
        with self.connect() as db:
            cur = db.execute("DELETE FROM skills WHERE name=?", (name,))
            return cur.rowcount > 0

    def record_tool(self, name: str, input_value: Any, status: str, output: str) -> None:
        with self.connect() as db:
            safe_input = redact_sensitive(json.dumps(input_value, ensure_ascii=False, default=str))
            safe_output = redact_sensitive(output)
            db.execute("INSERT INTO tool_runs(tool_name,input_json,status,output,created_at) VALUES(?,?,?,?,?)", (name, safe_input, status, safe_output[:4000], now()))

    def tool_history(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM tool_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]

    def counts(self) -> dict[str, int]:
        with self.connect() as db:
            return {table: int(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in ("messages", "memories", "goals", "skills")}

    def backup_to(self, destination: str | Path) -> Path:
        target = Path(destination).expanduser().resolve()
        if target == self.path:
            raise ValueError("Backup destination must be different from the active database")
        target.parent.mkdir(parents=True, exist_ok=True)
        dest = sqlite3.connect(target)
        try:
            with sqlite3.connect(self.path) as source:
                source.backup(dest)
        finally:
            dest.close()
        return target

    @staticmethod
    def check_backup(path: str | Path) -> bool:
        target = Path(path).expanduser().resolve()
        if not target.is_file():
            return False
        try:
            db = sqlite3.connect(target.as_uri() + "?mode=ro", uri=True)
        except (sqlite3.DatabaseError, OSError):
            return False
        try:
            result = db.execute("PRAGMA integrity_check").fetchone()
            if not result or result[0] != "ok":
                return False
            version = db.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
            required = {"messages", "memories", "identity", "goals", "decisions", "skills", "tool_runs"}
            present = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            return bool(version and version[0] == 1 and required <= present)
        except sqlite3.DatabaseError:
            return False
        finally:
            db.close()
