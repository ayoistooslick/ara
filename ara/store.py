from __future__ import annotations

import json
import math
import sqlite3
import struct
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from .safety import contains_credential, redact_sensitive

SCHEMA_VERSION = 2


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


CORE_TABLES = {"messages", "memories", "identity", "goals", "decisions", "skills", "tool_runs"}


class Store:
    """Small local SQLite store. Every public write is committed atomically."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.last_migration_backup: Path | None = None
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

    # ---------------------------------------------------------------- schema
    def initialize(self) -> None:
        try:
            with self.connect() as db:
                db.execute("PRAGMA journal_mode=WAL")
                db.executescript(
                    """
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
                    """
                )
                row = db.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
                if row is None:
                    db.execute("INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,))
                    version = SCHEMA_VERSION
                else:
                    version = int(row["version"])
                if version > SCHEMA_VERSION:
                    raise RuntimeError(
                        f"Database schema {version} is newer than this Ara version ({SCHEMA_VERSION}). "
                        "Upgrade Ara instead of downgrading the database."
                    )
                if version < SCHEMA_VERSION:
                    self._backup_before_migration()
                    self._migrate(db, version)
                else:
                    # Idempotent repair for databases created by early v2 builds
                    # that predate the news and session tables.
                    self._ensure_v2(db)
        except sqlite3.DatabaseError as exc:
            raise RuntimeError(f"Ara could not initialize its local database at {self.path}: {exc}") from exc

    def _backup_before_migration(self) -> None:
        """Copy the database before any destructive schema change."""
        try:
            target_dir = self.path.parent / "backups"
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / ("ara-pre-upgrade-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".sqlite3")
            source = sqlite3.connect(self.path)
            dest = sqlite3.connect(target)
            try:
                source.backup(dest)
            finally:
                dest.close()
                source.close()
            self.last_migration_backup = target
        except (OSError, sqlite3.DatabaseError):
            # A failed backup must not corrupt the live database; migrations remain
            # additive and each one is transactional. The failure is surfaced by the
            # caller through last_migration_backup remaining None.
            self.last_migration_backup = None

    @staticmethod
    def _columns(db: sqlite3.Connection, table: str) -> set[str]:
        return {row[1] for row in db.execute(f"PRAGMA table_info({table})")}

    def _migrate(self, db: sqlite3.Connection, version: int) -> None:
        """Additive migrations. Each upgrade keeps existing rows untouched."""
        if version < 2:
            self._ensure_v2(db)
            db.execute("UPDATE schema_version SET version=2")

    def _ensure_v2(self, db: sqlite3.Connection) -> None:
        columns = self._columns(db, "memories")
        for name, ddl in [
            ("topic", "TEXT NOT NULL DEFAULT ''"),
            ("language", "TEXT NOT NULL DEFAULT ''"),
            ("region", "TEXT NOT NULL DEFAULT ''"),
            ("source_url", "TEXT NOT NULL DEFAULT ''"),
        ]:
            if name not in columns:
                db.execute(f"ALTER TABLE memories ADD COLUMN {name} {ddl}")
        db.executescript(
                """
                    CREATE TABLE IF NOT EXISTS memory_vectors (
                        memory_id INTEGER PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
                        model TEXT NOT NULL, dim INTEGER NOT NULL, vector BLOB NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS news (
                        id INTEGER PRIMARY KEY, dedupe_key TEXT NOT NULL UNIQUE, title TEXT NOT NULL,
                        summary TEXT NOT NULL, link TEXT NOT NULL, source TEXT NOT NULL,
                        source_url TEXT NOT NULL,                        published_at TEXT NOT NULL, fetched_at TEXT NOT NULL,
                        topic TEXT NOT NULL, origin TEXT NOT NULL DEFAULT 'feed',
                        related_sources TEXT NOT NULL DEFAULT '[]'
                    );
                    CREATE INDEX IF NOT EXISTS news_published ON news(published_at DESC);
                    CREATE TABLE IF NOT EXISTS web_sessions (
                        token_hash TEXT PRIMARY KEY, csrf TEXT NOT NULL, conversation_id TEXT NOT NULL,
                        created_at TEXT NOT NULL, expires_at TEXT NOT NULL, last_seen TEXT NOT NULL,
                        label TEXT NOT NULL DEFAULT ''
                    );
                    CREATE TABLE IF NOT EXISTS web_events (
                        id INTEGER PRIMARY KEY, session_hash TEXT NOT NULL, kind TEXT NOT NULL,
                        detail TEXT NOT NULL, created_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS web_events_session ON web_events(session_hash, id DESC);
                    """
            )
        news_columns = self._columns(db, "news")
        if "related_sources" not in news_columns:
            db.execute("ALTER TABLE news ADD COLUMN related_sources TEXT NOT NULL DEFAULT '[]'")

    # ------------------------------------------------------------- conversations
    def add_message(self, conversation_id: str, role: str, content: str) -> int:
        content = redact_sensitive(content)
        with self.connect() as db:
            cur = db.execute(
                "INSERT INTO messages(conversation_id,role,content,created_at) VALUES(?,?,?,?)",
                (conversation_id, role, content, now()),
            )
            return int(cur.lastrowid)

    def recent_messages(self, conversation_id: str, limit: int = 12) -> list[dict[str, str]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT role,content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
            return [dict(row) for row in reversed(rows)]

    def conversation_messages(self, conversation_id: str, limit: int = 200) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT id,role,content,created_at FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
            return [dict(row) for row in reversed(rows)]

    def conversations(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT conversation_id,COUNT(*) AS message_count,MAX(created_at) AS updated_at,MAX(id) AS last_id "
                "FROM messages GROUP BY conversation_id ORDER BY last_id DESC"
            ).fetchall()
            return [dict(row) for row in rows]

    def delete_conversation(self, conversation_id: str) -> int:
        with self.connect() as db:
            cur = db.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
            return cur.rowcount

    # ----------------------------------------------------------------- memories
    def add_memory(
        self,
        kind: str,
        content: str,
        source: str = "user",
        confidence: float = 0.8,
        topic: str = "",
        language: str = "",
        region: str = "",
        source_url: str = "",
    ) -> tuple[int, list[dict[str, Any]]]:
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
            cur = db.execute(
                "INSERT INTO memories(kind,content,source,confidence,created_at,updated_at,evidence,topic,language,region,source_url) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (kind, content, source, confidence, stamp, stamp, json.dumps([source]), topic, language, region, source_url),
            )
            return int(cur.lastrowid), conflicts

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {
            token.casefold().strip(".,!?;:'\"()[]{}")
            for token in text.split()
            if len(token.strip(".,!?;:'\"()[]{}")) > 2
        }

    def search_memories(self, query: str, limit: int = 8, kind: str | None = None) -> list[dict[str, Any]]:
        terms = self._terms(query)
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
                overlap = len(terms & self._terms(row["content"]))
                if overlap:
                    score = overlap / (len(terms) ** 0.5) + min(float(row["confidence"]), 1.0) * 0.15
                    scored.append((score, dict(row)))
            scored.sort(key=lambda item: (item[0], item[1]["updated_at"]), reverse=True)
            result = [row for _, row in scored[:limit]]
            ids = [row["id"] for row in result]
            if ids:
                db.executemany("UPDATE memories SET access_count=access_count+1 WHERE id=?", [(i,) for i in ids])
            return result

    def search_memories_hybrid(
        self,
        query: str,
        embedder: Callable[[str], Sequence[float]] | None = None,
        limit: int = 8,
        kind: str | None = None,
        lexical_weight: float = 0.55,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Blend lexical overlap with stored vector similarity when vectors exist.

        Falls back to pure lexical ranking if no vector is available. This function
        never labels its output "semantic" unless a vector contributed to the score.
        """
        lexical = {row["id"]: row for row in self.search_memories(query, limit=max(limit * 4, 20), kind=kind)}

        def lexical_only() -> list[dict[str, Any]]:
            marked = []
            for row in list(lexical.values())[:limit]:
                item = dict(row)
                item.setdefault("retrieval", "lexical")
                marked.append(item)
            return marked

        if embedder is None:
            return lexical_only()
        # Only vectors from the active embedder may contribute. A hashed fallback
        # vector from an earlier configuration must not be scored with a server
        # embedding of a different dimension or meaning.
        vectors = self.memory_vectors(model)
        if not vectors:
            return lexical_only()
        try:
            query_vector = list(embedder(query))
        except Exception:
            return lexical_only()
        if not query_vector:
            return lexical_only()
        norm = math.sqrt(sum(value * value for value in query_vector)) or 1.0
        query_vector = [value / norm for value in query_vector]
        candidates: dict[int, dict[str, Any]] = {}
        with self.connect() as db:
            sql = "SELECT * FROM memories WHERE status='active'"
            params: list[Any] = []
            if kind:
                sql += " AND kind=?"
                params.append(kind)
            for row in db.execute(sql, params).fetchall():
                item = dict(row)
                if item["id"] in vectors:
                    candidates[item["id"]] = item
        for memory_id, item in lexical.items():
            candidates.setdefault(memory_id, item)
        scored: list[tuple[float, dict[str, Any]]] = []
        for memory_id, item in candidates.items():
            vector = vectors.get(memory_id)
            similarity = 0.0
            if vector:
                similarity = max(0.0, sum(a * b for a, b in zip(query_vector, vector)))
            terms = self._terms(query)
            overlap = len(terms & self._terms(item["content"]))
            lexical_score = overlap / max(1, len(terms)) if terms else 0.0
            base = lexical_weight * lexical_score + (1 - lexical_weight) * similarity
            if base <= 0:
                continue
            score = base + min(float(item["confidence"]), 1.0) * 0.1
            item = dict(item)
            item["retrieval_score"] = round(score, 4)
            item["retrieval"] = "hybrid" if vector else "lexical"
            scored.append((score, item))
        scored.sort(key=lambda item: (item[0], item[1]["updated_at"]), reverse=True)
        result = [item for _, item in scored[:limit]]
        ids = [item["id"] for item in result]
        if ids:
            with self.connect() as db:
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
            cur = db.execute(
                "UPDATE memories SET content=?,confidence=?,source='user-correction',updated_at=? WHERE id=? AND status='active'",
                (content.strip(), confidence, now(), memory_id),
            )
            if cur.rowcount:
                db.execute("DELETE FROM memory_vectors WHERE memory_id=?", (memory_id,))
            return cur.rowcount > 0

    def forget_memory(self, memory_id: int) -> bool:
        with self.connect() as db:
            db.execute("DELETE FROM memory_vectors WHERE memory_id=?", (memory_id,))
            cur = db.execute("UPDATE memories SET status='forgotten',updated_at=? WHERE id=? AND status='active'", (now(), memory_id))
            return cur.rowcount > 0

    def forget_all_memories(self) -> int:
        with self.connect() as db:
            db.execute("DELETE FROM memory_vectors")
            cur = db.execute("UPDATE memories SET status='forgotten',updated_at=? WHERE status='active'", (now(),))
            return cur.rowcount

    # -------------------------------------------------------------- vectors
    def set_memory_vector(self, memory_id: int, model: str, vector: Sequence[float]) -> None:
        payload = struct.pack(f"<{len(vector)}f", *[float(value) for value in vector])
        with self.connect() as db:
            db.execute(
                "INSERT INTO memory_vectors(memory_id,model,dim,vector,updated_at) VALUES(?,?,?,?,?) "
                "ON CONFLICT(memory_id) DO UPDATE SET model=excluded.model,dim=excluded.dim,"
                "vector=excluded.vector,updated_at=excluded.updated_at",
                (memory_id, model, len(vector), payload, now()),
            )

    def memory_vectors(self, model: str | None = None) -> dict[int, list[float]]:
        result: dict[int, list[float]] = {}
        with self.connect() as db:
            sql = "SELECT memory_id,dim,vector FROM memory_vectors"
            params: list[Any] = []
            if model:
                sql += " WHERE model=?"
                params.append(model)
            try:
                rows = db.execute(sql, params).fetchall()
            except sqlite3.OperationalError:
                return result
            for row in rows:
                dim = int(row["dim"])
                result[int(row["memory_id"])] = list(struct.unpack(f"<{dim}f", row["vector"]))
        return result

    def count_missing_vectors(self, model: str | None = None) -> int:
        with self.connect() as db:
            if model:
                row = db.execute(
                    "SELECT COUNT(*) FROM memories m LEFT JOIN memory_vectors v ON m.id=v.memory_id "
                    "WHERE m.status='active' AND (v.memory_id IS NULL OR v.model<>?)", (model,)
                ).fetchone()
            else:
                row = db.execute(
                    "SELECT COUNT(*) FROM memories m LEFT JOIN memory_vectors v ON m.id=v.memory_id "
                    "WHERE m.status='active' AND v.memory_id IS NULL"
                ).fetchone()
            return int(row[0])

    # --------------------------------------------------------------- identity
    def set_identity(self, key: str, value: str) -> None:
        if contains_credential(value):
            raise ValueError("Ara will not store credential values in identity settings")
        with self.connect() as db:
            db.execute(
                "INSERT INTO identity(key,value,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at",
                (key, value, now()),
            )

    def get_identity(self) -> dict[str, str]:
        defaults = {
            "temperament": "steady, curious, candid, and considerate",
            "values": "honesty, user ownership, privacy, and safe boundaries",
            "communication": "natural and context-aware; avoid filler and false certainty",
            "language": "match the user: standard English, Nigerian English, or Pidgin",
            "variety": "do not perform slang or assume a shared ethnicity, religion, or region",
        }
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

    # ------------------------------------------------------------------ goals
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

    # -------------------------------------------------------------- decisions
    def record_decision(self, context: str, options: list[str], selected: str, rationale: str) -> int:
        if any(contains_credential(item) for item in [context, selected, rationale, *options]):
            raise ValueError("Ara will not store credential values in decisions")
        with self.connect() as db:
            cur = db.execute("INSERT INTO decisions(context,options,selected,rationale,created_at) VALUES(?,?,?,?,?)", (context, json.dumps(options), selected, rationale, now()))
            return int(cur.lastrowid)

    def decisions(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]

    # ----------------------------------------------------------------- skills
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

    # ------------------------------------------------------------ tool audit
    def record_tool(self, name: str, input_value: Any, status: str, output: str) -> None:
        with self.connect() as db:
            safe_input = redact_sensitive(json.dumps(input_value, ensure_ascii=False, default=str))
            safe_output = redact_sensitive(output)
            db.execute("INSERT INTO tool_runs(tool_name,input_json,status,output,created_at) VALUES(?,?,?,?,?)", (name, safe_input, status, safe_output[:4000], now()))

    def tool_history(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM tool_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]

    # ------------------------------------------------------------------ news
    def save_news(self, items: Sequence[dict[str, Any]]) -> tuple[int, int]:
        """Insert new stories and refresh changed ones. Returns (added, updated)."""
        added = updated = 0
        stamp = now()
        with self.connect() as db:
            for item in items:
                key = str(item["dedupe_key"])
                existing = db.execute("SELECT id FROM news WHERE dedupe_key=?", (key,)).fetchone()
                values = (
                    str(item["title"]), str(item.get("summary", "")), str(item["link"]),
                    str(item["source"]), str(item.get("source_url", "")),
                    str(item["published_at"]), str(item.get("fetched_at", stamp)),
                    str(item.get("topic", "General")), str(item.get("origin", "feed")),
                    json.dumps(item.get("related_sources", []), ensure_ascii=False),
                )
                if existing:
                    db.execute(
                        "UPDATE news SET title=?,summary=?,link=?,source=?,source_url=?,published_at=?,fetched_at=?,topic=?,origin=?,related_sources=? WHERE id=?",
                        (*values, existing["id"]),
                    )
                    updated += 1
                else:
                    db.execute(
                        "INSERT INTO news(dedupe_key,title,summary,link,source,source_url,published_at,fetched_at,topic,origin,related_sources) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (key, *values),
                    )
                    added += 1
        return added, updated

    def news_items(self, limit: int = 40, topic: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as db:
            sql = "SELECT * FROM news"
            params: list[Any] = []
            if topic:
                sql += " WHERE topic=?"
                params.append(topic)
            sql += " ORDER BY published_at DESC, id DESC LIMIT ?"
            params.append(limit)
            rows = []
            for row in db.execute(sql, params).fetchall():
                item = dict(row)
                try:
                    item["related_sources"] = json.loads(item.get("related_sources") or "[]")
                except json.JSONDecodeError:
                    item["related_sources"] = []
                rows.append(item)
            return rows

    def news_topics(self) -> list[str]:
        with self.connect() as db:
            return [row[0] for row in db.execute("SELECT DISTINCT topic FROM news ORDER BY topic").fetchall()]

    def news_last_fetch(self) -> str | None:
        with self.connect() as db:
            row = db.execute("SELECT MAX(fetched_at) FROM news").fetchone()
            return str(row[0]) if row and row[0] else None

    def clear_news(self) -> int:
        with self.connect() as db:
            cur = db.execute("DELETE FROM news")
            return cur.rowcount

    # ----------------------------------------------------------- web sessions
    def create_web_session(self, token_hash: str, csrf: str, conversation_id: str, hours: int, label: str = "") -> None:
        created = now()
        expires = datetime.now(timezone.utc).timestamp() + hours * 3600
        expires_at = datetime.fromtimestamp(expires, timezone.utc).isoformat(timespec="seconds")
        with self.connect() as db:
            db.execute(
                "INSERT INTO web_sessions(token_hash,csrf,conversation_id,created_at,expires_at,last_seen,label) "
                "VALUES(?,?,?,?,?,?,?)",
                (token_hash, csrf, conversation_id, created, expires_at, created, label),
            )

    def web_session(self, token_hash: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM web_sessions WHERE token_hash=?", (token_hash,)).fetchone()
            if not row:
                return None
            item = dict(row)
            if item["expires_at"] <= now():
                db.execute("DELETE FROM web_sessions WHERE token_hash=?", (token_hash,))
                return None
            db.execute("UPDATE web_sessions SET last_seen=? WHERE token_hash=?", (now(), token_hash))
            return item

    def set_web_session_conversation(self, token_hash: str, conversation_id: str) -> bool:
        with self.connect() as db:
            cur = db.execute("UPDATE web_sessions SET conversation_id=? WHERE token_hash=?", (conversation_id, token_hash))
            return cur.rowcount > 0

    def delete_web_session(self, token_hash: str) -> bool:
        with self.connect() as db:
            return db.execute("DELETE FROM web_sessions WHERE token_hash=?", (token_hash,)).rowcount > 0

    def prune_web_sessions(self) -> int:
        with self.connect() as db:
            return db.execute("DELETE FROM web_sessions WHERE expires_at<=?", (now(),)).rowcount

    def log_web_event(self, session_hash: str, kind: str, detail: str = "") -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO web_events(session_hash,kind,detail,created_at) VALUES(?,?,?,?)",
                (session_hash[:32], kind, redact_sensitive(detail)[:500], now()),
            )

    def web_event_count(self, session_hash: str, kind: str, since: str) -> int:
        with self.connect() as db:
            row = db.execute(
                "SELECT COUNT(*) FROM web_events WHERE session_hash=? AND kind=? AND created_at>=?",
                (session_hash[:32], kind, since),
            ).fetchone()
            return int(row[0]) if row else 0

    # ----------------------------------------------------------------- totals
    def counts(self) -> dict[str, int]:
        with self.connect() as db:
            result = {}
            for table in ("messages", "memories", "goals", "skills", "news"):
                try:
                    result[table] = int(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                except sqlite3.OperationalError:
                    result[table] = 0
            return result

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
        except (sqlite3.DatabaseError, OSError, ValueError):
            return False
        try:
            result = db.execute("PRAGMA integrity_check").fetchone()
            if not result or result[0] != "ok":
                return False
            version = db.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
            present = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            return bool(version and 1 <= int(version[0]) <= SCHEMA_VERSION and CORE_TABLES <= present)
        except (sqlite3.DatabaseError, ValueError):
            return False
        finally:
            db.close()
