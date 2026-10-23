# Architecture

Ara is one web-first application: the browser interface is primary, and the terminal (`ara chat`) is a secondary maintenance interface. Both construct a `ConversationEngine` over the same `Store`, `KnowledgeBase`, model adapter, memory retriever, safety functions, and configuration. There is no second prompt stack in the browser and no separate web database.

## Module map

| Module | Responsibility |
| --- | --- |
| `ara/config.py` | TOML loading, `ARA_*` environment overrides, validation, hardware profiles |
| `ara/store.py` | SQLite schema, versioned migrations, conversations, memories, news, web sessions |
| `ara/engine.py` | Prompt assembly, conservative learning, memory retrieval, streaming, bounded action loop |
| `ara/model.py` | `ModelAdapter`, OpenAI-compatible hosted and local clients, backend selection, context fitting, timeout/streaming |
| `ara/downloads.py` | Optional resumable, checksum-verified local weight downloads |
| `ara/nigeria.py` | Curated knowledge loading, interpretation, retrieval, evaluation |
| `ara/nigeria_pipeline.py` | Corpus validation, manifest hashing, guarded JSONL ingestion |
| `ara/embeddings.py` | Local embedding client, deterministic hashed fallback, hybrid retrieval scoring |
| `ara/news.py` | Publisher RSS parsing, provenance, topic classification, dedupe, cache |
| `ara/perception.py` | Image validation, vision requests, audio validation, transcription requests |
| `ara/tools.py` | Restricted file tools, safe arithmetic, skill execution |
| `ara/safety.py` | Credential-assignment redaction |
| `ara/brand.py` | ASCII frog, terminal hop animation, fallback rules |
| `ara/cli.py` | Maintenance commands; plain `ara` serves the web app, `ara chat` opens the terminal chat |
| `ara/web/server.py` | HTTP/SSE server, auth, CSRF, sessions, upload limits, static assets |
| `ara/web/static/` | HTML, CSS, original frog SVG, browser client |
| `ara/training/` | Tokenizer, dataset preparation, CPU baseline, checkpoints, export, model card |

## Request flow

### Terminal (`ara chat`)

1. `ara chat` loads `Config`, opens `Store` (migrating if needed), and calls `get_model`: hosted ara-ng when configured, else the local server, else no model.
2. `ConversationEngine.respond` or `stream_response` saves the user message, detects explicitly signalled learning, and retrieves relevant memories. If no model is reachable, it returns a reply that starts with `[retrieval-only]` and records that backend, instead of pretending retrieval is generated text.
3. `KnowledgeBase.interpret` and `context_for` add only relevant Nigerian-context entries; news context is added when the message clearly asks about current events and a cache exists.
4. The system prompt, bounded history, memories, corrections, and user turn go through the adapter.
5. The answer is saved, learning is applied, and the interactive loop continues. Cancellation stops streaming without deleting the stored turn.

### Web

1. `ara web` starts `AraServer`, which owns one `Store`, `KnowledgeBase`, and embedder.
2. `POST /api/session` creates a random session token and stores only its SHA-256 hash in `web_sessions`, with a CSRF token and a per-session conversation ID.
3. `POST /api/chat` validates the cookie, CSRF header, origin, body size, and rate limit. It then builds the same `ConversationEngine` with `activate=False` so a web session cannot hijack the terminal's active conversation. An unreachable model does not fail the request: the stream carries a labelled retrieval reply and the `done` event includes `backend` and `answer_kind`.
4. The engine streams text chunks as Server-Sent Events (`start`, `delta`, `done`, `error`). The final event carries the stored learning result and interpretation.
5. Uploads (`/api/vision`, `/api/transcribe`) are validated as data URLs, size-limited, held in memory, sent to the configured local endpoint, and never written to SQLite.
6. Memory endpoints operate on the same shared memory table with per-request validation and CSRF.

## Storage

`SCHEMA_VERSION = 2`.

Core tables from the original application:

- `messages`: conversation ID, role, content, timestamp.
- `memories`: kind, content, source, confidence, status, evidence, access count, timestamps, plus v2 topic/language/region/source URL fields.
- `identity`: key/value profile fields.
- `goals`, `decisions`, `skills`, `tool_runs`: durable planning and audited tool execution.

Added in v2:

- `memory_vectors`: memory ID and embedding blob (JSON text) for hybrid retrieval.
- `news`: publisher story fields, dedupe key, publication/fetch timestamps, related sources.
- `web_sessions`: hashed token, CSRF token, conversation ID, expiry, label.
- `web_events`: hashed session, event type, short detail, timestamp.

SQLite uses WAL mode and foreign keys where applicable. Memory deletion marks a row forgotten, removes its vector, and excludes it from prompts and listings. Conversation messages remain until the user removes the database or uses a future deletion command; they are never uploaded by Ara.

## Migrations and recovery

`Store` upgrades additive schemas without touching existing rows:

1. Read `schema_version`.
2. Refuse a database newer than the running code.
3. Copy the database to `backups/ara-pre-upgrade-{timestamp}.sqlite3` using SQLite's backup API.
4. Add v2 columns only when missing, create new tables with `IF NOT EXISTS`, and update the version.
5. A failed statement raises and is reported; the pre-upgrade backup remains available for recovery.

Closing the database on every operation keeps each public write atomic. Schema changes are additive, so an older Ara can still read the core tables even though it will refuse a newer schema version.

`Store.check_backup` requires a valid `schema_version`, the core tables, and `PRAGMA integrity_check`. `ara restore` copies to a temporary file, verifies it, removes stale WAL/SHM files, and replaces the live database only after interactive confirmation. Before replacing anything it writes a verified `ara-pre-restore-{timestamp}.sqlite3` backup of the current database and cancels if that backup fails.

## Model boundary

`ModelAdapter` is the only interface the engine needs. `HostedChatModel` and `LocalChatModel` share one OpenAI chat-completions client; the hosted one requires HTTPS (except loopback), reads its key from `ARA_HOSTED_API_KEY`, and sends the model name `ara-ng`. The configured model name is a label sent to the server; Ara's identity does not depend on it. Capability discovery reads `/models` and, where available, llama.cpp `/props`. Vision and transcription are separate endpoints because a text model must not be assumed to see images or hear audio.

`fit_messages` keeps the newest turns within `runtime.max_context_chars`, preserving one system prompt. It truncates a single oversized message from the left. This is a character budget, not an exact tokenizer count.

## Nigerian knowledge

`KnowledgeBase.load` loads built-in JSON from `ara/data/nigeria/` and optionally one extra directory. Entries are validated for required fields, confidence, regions, and duplicates. Retrieval is lexical with normalisation and simple stemming; interpretation adds register detection, ambiguity flags, stereotype warnings, and explicit misreadings to avoid.

`KnowledgeBase.interpret` is deliberately not a translator. It returns evidence and a clarification request when the same phrase has multiple plausible social meanings.

## Testing strategy

Tests use local `ThreadingHTTPServer` stubs for the model, vision, transcription, embeddings, and news. This exercises real HTTP encoding, streaming, timeout, and error paths without pretending a local LLM is running. Unit tests cover the engine, storage, corpus, training, and browser asset invariants. Hardware-dependent paths are documented as unverified rather than mocked and called complete.
