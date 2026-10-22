from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import textwrap
from datetime import datetime
from pathlib import Path

from . import __version__
from .brand import hop, static_frog
from .config import compatibility_report, effective_profile, load_config, write_default_config
from .downloads import MODELS, DownloadError, catalog_entry, download, human_size, probe
from .embeddings import backfill_vectors, get_embedder, retrieval_report
from .engine import BoundedActionLoop, ConversationEngine, evaluate_options, identify_learning, select_goal
from .model import ModelError, backend_summary, get_model, runtime_diagnostics
from .news import format_news, news_report
from .nigeria import KnowledgeBase, evaluate
from .nigeria_pipeline import ingest_jsonl, validate_corpus, write_manifest
from .perception import PerceptionError, PerceptionUnavailable, Transcriber, VisionClient, perception_status
from .safety import redact_sensitive
from .store import SCHEMA_VERSION, Store
from .tools import ToolError, ToolRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ara", description="Ara is a local-first conversational companion.")
    parser.add_argument("--conversation-id", help="Resume a specific saved conversation")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("status", help="Show local data and model status")
    sub.add_parser("doctor", help="Check the local installation and model connection")
    history = sub.add_parser("history", help="Show recent messages in the active conversation")
    history.add_argument("--limit", type=int, default=30)
    history.add_argument("--conversation-id", help="Show a specific conversation")
    sub.add_parser("conversations", help="List saved conversation threads")
    model = sub.add_parser("model", help="Inspect the backend and download optional local weights")
    model_sub = model.add_subparsers(dest="model_command", required=True)
    model_sub.add_parser("status", help="Check the active backend and the configured weights")
    download_model = model_sub.add_parser("download", help="Download optional local weights (resumable, checksum-verified)")
    download_model.add_argument("name", help="A catalog name (qwen3-1.7b, qwen3-4b) or a .gguf URL")
    download_model.add_argument("--out", help="Destination file (default: Ara's models folder)")
    download_model.add_argument("--sha256", help="Expected SHA-256 from the publisher")
    download_model.add_argument("--yes", action="store_true", help="Start without asking for confirmation")
    ask = sub.add_parser("ask", help="Send one message and print Ara's reply")
    ask.add_argument("message", nargs="+")
    ask.add_argument("--mode", choices=["chat", "learn"], default="chat")
    ask.add_argument("--json", action="store_true", help="Print a JSON result")
    nigeria = sub.add_parser("nigeria", help="Search and measure Ara's Nigerian-context knowledge")
    ng = nigeria.add_subparsers(dest="nigeria_command", required=True)
    ng_search = ng.add_parser("search"); ng_search.add_argument("query"); ng_search.add_argument("--limit", type=int, default=8)
    ng_explain = ng.add_parser("explain"); ng_explain.add_argument("text", nargs="+"); ng_explain.add_argument("--json", action="store_true")
    ng_eval = ng.add_parser("eval"); ng_eval.add_argument("--model", action="store_true", help="Also score actual model answers (slow, needs a running model)"); ng_eval.add_argument("--json", action="store_true")
    ng.add_parser("stats")
    ng.add_parser("validate")
    ng.add_parser("manifest")
    ng_ingest = ng.add_parser("ingest"); ng_ingest.add_argument("source"); ng_ingest.add_argument("--kind", choices=["expressions", "history", "culture"], required=True); ng_ingest.add_argument("--out", required=True); ng_ingest.add_argument("--license", required=True); ng_ingest.add_argument("--consent", required=True)
    news = sub.add_parser("news", help="Read Nigerian headlines from publisher feeds")
    news.add_argument("--refresh", action="store_true"); news.add_argument("--offline", action="store_true")
    news.add_argument("--topic", default=None); news.add_argument("--limit", type=int, default=12); news.add_argument("--json", action="store_true")
    web = sub.add_parser("web", help="Serve the shared-core web interface")
    web.add_argument("--host", default=None); web.add_argument("--port", type=int, default=None)
    animate = sub.add_parser("animate", help="Show the frog; animate in an interactive terminal")
    animate.add_argument("--loops", type=int, default=2); animate.add_argument("--plain", action="store_true")
    vision = sub.add_parser("vision", help="Analyse an image through the configured vision endpoint")
    vision.add_argument("path"); vision.add_argument("--prompt", default="Describe this image accurately and name any uncertainty.")
    audio = sub.add_parser("audio", help="Transcribe an audio file through the configured local endpoint")
    audio.add_argument("path")
    memory = sub.add_parser("memory", help="Inspect, add, correct, or forget stored memories")
    msub = memory.add_subparsers(dest="memory_command", required=True)
    msub.add_parser("inspect", help="List active memories")
    add = msub.add_parser("add", help="Store a memory explicitly")
    add.add_argument("content"); add.add_argument("--kind", default="fact")
    export = msub.add_parser("export", help="Export active memories as JSON")
    export.add_argument("destination", nargs="?", default="memories.json")
    correct = msub.add_parser("correct", help="Correct an existing memory")
    correct.add_argument("id", type=int); correct.add_argument("content")
    forget = msub.add_parser("forget", help="Forget an existing memory")
    forget.add_argument("id", type=int)
    msub.add_parser("embed", help="Compute vectors for memories that need them")
    mforget = msub.add_parser("forget-all", help="Forget every active memory")
    mforget.add_argument("--yes", action="store_true", help="Confirm forgetting every active memory")
    backup = sub.add_parser("backup", help="Create a consistent local SQLite backup")
    backup.add_argument("destination", nargs="?")
    restore = sub.add_parser("restore", help="Restore from an Ara SQLite backup")
    restore.add_argument("source")
    identity = sub.add_parser("identity", help="Inspect or update Ara’s persistent identity")
    isub = identity.add_subparsers(dest="identity_command", required=True)
    isub.add_parser("show")
    identity_set = isub.add_parser("set"); identity_set.add_argument("key"); identity_set.add_argument("value")
    goals = sub.add_parser("goal", help="Track active goals")
    gsub = goals.add_subparsers(dest="goal_command", required=True)
    ga = gsub.add_parser("add"); ga.add_argument("title"); ga.add_argument("--details", default=""); ga.add_argument("--priority", type=int, default=3)
    gsub.add_parser("list")
    gd = gsub.add_parser("status"); gd.add_argument("id", type=int); gd.add_argument("value", choices=["active", "done", "paused", "cancelled"])
    decide = sub.add_parser("decide", help="Compare explicit scored options and record the decision")
    decide.add_argument("--context", required=True); decide.add_argument("--option", action="append", required=True); decide.add_argument("--score", action="append", default=[], metavar="OPTION=NUMBER")
    sub.add_parser("autonomous", help="Run a bounded goal-directed action loop")
    decisions = sub.add_parser("decisions", help="Inspect recorded decisions")
    decisions.add_argument("--limit", type=int, default=20)
    tools = sub.add_parser("tools", help="List available local tools")
    tlist = tools.add_subparsers(dest="tools_command", required=True)
    tlist.add_parser("list")
    thistory = tlist.add_parser("history"); thistory.add_argument("--limit", type=int, default=50)
    tool = sub.add_parser("tool", help="Run a bounded local tool")
    tsub = tool.add_subparsers(dest="tool_command", required=True)
    calc = tsub.add_parser("calculate"); calc.add_argument("expression")
    read = tsub.add_parser("read-file"); read.add_argument("path")
    listing = tsub.add_parser("list-dir"); listing.add_argument("path")
    write = tsub.add_parser("write-file"); write.add_argument("path"); write.add_argument("content")
    write.add_argument("--yes", action="store_true", help="Confirm this specific file write")
    skill = sub.add_parser("skill", help="Inspect and manage documented reusable procedures")
    ssub = skill.add_subparsers(dest="skill_command", required=True)
    ssub.add_parser("list")
    sa = ssub.add_parser("add"); sa.add_argument("name"); sa.add_argument("description"); sa.add_argument("procedure_json", help='JSON list, e.g. \'[{"action":"..."}]\''); sa.add_argument("--conditions", default="")
    sshow = ssub.add_parser("show"); sshow.add_argument("name")
    sr = ssub.add_parser("remove"); sr.add_argument("name")
    srun = ssub.add_parser("run"); srun.add_argument("name"); srun.add_argument("--yes", action="store_true", help="Approve this skill execution")
    return parser


def _model(config):
    """Return the configured backend, or None for honest retrieval-only mode."""
    return get_model(config)


def _print_wrapped(text: str) -> None:
    width = max(40, min(100, shutil.get_terminal_size((80, 24)).columns - 2))
    for paragraph in text.splitlines() or [""]:
        print(textwrap.fill(paragraph, width=width, break_long_words=False, break_on_hyphens=False) if paragraph else "")


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def storage_lines(config) -> list[str]:
    """Show where disk goes so users can decide what to clear."""
    database = _size(config.db_path) + _size(Path(str(config.db_path) + "-wal"))
    caches = _size(config.news_cache_path) + _size(config.embedding_cache_path)
    return [
        f"Storage: database {human_size(database)}, backups {human_size(_size(config.backup_dir))}, "
        f"models {human_size(_size(config.local_models_dir))}, caches {human_size(caches)}",
        f"  data: {config.home} (local; 'ara backup' copies it, 'ara restore FILE' replaces it safely)",
    ]


def download_model(config, args) -> int:
    """Optional weights. Never called during install or a normal chat start."""
    if args.name in MODELS:
        entry = catalog_entry(args.name)
        url = str(entry["url"])
        label = str(entry["label"])
        hint_size = int(entry.get("bytes") or 0)
        ram = str(entry.get("ram") or "")
        licence = str(entry.get("license") or "check the publisher")
        note = str(entry.get("note") or "")
        default_name = url.rsplit("/", 1)[-1]
    elif args.name.startswith(("http://", "https://")):
        url = args.name
        label = url.rsplit("/", 1)[-1] or "model.gguf"
        hint_size = 0
        ram = "at least the file size plus context memory"
        licence = "check the publisher"
        note = "Downloaded from a URL you supplied; Ara has no catalogue entry for it."
        default_name = label.split("?", 1)[0]
    else:
        raise DownloadError(f"Unknown model '{args.name}'. Choose one of: {', '.join(sorted(MODELS))}, or pass a .gguf URL.")

    destination = Path(args.out).expanduser().resolve() if args.out else config.local_models_dir / default_name
    if destination.suffix.lower() != ".gguf":
        raise DownloadError("The destination must end in .gguf")
    print(f"Optional local model: {label}")
    print(f"  source:  {url}")
    print(f"  target:  {destination}")
    print("Checking size and checksum with the publisher...")
    info = probe(url)
    size = info["size"] if isinstance(info["size"], int) else hint_size or None
    print(f"  download: {human_size(size)} (server-reported)" if size else "  download: size not reported by the server")
    if size:
        print(f"  storage:  about {human_size(size * 1.05)} free where you save it")
    print(f"  hardware: {ram}; CPU inference, no GPU required")
    print(f"  licence:  {licence}")
    if note:
        print(f"  note:     {note}")
    print("This is optional. Ara chats without it; only offline local inference needs these weights.")
    if not args.yes:
        if not sys.stdin.isatty():
            print("Not downloading without confirmation. Re-run with --yes in a non-interactive shell.", file=sys.stderr)
            return 2
        if input("Type 'download' to start (anything else cancels): ").strip().lower() != "download":
            print("Download cancelled. Nothing was changed.")
            return 1
    last = [0.0]

    def progress(done: int, total: int) -> None:
        import time

        if total and done < total and time.monotonic() - last[0] < 0.2:
            return
        last[0] = time.monotonic()
        share = f" ({done * 100 // total}%)" if total else ""
        print(f"\r  {human_size(done)} / {human_size(total) if total else 'unknown'}{share}", end="", flush=True)

    try:
        result = download(url, destination, expected_size=size, expected_sha256=args.sha256, on_progress=progress)
    except KeyboardInterrupt:
        print("\nStopped. Re-run the same command to resume from the partial file.")
        return 130
    print()
    print(f"Saved {human_size(result['bytes'])} to {result['path']}")
    print(f"SHA-256: {result['sha256']}" + (" (verified against the publisher)" if result["checksum_verified"] else " (no publisher checksum available; recorded for your own check)"))
    print("To use it offline, start a llama.cpp server yourself:")
    print(f'  llama-server -m "{result["path"]}" --host 127.0.0.1 --port 8080 -c 4096')
    print("Then run 'ara model status'. Ara does not start or download servers for you.")
    return 0


def interactive(config, store: Store, conversation_id: str | None = None) -> int:
    try:
        model = _model(config)
    except ModelError as exc:
        print(f"Model configuration error: {exc}", file=sys.stderr)
        return 2
    engine = ConversationEngine(config, store, model, conversation_id)
    if sys.stdout.isatty():
        print(static_frog(color=True))
    print(f"Ara {__version__} · {backend_summary(config, model)}")
    print("Type your message and press Enter. /help lists commands, /exit leaves. End a line with \\ for a second line.")
    if model is None:
        print("No model is configured, so replies are retrieved notes labelled retrieval-only, never fake model text.")
    mode = "chat"
    while True:
        try:
            text = input("you> ").strip()
            while text.endswith("\\"):
                text = text[:-1].rstrip() + "\n" + input("...> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAra session ended.")
            return 0
        if not text:
            continue
        if text in {"/exit", "/quit"}:
            print("Ara session ended.")
            return 0
        if text in {"/help", "/?"}:
            print("/model  /memory  /history  /new  /clear  /skills  /remember TEXT")
            print("/news [refresh]  /explain TEXT  /image PATH  /audio PATH  /study  /animate")
            print("/calc EXPR  /embed  /doctor  /exit")
            continue
        if text == "/clear":
            if sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print("Screen cleared. Your history is still saved; /new starts a fresh thread.")
            continue
        if text == "/model":
            print(f"Backend: {engine.last_backend} · {backend_summary(config, model)}")
            if model is not None:
                ok, message = model.status()
                print(message)
                if not ok:
                    print("Replies will fall back to labelled retrieval until the model is reachable.")
            print("Hosted: set ARA_HOSTED_URL (and ARA_HOSTED_API_KEY if needed).")
            print("Local: run a llama.cpp server on 127.0.0.1:8080, or 'ara model download qwen3-1.7b'.")
            print("Retrieval-only: 'ara model' never downloads anything; use /model to check again.")
            continue
        if text == "/study":
            mode = "learn" if mode == "chat" else "chat"
            print("Study mode on. Ara will explain in steps and check understanding." if mode == "learn" else "Back to ordinary conversation.")
            continue
        if text == "/animate":
            hop(loops=1)
            continue
        if text == "/doctor":
            print(f"Database schema: {SCHEMA_VERSION}; knowledge entries: {len(engine.knowledge.entries)}")
            for line in compatibility_report(config):
                print(line)
            if model is not None:
                ok, message = model.status()
                print(message)
            print(f"Retrieval: {json.dumps(retrieval_report(store, engine.embedder))}")
            continue
        if text == "/embed":
            try:
                count = backfill_vectors(store, engine.embedder)
                print(f"Computed {count} new memory vector(s) with {engine.embedder.describe()}.")
            except Exception as exc:
                print(f"Embedding error: {exc}", file=sys.stderr)
            continue
        if text.startswith("/news"):
            refresh = "refresh" in text.casefold()
            report = news_report(store, config, refresh=refresh, knowledge=engine.knowledge)
            if report.get("errors"):
                print("Feed errors: " + "; ".join(report["errors"]), file=sys.stderr)
            if not report["items"]:
                print("No cached news. Run 'ara news --refresh' when online.")
            for item in report["items"][:8]:
                print(format_news(item))
                print()
            continue
        if text.startswith("/explain "):
            interpretation = engine.knowledge.interpret(text[9:].strip())
            print(f"Register: {interpretation.register}" + (" (ambiguous; ask for context)" if interpretation.clarify else ""))
            for entry in interpretation.detected:
                print(f"- {entry.term}: {entry.summary} [{', '.join(entry.regions)}; confidence {entry.confidence:.2f}]")
            for note in interpretation.guidance:
                print(f"  note: {note}")
            if not interpretation.detected:
                print("No specific expression detected. Try a term or a short sentence.")
            continue
        if text.startswith("/image "):
            parts = text[7:].strip().split(" ", 1)
            try:
                data = Path(parts[0]).expanduser().read_bytes()
                result = VisionClient(config).describe(data, parts[1] if len(parts) > 1 else "Describe this image accurately and name any uncertainty.")
                print(result["text"])
            except (OSError, PerceptionError) as exc:
                print(f"Vision error: {exc}", file=sys.stderr)
            continue
        if text.startswith("/audio "):
            try:
                data = Path(text[7:].strip()).expanduser().read_bytes()
                print(Transcriber(config).transcribe(data)["text"])
            except (OSError, PerceptionError) as exc:
                print(f"Transcription error: {exc}", file=sys.stderr)
            continue
        if text == "/new":
            engine.start_new_conversation()
            print("Started a new conversation. Earlier history remains stored.")
            continue
        if text == "/history":
            for message in store.recent_messages(engine.conversation_id, 30):
                print(f"{message['role']}> {message['content']}")
            continue
        if text == "/memory":
            for item in store.list_memories():
                print(f"#{item['id']} [{item['kind']}; confidence {item['confidence']:.2f}] {item['content']}")
            if not store.list_memories(): print("No stored memories.")
            continue
        if text == "/skills":
            skills = store.skills()
            for item in skills: print(f"{item['name']} v{item['version']}: {item['description']}")
            if not skills: print("No learned procedures yet.")
            continue
        if text.startswith("/calc "):
            try: print(ToolRegistry(config, store).run("calculate", {"expression": text[6:]}))
            except ToolError as exc: print(f"Tool error: {exc}", file=sys.stderr)
            continue
        if text.startswith("/remember "):
            raw_content = text[10:].strip()
            if redact_sensitive(raw_content) != raw_content:
                print("Credential-looking text was not stored; any assignment value is redacted from conversation history.")
                continue
            candidate = identify_learning("remember " + raw_content)
            content = candidate[1] if candidate else raw_content
            if not content:
                print("Usage: /remember the detail to save")
                continue
            mid, conflicts = store.add_memory("fact", content, "explicit-user-command", 1.0)
            print(f"Stored memory #{mid}." + (f" Possible related memories: {', '.join('#'+str(x['id']) for x in conflicts)}; inspect them for contradictions." if conflicts else ""))
            continue
        try:
            answer, learning = engine.respond(text, mode=mode)
            print("ara (retrieval-only)> " if engine.last_backend == "retrieval-only" else "ara> ", end="")
            _print_wrapped(answer)
            if engine.last_interpretation and engine.last_interpretation.detected:
                terms = ", ".join(entry.term for entry in engine.last_interpretation.detected[:3])
                print(f"[Context: {engine.last_interpretation.register} · {terms}]")
            if engine.last_interpretation and engine.last_interpretation.clarify:
                print("[The expression was ambiguous; ask for the missing context instead of assuming.]")
            if learning.saved_id:
                print(f"[Stored {learning.kind} as memory #{learning.saved_id}.]" + (" Related memories may conflict; use ara memory inspect." if learning.conflicts else ""))
            if learning.sensitive_rejected:
                print("[Credential-looking text was redacted from saved conversation history and was not stored as a memory.]")
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"Ara error: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config()
        if args.command is None:
            config.home.mkdir(parents=True, exist_ok=True)
            write_default_config(config)
            store = Store(config.db_path)
            return interactive(config, store, args.conversation_id)
        if args.command == "restore":
            source = Path(args.source).expanduser().resolve()
            if not Store.check_backup(source):
                print("Backup is invalid or failed SQLite integrity_check.", file=sys.stderr); return 2
            if not sys.stdin.isatty():
                print("Restore replaces the current Ara database. Run interactively and confirm when prompted.", file=sys.stderr); return 2
            answer = input(f"Replace {config.db_path} with {source}? Type 'restore' to confirm: ")
            if answer != "restore": print("Restore cancelled."); return 1
            config.home.mkdir(parents=True, exist_ok=True)
            # Restore is destructive: keep a verified copy of the current data first.
            if config.db_path.exists():
                config.backup_dir.mkdir(parents=True, exist_ok=True)
                safety = config.backup_dir / ("ara-pre-restore-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".sqlite3")
                try:
                    with sqlite3.connect(config.db_path) as live, sqlite3.connect(safety) as copy:
                        live.backup(copy)
                except (OSError, sqlite3.DatabaseError) as exc:
                    print(f"Could not back up the current database; restore cancelled: {exc}", file=sys.stderr)
                    return 2
                if not Store.check_backup(safety):
                    print("The pre-restore backup failed its integrity check; restore cancelled.", file=sys.stderr)
                    return 2
                print(f"Saved the current database to {safety}")
            temp_path = config.db_path.with_suffix(config.db_path.suffix + ".restore-tmp")
            shutil.copy2(source, temp_path)
            if not Store.check_backup(temp_path):
                temp_path.unlink(missing_ok=True)
                print("Restore copy failed its integrity check.", file=sys.stderr); return 2
            Path(str(config.db_path) + "-wal").unlink(missing_ok=True)
            Path(str(config.db_path) + "-shm").unlink(missing_ok=True)
            os.replace(temp_path, config.db_path)
            print(f"Restored Ara data from {source}")
            return 0
        store = Store(config.db_path)
        if args.command == "status":
            model = _model(config)
            ok, message = (model.status() if model is not None else (True, "No model configured; replies are labelled retrieval-only."))
            knowledge = KnowledgeBase.load(config.knowledge_dir)
            print(f"Ara {__version__}\nData: {config.home}\nDatabase: {config.db_path}\nSchema: {SCHEMA_VERSION}\nCounts: {json.dumps(store.counts())}\nKnowledge: {json.dumps(knowledge.stats()['by_kind'])}\nNews cached: {len(store.news_items(1))}\nProfile: {effective_profile(config)}\nBackend: {config.active_backend}\nModel: {message}")
            for line in storage_lines(config): print(line)
            return 0 if ok else 1
        if args.command == "doctor":
            print(f"Ara {__version__}; Python {sys.version.split()[0]}; database: OK ({config.db_path}); schema {SCHEMA_VERSION}")
            for line in runtime_diagnostics(config): print(line)
            for line in compatibility_report(config): print(line)
            validation = validate_corpus(config.knowledge_dir)
            print(f"Nigerian knowledge: {validation['total']} entries; valid={validation['ok']}; warnings={len(validation['warnings'])}")
            for problem in validation["errors"]:
                print(f"  error: {problem}")
            state = perception_status(config)
            print(f"Vision: {state['vision']['note']}")
            print(f"Transcription: {state['transcription']['note']}")
            print(f"Web password: {'set' if config.web_password else 'not set (local-only binding)'}")
            print(f"News cache: {len(store.news_items(1))} stories")
            for line in storage_lines(config): print(line)
            model = _model(config)
            if model is None:
                print("No model configured; retrieval-only is available.")
                return 0
            ok, message = model.status(); print(message)
            return 0 if ok else 1
        if args.command == "model":
            if args.model_command == "download":
                return download_model(config, args)
            for line in runtime_diagnostics(config): print(line)
            for line in compatibility_report(config): print(line)
            model = _model(config)
            if model is None:
                print("Active backend: retrieval-only (no model configured).")
                print("Hosted: set ARA_HOSTED_URL (and ARA_HOSTED_API_KEY).")
                print("Local: 'ara model download qwen3-1.7b' then run a llama.cpp server, or point [model].url at one.")
                return 0
            print(json.dumps(model.capabilities(), indent=2))
            ok, message = model.status(); print(message)
            return 0 if ok else 1
        if args.command == "ask":
            model = _model(config)
            engine = ConversationEngine(config, store, model, args.conversation_id)
            answer, learning = engine.respond(" ".join(args.message), mode=args.mode)
            if args.json:
                print(json.dumps({
                    "answer": answer,
                    "backend": engine.last_backend,
                    "backend_error": engine.last_backend_error,
                    "conversation_id": engine.conversation_id,
                    "learning": {"saved": bool(learning.saved_id), "kind": learning.kind, "id": learning.saved_id},
                    "interpretation": engine.last_interpretation.as_dict() if engine.last_interpretation else None,
                }, ensure_ascii=False, indent=2))
            else:
                print(answer)
            return 0
        if args.command == "nigeria":
            knowledge = KnowledgeBase.load(config.knowledge_dir)
            if args.nigeria_command == "search":
                for entry in knowledge.search(args.query, limit=args.limit):
                    if entry.kind == "expressions":
                        print(f"{entry.term} [{entry.register}; {', '.join(entry.regions)}; confidence {entry.confidence:.2f}]: {entry.summary}")
                    else:
                        print(f"{entry.title} [{entry.kind}; confidence {entry.confidence:.2f}]: {entry.summary}")
                return 0
            if args.nigeria_command == "explain":
                interpretation = knowledge.interpret(" ".join(args.text))
                payload = interpretation.as_dict()
                payload["context"] = knowledge.context_for(" ".join(args.text))
                if args.json:
                    print(json.dumps(payload, ensure_ascii=False, indent=2))
                else:
                    print(f"Register: {interpretation.register}" + (" · ambiguous, clarify" if interpretation.clarify else ""))
                    for entry in interpretation.detected:
                        print(f"- {entry.term}: {entry.summary}")
                        print(f"  regions: {', '.join(entry.regions)}; languages: {', '.join(entry.languages)}; confidence {entry.confidence:.2f}")
                        for misreading in entry.misreadings:
                            print(f"  avoid: {misreading}")
                    for note in interpretation.guidance:
                        print(f"note: {note}")
                    if not interpretation.detected:
                        print("No specific expression detected.")
                return 0
            if args.nigeria_command == "eval":
                answer_fn = None
                if args.model:
                    model = _model(config)
                    if model is None:
                        print("No model is configured, so --model cannot score generated answers. Run the offline evaluation without it.", file=sys.stderr)
                        return 2
                    engine = ConversationEngine(config, store, model, conversation_id=f"evaluation-{datetime.now().strftime('%Y%m%d%H%M%S')}", knowledge=knowledge, activate=False)
                    answer_fn = lambda question: model.generate(engine.preview_prompt(question))
                report = evaluate(knowledge, answer_fn)
                if args.json:
                    print(json.dumps(report, ensure_ascii=False, indent=2))
                else:
                    print(f"Retrieval: {report['retrieval']['hits']}/{report['retrieval']['cases']} ({report['retrieval']['accuracy']:.1%})")
                    for failure in report["retrieval"]["failures"]:
                        print(f"  miss {failure['id']}: expected {failure['expected']}, got {failure['got']}")
                    print(f"Interpretation: {report['interpretation']['hits']}/{report['interpretation']['cases']} ({report['interpretation']['accuracy']:.1%})")
                    for failure in report["interpretation"]["failures"]:
                        print(f"  fail {failure['id']}: {failure['problems']}")
                    model_report = report["model"]
                    if model_report.get("run"):
                        print(f"Model keyword rubric: {model_report['passed']}/{model_report['cases']} ({model_report['accuracy']:.1%}) — not human evaluation")
                        for category, values in sorted(model_report["by_category"].items()):
                            print(f"  {category}: {values['passed']}/{values['cases']}")
                        for failure in model_report["failures"][:10]:
                            print(f"  fail {failure['id']}: {failure['reason']}")
                    else:
                        print("Model answers: not scored (run with --model and a running local model).")
                return 0
            if args.nigeria_command == "stats":
                print(json.dumps(knowledge.stats(), indent=2))
                print(f"Provenance: {knowledge.metadata.get('expressions', {}).get('provenance', '')}")
                return 0
            if args.nigeria_command == "validate":
                report = validate_corpus(config.knowledge_dir)
                print(json.dumps(report, indent=2))
                return 0 if report["ok"] else 1
            if args.nigeria_command == "manifest":
                print(write_manifest(config.knowledge_dir))
                return 0
            if args.nigeria_command == "ingest":
                report = ingest_jsonl(args.source, args.kind, args.out, args.license, args.consent)
                print(json.dumps(report, indent=2))
                return 0 if report["accepted"] else 1
        if args.command == "news":
            report = news_report(store, config, refresh=args.refresh, offline=args.offline, topic=args.topic, knowledge=KnowledgeBase.load(config.knowledge_dir))
            if args.json:
                print(json.dumps(report, ensure_ascii=False, indent=2))
                return 0
            for error in report.get("errors", []):
                print(f"feed warning: {error}", file=sys.stderr)
            if report.get("offline"):
                print("Showing cached stories only; nothing was refreshed.")
            items = report["items"][: args.limit]
            if not items:
                print("No news cached. Run 'ara news --refresh' when online.")
            for item in items:
                print(format_news(item))
                print()
            return 0
        if args.command == "web":
            from .web.server import serve
            return serve(config, args.host, args.port)
        if args.command == "animate":
            if args.plain or not sys.stdout.isatty():
                print(static_frog())
            else:
                hop(loops=args.loops)
            return 0
        if args.command == "vision":
            data = Path(args.path).expanduser().read_bytes()
            result = VisionClient(config).describe(data, args.prompt)
            print(result["text"])
            print(f"\n[image: {result['image']['width']}x{result['image']['height']} {result['image']['mime']}; not stored]", file=sys.stderr)
            return 0
        if args.command == "audio":
            data = Path(args.path).expanduser().read_bytes()
            print(Transcriber(config).transcribe(data)["text"])
            return 0
        if args.command == "history":
            if args.limit < 1 or args.limit > 1000:
                print("History limit must be between 1 and 1000.", file=sys.stderr); return 2
            for message in store.recent_messages(args.conversation_id or store.active_conversation_id(), args.limit):
                print(f"{message['role']}> {message['content']}")
            return 0
        if args.command == "conversations":
            rows = store.conversations()
            active = store.active_conversation_id()
            for row in rows: print(f"{'*' if row['conversation_id'] == active else ' '} {row['conversation_id']} ({row['message_count']} messages; last updated {row['updated_at']})")
            if not rows: print("No saved conversations yet.")
            return 0
        if args.command == "memory":
            if args.memory_command == "inspect":
                rows = store.list_memories()
                for m in rows: print(f"#{m['id']} [{m['kind']}; {m['source']}; confidence {m['confidence']:.2f}] {m['content']}")
                if not rows: print("No stored memories.")
            elif args.memory_command == "add":
                mid, conflicts = store.add_memory(args.kind, args.content, "explicit-cli-entry", 1.0)
                print(f"Stored memory #{mid}." + (f" Related memories to review: {', '.join('#'+str(x['id']) for x in conflicts)}" if conflicts else ""))
            elif args.memory_command == "export":
                destination = Path(args.destination).expanduser().resolve()
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(store.list_memories(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(f"Exported {len(store.list_memories())} active memories to {destination}")
            elif args.memory_command == "correct":
                if not store.correct_memory(args.id, args.content): print("No active memory with that id.", file=sys.stderr); return 1
                print(f"Corrected memory #{args.id}.")
            elif args.memory_command == "forget":
                if not store.forget_memory(args.id): print("No active memory with that id.", file=sys.stderr); return 1
                print(f"Forgot memory #{args.id}.")
            elif args.memory_command == "embed":
                embedder = get_embedder(config)
                count = backfill_vectors(store, embedder)
                print(f"Computed {count} vector(s) with {embedder.describe()}.")
                print(json.dumps(retrieval_report(store, embedder), indent=2))
            elif args.memory_command == "forget-all":
                if not args.yes:
                    print("This marks every active memory forgotten. Re-run with --yes to confirm.", file=sys.stderr)
                    return 2
                print(f"Forgot {store.forget_all_memories()} active memories.")
            return 0
        if args.command == "backup":
            target = Path(args.destination).expanduser() if args.destination else config.backup_dir / ("ara-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".sqlite3")
            print(store.backup_to(target)); return 0
        if args.command == "identity":
            if args.identity_command == "show":
                for key, value in store.get_identity().items(): print(f"{key}: {value}")
            else:
                store.set_identity(args.key, args.value); print(f"Updated identity field '{args.key}'.")
            return 0
        if args.command == "goal":
            if args.goal_command == "add":
                gid = store.add_goal(args.title, args.details, args.priority); print(f"Added goal #{gid}.")
            elif args.goal_command == "list":
                goals = store.goals()
                for goal in goals: print(f"#{goal['id']} [priority {goal['priority']}/5] {goal['title']} — {goal['details']}")
                if not goals: print("No active goals.")
            else:
                if not store.set_goal_status(args.id, args.value): print("No goal with that id.", file=sys.stderr); return 1
                print(f"Goal #{args.id} is now {args.value}.")
            return 0
        if args.command == "decide":
            scores = {}
            for item in args.score:
                try: key, raw = item.rsplit("=", 1); scores[key] = float(raw)
                except ValueError: parser.error(f"Invalid score {item!r}; use OPTION=NUMBER")
            selected, rationale = evaluate_options(args.option, scores)
            store.record_decision(args.context, args.option, selected, rationale)
            print(f"Selected: {selected}\n{rationale}")
            return 0
        if args.command == "decisions":
            if not 1 <= args.limit <= 1000: print("Limit must be between 1 and 1000.", file=sys.stderr); return 2
            for row in store.decisions(args.limit): print(f"#{row['id']} {row['context']} -> {row['selected']}\n  Options: {row['options']}\n  {row['rationale']}")
            return 0
        if args.command == "tools":
            if args.tools_command == "list":
                for name, schema in ToolRegistry(config, store).schemas.items(): print(f"{name}: {schema['description']} [{schema['permission']}]")
            else:
                if not 1 <= args.limit <= 1000: print("Limit must be between 1 and 1000.", file=sys.stderr); return 2
                for row in store.tool_history(args.limit): print(f"#{row['id']} {row['created_at']} {row['tool_name']} [{row['status']}] {row['output']}")
            return 0
        if args.command == "tool":
            registry = ToolRegistry(config, store)
            if args.tool_command == "calculate": name, inputs = "calculate", {"expression": args.expression}
            elif args.tool_command == "read-file": name, inputs = "read_file", {"path": args.path}
            elif args.tool_command == "list-dir": name, inputs = "list_directory", {"path": args.path}
            else:
                name, inputs = "write_file", {"path": args.path, "content": args.content}
                if not args.yes:
                    if not sys.stdin.isatty(): print("Write denied: use --yes to approve this exact file write.", file=sys.stderr); return 2
                    print(f"Ara will create {Path(args.path).expanduser().resolve()} with {len(args.content)} characters; existing files are never overwritten.")
                    if input("Type 'write' to approve: ") != "write": print("Write cancelled."); return 1
            print(registry.run(name, inputs, approved=(name != "write_file" or args.yes or sys.stdin.isatty())))
            return 0
        if args.command == "autonomous":
            if not sys.stdin.isatty():
                print("autonomous needs an interactive terminal for approvals.", file=sys.stderr)
                return 2
            model = _model(config)
            if model is None:
                print("The autonomous loop needs a model backend. Start one or configure hosted ara-ng first.")
                return 2
            engine = ConversationEngine(config, store, model)
            loop = BoundedActionLoop(config, store, model, ToolRegistry(config, store))
            goals = store.goals("active")
            if not goals:
                print("No active goals. Add one with ara goal add.")
                return 1
            goal = select_goal(goals)
            print(f"Active goal: {goal['title']} (priority {goal['priority']}/5)")
            print("This is a bounded loop. Ara will propose steps, ask before running anything that modifies files, and stop when the goal is done or stuck.")
            print("Type 'plan' to propose steps, 'run' to execute the next approved step, 'approve' to allow the next file-modifying step, 'status' to see progress, 'stop' to quit.")
            while True:
                try:
                    text = input("autonomous> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nAutonomous loop stopped.")
                    return 0
                if text in {"stop", "exit", "/quit"}:
                    print("Autonomous loop stopped.")
                    return 0
                if text == "plan":
                    idea = engine.respond(f"I have an active goal: {goal['title']}. Suggest 2-4 concrete read-only or reversible steps to make progress on it. Return each step as ACTION: <action label> | TOOL: <tool name> | ARGS: <json> | REASON: <why> | PERMISSION: read-only or reversible. Do not suggest destructive actions.")[0]
                    print("Ara's suggested plan:")
                    print(idea)
                    continue
                if text == "status":
                    if not loop.log:
                        print("No steps executed yet.")
                        continue
                    for entry in loop.log[-5:]:
                        print(f"-{entry.get('name','?')} [{entry.get('status','?')}] {entry.get('result') or entry.get('error') or ''}")
                    continue
                if text in {"run", "next"}:
                    step = loop.execute_next(approved=False)
                    if step is None:
                        if loop.plan_done():
                            print("All steps completed.")
                        else:
                            print("No pending steps. Use 'plan' first.")
                    else:
                        print(f"Step '{step.name}': {step.status}" + (f" -> {step.result}" if step.result else f" - {step.error}"))
                    continue
                if text == "approve":
                    step = loop.execute_next(approved=True)
                    if step is None:
                        print("No pending steps need approval.")
                    else:
                        print(f"Approved step '{step.name}': {step.status}" + (f" -> {step.result}" if step.result else f" - {step.error}"))
                    continue
                if text.startswith("/calc "):
                    try:
                        print(ToolRegistry(config, store).run("calculate", {"expression": text[6:]}))
                    except ToolError as exc:
                        print(f"Tool error: {exc}", file=sys.stderr)
                    continue
                print("Commands: plan, run, approve, status, stop, /calc EXPR")
            return 0
        parser.print_help(); return 0
    except (ValueError, RuntimeError, OSError, ToolError, ModelError, DownloadError) as exc:
        print(f"ara: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
