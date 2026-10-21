from __future__ import annotations

import argparse
import json
import shutil
import sys
import textwrap
from datetime import datetime
from pathlib import Path

from . import __version__
from .config import load_config, write_default_config
from .engine import ConversationEngine, evaluate_options, identify_learning
from .model import LocalChatModel, ModelError, runtime_diagnostics
from .safety import redact_sensitive
from .store import Store
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
    model = sub.add_parser("model", help="Inspect the configured local model")
    model.add_subparsers(dest="model_command", required=True).add_parser("status", help="Check model server and configured weights")
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
    return parser


def _model(config):
    return LocalChatModel(config)


def _print_wrapped(text: str) -> None:
    width = max(40, min(100, shutil.get_terminal_size((80, 24)).columns - 2))
    for paragraph in text.splitlines() or [""]:
        print(textwrap.fill(paragraph, width=width, break_long_words=False, break_on_hyphens=False) if paragraph else "")


def interactive(config, store: Store, conversation_id: str | None = None) -> int:
    try:
        model = _model(config)
    except ModelError as exc:
        print(f"Model configuration error: {exc}", file=sys.stderr)
        return 2
    engine = ConversationEngine(config, store, model, conversation_id)
    print("Ara is ready. Type /help for local commands, or /exit to leave.")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAra session ended.")
            return 0
        if not text:
            continue
        if text in {"/exit", "/quit"}:
            print("Ara session ended.")
            return 0
        if text == "/help":
            print("/memory  /history  /new  /skills  /calc EXPR  /remember TEXT  /exit")
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
            answer, learning = engine.respond(text)
            print("ara> ", end="")
            _print_wrapped(answer)
            if learning.saved_id:
                print(f"[Stored {learning.kind} as memory #{learning.saved_id}.]" + (" Related memories may conflict; use ara memory inspect." if learning.conflicts else ""))
            if learning.sensitive_rejected:
                print("[Credential-looking text was redacted from saved conversation history and was not stored as a memory.]")
        except ModelError as exc:
            print(f"ara> I could not reach the configured local model. Your message is in this conversation history. {exc}", file=sys.stderr)
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
            import shutil
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
            ok, message = _model(config).status()
            print(f"Ara {__version__}\nData: {config.home}\nDatabase: {config.db_path}\nCounts: {json.dumps(store.counts())}\nModel: {message}")
            return 0 if ok else 1
        if args.command == "doctor":
            print(f"Ara {__version__}; Python {sys.version.split()[0]}; database: OK ({config.db_path})")
            for line in runtime_diagnostics(config): print(line)
            ok, message = _model(config).status(); print(message)
            return 0 if ok else 1
        if args.command == "model":
            for line in runtime_diagnostics(config): print(line)
            ok, message = _model(config).status(); print(message)
            return 0 if ok else 1
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
        if args.command == "skill":
            if args.skill_command == "list":
                rows = store.skills()
                for skill in rows: print(f"{skill['name']} v{skill['version']}: {skill['description']} ({len(skill['procedure'])} steps)")
                if not rows: print("No documented procedures.")
            elif args.skill_command == "add":
                try: steps = json.loads(args.procedure_json)
                except json.JSONDecodeError as exc: print(f"Invalid procedure JSON: {exc}", file=sys.stderr); return 2
                sid = store.save_skill(args.name, args.description, steps, args.conditions); print(f"Saved skill version as record #{sid}.")
            elif args.skill_command == "show":
                matches = [item for item in store.skills() if item["name"] == args.name]
                if not matches: print("No skill with that name.", file=sys.stderr); return 1
                item = matches[0]
                print(f"{item['name']} v{item['version']}: {item['description']}\nConditions: {item['conditions'] or '(none documented)'}\nProcedure:")
                for index, step in enumerate(item["procedure"], 1): print(f"{index}. {step}")
            else:
                if not store.remove_skill(args.name): print("No skill with that name.", file=sys.stderr); return 1
                print(f"Removed all versions of skill '{args.name}'.")
            return 0
        parser.print_help(); return 0
    except (ValueError, RuntimeError, OSError, ToolError, ModelError) as exc:
        print(f"ara: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
