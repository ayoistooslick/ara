from __future__ import annotations

import ast
import math
import operator
from pathlib import Path
from typing import Any, Callable

from .config import Config
from .store import Store


class ToolError(RuntimeError):
    pass


_BINOPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def calculate(expression: str) -> int | float:
    if len(expression) > 200:
        raise ToolError("Expression is too long")
    try:
        tree = ast.parse(expression, mode="eval")
        def visit(node: ast.AST) -> int | float:
            if isinstance(node, ast.Expression):
                return visit(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                if not math.isfinite(node.value):
                    raise ToolError("Non-finite numbers are not allowed")
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
                left, right = visit(node.left), visit(node.right)
                if isinstance(node.op, ast.Pow) and abs(right) > 100:
                    raise ToolError("Exponent is too large")
                value = _BINOPS[type(node.op)](left, right)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 10**100:
                    raise ToolError("Result is too large")
                return value
            if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
                return _UNARY[type(node.op)](visit(node.operand))
            raise ToolError("Only numeric literals and basic arithmetic operators are allowed")
        return visit(tree)
    except (SyntaxError, ZeroDivisionError, OverflowError) as exc:
        raise ToolError(f"Invalid calculation: {exc}") from exc


class SkillExecutionError(RuntimeError):
    pass


class ToolRegistry:
    def __init__(self, config: Config, store: Store):
        self.config, self.store = config, store
        self.schemas = {
            "calculate": {"description": "Evaluate basic arithmetic; no code execution", "permission": "read-only"},
            "read_file": {"description": "Read a UTF-8 text file under an allowed root", "permission": "read-only"},
            "list_directory": {"description": "List entries under an allowed root", "permission": "read-only"},
            "write_file": {"description": "Create or replace a text file under an allowed root", "permission": "reversible; explicit approval required"},
            "run_skill": {"description": "Execute a documented reusable procedure step by step", "permission": "reversible; explicit approval required"},
        }

    def _resolve(self, path: str) -> Path:
        target = Path(path).expanduser().resolve()
        roots = self.config.allowed_roots or (Path.cwd().resolve(),)
        if not any(target == root or root in target.parents for root in roots):
            raise ToolError(f"Path is outside configured allowed roots: {target}")
        return target

    def run(self, name: str, arguments: dict[str, Any], approved: bool = False) -> str:
        try:
            if name == "calculate":
                result = str(calculate(str(arguments["expression"])))
            elif name == "read_file":
                path = self._resolve(str(arguments["path"]))
                if not path.is_file():
                    raise ToolError("File does not exist or is not a regular file")
                if path.stat().st_size > 1_000_000:
                    raise ToolError("Refusing to read files larger than 1 MB")
                result = path.read_text(encoding="utf-8")
            elif name == "list_directory":
                path = self._resolve(str(arguments["path"]))
                if not path.is_dir():
                    raise ToolError("Directory does not exist")
                result = "\n".join(sorted(entry.name + ("/" if entry.is_dir() else "") for entry in path.iterdir())[:500])
            elif name == "write_file":
                if not approved:
                    raise ToolError("This write needs explicit approval. Ara did not change the file.")
                path = self._resolve(str(arguments["path"]))
                if path.exists():
                    raise ToolError("Refusing to overwrite an existing file; choose a new path so the change is reversible")
                content = str(arguments["content"])
                if len(content.encode("utf-8")) > 1_000_000:
                    raise ToolError("Refusing to write more than 1 MB")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                result = f"Wrote {len(content)} characters to {path}"
            elif name == "run_skill":
                result = self._run_skill(str(arguments["skill_name"]), approved)
            else:
                raise ToolError(f"Unknown tool: {name}")
            audit_arguments = arguments
            audit_result = result
            if name == "read_file":
                audit_arguments = {"path": arguments.get("path")}
                audit_result = f"Read {len(result)} characters; file contents omitted from the audit log."
            elif name == "list_directory":
                audit_arguments = {"path": arguments.get("path")}
                audit_result = f"Listed {len(result.splitlines()) if result else 0} entries; names omitted from the audit log."
            elif name == "write_file":
                audit_arguments = {"path": arguments.get("path"), "content_length": len(str(arguments.get("content", "")))}
                audit_result = result
            elif name == "run_skill":
                audit_arguments = {"skill_name": arguments.get("skill_name"), "approved": approved}
            self.store.record_tool(name, audit_arguments, "success", audit_result)
            return result
        except Exception as exc:
            audit_arguments = arguments
            if name == "write_file":
                audit_arguments = {"path": arguments.get("path"), "content_length": len(str(arguments.get("content", "")))}
            elif name in {"read_file", "list_directory"}:
                audit_arguments = {"path": arguments.get("path")}
            elif name == "run_skill":
                audit_arguments = {"skill_name": arguments.get("skill_name"), "approved": approved}
            self.store.record_tool(name, audit_arguments, "error", str(exc))
            if isinstance(exc, ToolError):
                raise
            raise ToolError(str(exc)) from exc

    def _run_skill(self, skill_name: str, approved: bool) -> str:
        """Execute a documented procedure step by step.

        Each step must declare an action that maps to a registered tool. Steps are run
        in order; the first failure stops the procedure. This does not grant the skill
        any permissions beyond what the caller already approved.
        """
        if not approved:
            raise ToolError("Running a skill modifies state and needs explicit approval.")
        matches = [item for item in self.store.skills() if item["name"] == skill_name]
        if not matches:
            raise ToolError(f"No skill named {skill_name!r}")
        skill = matches[-1]
        procedure = skill["procedure"]
        if not procedure:
            raise ToolError(f"Skill {skill_name!r} has no documented steps")
        results: list[str] = []
        for index, step in enumerate(procedure, 1):
            action = str(step.get("action", "")).strip()
            if not action:
                results.append(f"Step {index}: skipped (no action)")
                continue
            tool_name = _action_to_tool(action)
            inputs = {k: v for k, v in step.items() if k not in {"action", "verify", "on_failure"}}
            try:
                step_approved = approved or (skill.get("permission", "read-only") != "read-only")
                step_result = self.run(tool_name, inputs, approved=step_approved)
                results.append(f"Step {index} [{action}]: ok - {step_result}")
            except ToolError as exc:
                on_failure = str(step.get("on_failure", "stop")).strip()
                results.append(f"Step {index} [{action}]: failed - {exc}")
                if on_failure == "stop":
                    return "\n".join(results) + f"\n\nProcedure stopped at step {index}."
                results.append(f"  on_failure={on_failure}; continuing.")
        return "\n".join(results) + "\n\nProcedure completed."


def _action_to_tool(action: str) -> str:
    """Map a human-readable action label to a registered tool name.

    This is deliberately narrow. Skills cannot invent new tools; they reuse the
    actions Ara already exposes.
    """
    mapping = {
        "read a file": "read_file",
        "read file": "read_file",
        "list a directory": "list_directory",
        "list directory": "list_directory",
        "list a folder": "list_directory",
        "list folder": "list_directory",
        "write a file": "write_file",
        "write file": "write_file",
        "create a file": "write_file",
        "create file": "write_file",
        "calculate": "calculate",
        "do arithmetic": "calculate",
    }
    lowered = action.lower().strip()
    if lowered in mapping:
        return mapping[lowered]
    for key, tool in mapping.items():
        if lowered.startswith(key):
            return tool
    known = ', '.join(sorted(mapping.keys()))
    raise ToolError('Unknown skill action ' + repr(action) + '; known actions: ' + known)

