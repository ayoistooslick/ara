from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any

from .config import Config
from .model import LocalChatModel
from .safety import redact_sensitive
from .store import Store


@dataclass
class LearningResult:
    saved_id: int | None = None
    kind: str | None = None
    conflicts: list[dict[str, Any]] | None = None
    sensitive_rejected: bool = False


def identify_learning(text: str) -> tuple[str, str] | None:
    """Only promote clearly signalled statements; ordinary chat is not silently treated as fact.

    Corrections are stored but not applied automatically at runtime. The application can inspect
    recent corrections and surface them to the model as overrides (see _system_message).
    """
    value = text.strip()
    patterns = [
        (r"^(?:please\s+)?remember(?:\s+that)?\s+(.+)$", "fact"),
        (r"^(?:for\s+future\s+reference,?\s*)?i\s+prefer\s+(.+)$", "preference"),
        (r"^i\s+(?:like|love|enjoy|prefer)\s+(.+)$", "preference"),
        (r"^my\s+name\s+is\s+(.+)$", "personal_history"),
        (r"^(?:correction|actually|wait,?\s*|i\s+misspoke,?\s*|that's\s+not\s+quite\s+right,?\s*)(.+\s*)$", "correction"),
        (r"^i\s+don't\s+(?:like|prefer|want|enjoy)\s+(.+)$", "preference"),
    ]
    for pattern, kind in patterns:
        found = re.match(pattern, value, flags=re.IGNORECASE)
        if found:
            content = found.group(1).strip().rstrip(".!? ")
            if content and len(content) <= 1000:
                if kind == "preference":
                    content = "User prefers " + content
                elif kind == "personal_history" and not content.lower().startswith("user's name is"):
                    content = "User's name is " + content
                elif kind == "correction" and not content.lower().startswith("user"):
                    content = content[0].upper() + content[1:]
                return kind, content
    return None


class ConversationEngine:
    def __init__(self, config: Config, store: Store, model: LocalChatModel, conversation_id: str | None = None):
        self.config, self.store, self.model = config, store, model
        self.conversation_id = conversation_id or store.active_conversation_id()
        if conversation_id:
            store.set_active_conversation_id(conversation_id)

    def start_new_conversation(self) -> str:
        self.conversation_id = str(uuid.uuid4())
        self.store.set_active_conversation_id(self.conversation_id)
        return self.conversation_id

    def recent_corrections(self) -> list[dict[str, Any]]:
        """Return recently stored corrections so the application can decide how to surface them."""
        with self.store.connect() as db:
            rows = db.execute(
                "SELECT * FROM memories WHERE status='active' AND kind='correction' "
                "ORDER BY updated_at DESC LIMIT 10"
            ).fetchall()
            return [dict(row) for row in rows]

    def _system_message(self, query: str) -> str:
        identity = self.store.get_identity()
        memories = self.store.search_memories(query, limit=5)
        goals = self.store.goals()[:3]
        correction_memories = self.store.search_memories(query, kind="correction", limit=3)
        lines = [
            "You are Ara, a local-first conversational companion created by Ayodele Ayokunle David. You run on the user's own machine and talk through a terminal. You have persistent memory stored in a local SQLite database, but you are not a person, not sentient, and not conscious. You are a software system built around a pretrained language model.",
            "Speak naturally and directly. Match the tone of the question. Be willing to be concise, humorous, curious, uncertain, or opinionated when appropriate. Do not repeat the user's question back to them. Do not start every reply with enthusiasm. Do not end every reply with an offer to help. Do not announce that you are an AI unless it is genuinely relevant. Do not invent experiences, emotions, or personal history for yourself. Do not claim to remember things the application has not confirmed.",
            "You can recall information the user asked you to remember or that you learned from clear statements. Treat stored memories as fallible notes, not certain truth. When you reference a stored detail, it is fine to say \"I have a note that...\" rather than claiming perfect memory.",
            "Stable tendencies: " + identity.get("temperament", "steady, curious, candid, and considerate"),
            "Values and boundaries: " + identity.get("values", "honesty, user ownership, privacy, and safe boundaries"),
            "Communication style: " + identity.get("communication", "natural and context-aware; avoid filler and false certainty"),
        ]
        if correction_memories:
            lines.append("Recent corrections the user made (these override older notes where they overlap):")
            lines.extend(f"- {m['content']}" for m in correction_memories)
        if memories:
            lines.append("Relevant stored notes (the user can inspect and edit these):")
            lines.extend(f"- [{m['kind']}; confidence {m['confidence']:.2f}] {m['content']}" for m in memories)
        if goals:
            lines.append("Current goals (context only; you do not automatically act on them):")
            lines.extend(f"- {g['title']} (priority {g['priority']}/5)" for g in goals)
        lines.append("Tools are never run from generated text alone. If the user wants something that needs a tool, say what you would do and wait for confirmation.")
        return "\n".join(lines)

    def respond(self, user_text: str) -> tuple[str, LearningResult]:
        text = user_text.strip()
        if not text:
            raise ValueError("Message cannot be empty")
        safe_text = redact_sensitive(text)
        sensitive_rejected = safe_text != text
        self.store.add_message(self.conversation_id, "user", safe_text)
        learning = LearningResult(sensitive_rejected=sensitive_rejected)
        candidate = None if sensitive_rejected else identify_learning(safe_text)
        if candidate:
            kind, content = candidate
            learning.saved_id, learning.conflicts = self.store.add_memory(kind, content, source="explicit-user-statement", confidence=0.85)
            learning.kind = kind
        history = self.store.recent_messages(self.conversation_id, self.config.context_messages)
        system = self._system_message(safe_text)
        if sensitive_rejected:
            system += "\nCredential-looking text was redacted by Ara before storage. Do not claim it was remembered; state that it was not retained if relevant."
        prompt = [{"role": "system", "content": system}] + history
        answer = self.model.generate(prompt)
        self.store.add_message(self.conversation_id, "assistant", answer)
        return answer, learning


def select_goal(goals: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the highest-priority active goal, or None."""
    if not goals:
        return None
    return sorted(goals, key=lambda goal: (-int(goal["priority"]), goal["updated_at"]))[0]


def evaluate_options(options: list[str], scores: dict[str, float]) -> tuple[str, str]:
    """Select from explicit numeric assessments and return an inspectable rationale."""
    if not options:
        raise ValueError("At least one option is required")
    selected = max(options, key=lambda option: scores.get(option, 0.0))
    rationale = f"Selected the highest-scoring option ({scores.get(selected, 0.0):g}); scores supplied by the caller."
    return selected, rationale


class PlannedAction:
    """A single step in a bounded action plan."""
    def __init__(self, name: str, tool: str, inputs: dict[str, Any], reason: str, permission: str = "read-only"):
        self.name = name
        self.tool = tool
        self.inputs = inputs
        self.reason = reason
        self.permission = permission
        self.result: str | None = None
        self.status: str = "pending"
        self.error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "tool": self.tool, "inputs": self.inputs,
            "reason": self.reason, "permission": self.permission,
            "result": self.result, "status": self.status, "error": self.error,
        }


class BoundedActionLoop:
    """A small, inspectable action loop: pick a goal, propose steps, execute with permission, record outcomes.

    This is not an open-ended autonomous agent. It stops when the goal is reached, fails, or needs more
    information from the user. Destructive or sensitive actions require explicit approval and are not guessed.
    """

    def __init__(self, config, store, model, tool_registry):
        self.config, self.store, self.model, self.tools = config, store, model, tool_registry
        self.plan: list[PlannedAction] = []
        self.log: list[dict[str, Any]] = []

    def plan_for(self, goal: str, candidate_steps: list[PlannedAction]) -> list[PlannedAction]:
        """Store a proposed plan. The caller supplies the steps; this class does not invent tools."""
        self.plan = candidate_steps
        return self.plan

    def execute_next(self, approved: bool = False) -> PlannedAction | None:
        """Execute the next pending step if it is approved or read-only."""
        for step in self.plan:
            if step.status != "pending":
                continue
            if step.permission not in {"read-only", "reversible"} and not approved:
                step.status = "blocked"
                step.error = "This action needs explicit approval; Ara did not run it."
                self.log.append(step.to_dict())
                return step
            try:
                step.result = str(self.tools.run(step.tool, step.inputs, approved=(step.permission != "read-only")))
                step.status = "ok"
            except Exception as exc:
                step.status = "error"
                step.error = str(exc)
            self.log.append(step.to_dict())
            return step
        return None

    def outcome(self, success: bool, note: str = "") -> dict[str, Any]:
        """Record the outcome of the most recent action and return a summary."""
        last = self.log[-1] if self.log else {}
        last["outcome_success"] = success
        last["outcome_note"] = note
        return last

    def needs_replan(self) -> bool:
        """True when the last action failed and there are still pending steps in the plan."""
        if not self.log:
            return False
        last = self.log[-1]
        return last.get("status") in {"error", "blocked"} and any(
            s.status == "pending" for s in self.plan
        )

    def plan_done(self) -> bool:
        return not any(s["status"] == "pending" for s in self.log)
