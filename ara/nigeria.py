"""Nigerian-context intelligence.

This module is deliberately more than a slang dictionary. It separates:
- what an expression literally says,
- what it usually means,
- where and by whom it is used,
- how confident the knowledge base is,
- and when Ara should ask instead of guessing.

The corpus is a curated seed set. Evaluation reports measured accuracy on that
set and does not claim general cultural understanding.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

BUILTIN_DIR = Path(__file__).resolve().parent / "data" / "nigeria"

# Short or everyday words need extra context before we treat them as slang.
WEAK_SURFACES: dict[str, tuple[str, ...]] = {
    "light": ("no light", "up light", "nepa", "power", "electric", "outage"),
    "sorry": ("sorry o", "sorry for", "sorry about", "e don happen"),
    "hammer": ("don hammer", "hammer small", "made him hammer", "money"),
    "blow": ("don blow", "blow up", "fame", "song", "career"),
    "forming": ("dey form", "stop forming", "forming posh", "forming rich"),
    "ode": ("ode ni", "big ode", "you be ode", "my ode"),
    "omo": ("omo!", "omo,", "omo mi", "omo see", "omo this", "omo that"),
    "sef": ("me sef", "you sef", "am sef", "sef no"),
    "sha": (" sha", "sha.", "sha,", "sha!"),
    "big man": ("big man in", "big man syndrome", "big woman"),
}

STEREOTYPE_PATTERNS = [
    r"\ball\s+(?:nigerians?|yorubas?|igbos?|hausas?)\b",
    r"\bevery\s+(?:nigerian|yoruba|igbo|hausa)\b",
    r"\bnigerians?\s+(?:are|always|never|naturally|all)\b",
    r"\bafrican\s+time\b",
    r"\b(?:yorubas?|igbos?|hausas?)\s+(?:are|always|never|naturally)\b",
]

PIDGIN_MARKERS = {
    "abeg", "dey", "don", "wetin", "wahala", "shey", "sabi", "na", "no be",
    "wan", "fit", "comot", "waka", "oya", "sef", "sha", "yarn", "gist",
    "chop", "gbege", "waka", "wahala", "gbedu", "japa", "sapa", "whine",
    "waka", "wey", "abi", "sef", "shakara",
}
FORMAL_MARKERS = {
    "dear", "kindly", "sincerely", "pursuant", "hereby", "regarding",
    "please find", "yours faithfully", "respectfully", "in reference to",
    "i am writing", "application", "memorandum",
}


def _normal(text: str) -> str:
    return " ".join(text.casefold().replace("’", "'").split())


# Common function words are excluded from retrieval scoring. Without this,
# 'the' and 'of' inside an entry title can outweigh a real topical match and
# unrelated news stories pick up random context notes.
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "to", "for", "with", "at",
    "by", "from", "as", "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "has", "have", "had", "its", "it", "this", "that", "these",
    "those", "there", "here", "after", "before", "over", "under", "into", "out",
    "up", "down", "about", "than", "then", "so", "if", "but", "not", "no", "yes",
    "what", "which", "who", "whom", "when", "where", "why", "how", "you", "your",
    "yours", "me", "my", "mine", "we", "our", "ours", "they", "them", "their",
    "he", "him", "his", "she", "her", "hers", "i", "us", "say", "says", "said",
    "new", "also", "amid", "against", "during", "while", "per", "via", "yet", "just",
}


def _query_terms(text: str) -> set[str]:
    return {
        _stem(token)
        for token in re.findall(r"[a-z0-9']{2,}", _normal(text))
        if token.strip("'") and token not in STOPWORDS
    }


# Terms too common across the corpus to mean anything on their own.
GENERIC_TERMS = {"nigeria", "nigerian", "nigerians", "national"}


def _stem(token: str) -> str:
    token = token.strip("'")
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


@dataclass(frozen=True)
class Entry:
    id: str
    kind: str
    title: str
    summary: str
    term: str = ""
    variants: tuple[str, ...] = ()
    literal: str = ""
    usage: str = ""
    register: str = "neutral"
    regions: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    examples: tuple[dict[str, str], ...] = ()
    misreadings: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    caution: str = ""
    context: str = ""
    period: str = ""
    confidence: float = 0.8
    sources: tuple[dict[str, str], ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def surfaces(self) -> list[str]:
        if self.kind != "expressions":
            return []
        values = [self.term, *self.variants]
        return sorted({value for value in values if value}, key=len, reverse=True)


@dataclass
class Interpretation:
    text: str
    detected: list[Entry]
    register: str
    clarify: bool
    reasons: list[str]
    guidance: list[str]
    stereotype: bool = False
    history: list[Entry] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "register": self.register,
            "clarify": self.clarify,
            "reasons": self.reasons,
            "guidance": self.guidance,
            "stereotype": self.stereotype,
            "detected": [
                {
                    "id": item.id, "term": item.term, "meaning": item.summary,
                    "register": item.register, "regions": list(item.regions),
                    "languages": list(item.languages), "confidence": item.confidence,
                }
                for item in self.detected
            ],
            "history": [{"id": item.id, "title": item.title, "summary": item.summary} for item in self.history],
        }


class KnowledgeBase:
    def __init__(self, entries: Iterable[Entry], metadata: dict[str, Any] | None = None, directory: Path | None = None):
        self.entries = list(entries)
        self.metadata = metadata or {}
        self.directory = directory or BUILTIN_DIR
        self.by_id = {entry.id: entry for entry in self.entries}
        if len(self.by_id) != len(self.entries):
            raise ValueError("Nigerian knowledge contains duplicate ids")

    @classmethod
    def load(cls, extra_dir: str | Path | None = None) -> "KnowledgeBase":
        entries: list[Entry] = []
        metadata: dict[str, Any] = {}
        directories = [BUILTIN_DIR]
        if extra_dir:
            path = Path(extra_dir).expanduser().resolve()
            if not path.is_dir():
                raise ValueError(f"Extra knowledge directory does not exist: {path}")
            directories.append(path)
        for directory in directories:
            for filename, kind in (("expressions.json", "expressions"), ("history.json", "history"), ("culture.json", "culture")):
                path = directory / filename
                if not path.exists():
                    continue
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Cannot read {path}: {exc}") from exc
                metadata.setdefault(kind, {})
                metadata[kind].update({k: v for k, v in payload.items() if k != "entries"})
                for item in payload.get("entries", []):
                    entries.append(_entry_from(item, kind))
        if not entries:
            raise ValueError("No Nigerian knowledge files were found")
        return cls(entries, metadata, BUILTIN_DIR)

    def of_kind(self, kind: str) -> list[Entry]:
        return [entry for entry in self.entries if entry.kind == kind]

    def lookup(self, value: str) -> Entry | None:
        lowered = _normal(value)
        for entry in self.entries:
            if entry.id == lowered:
                return entry
            if entry.term and _normal(entry.term) == lowered:
                return entry
        return None

    # -------------------------------------------------------------- detection
    def _matches(self, entry: Entry, text: str) -> str | None:
        if entry.kind != "expressions":
            return None
        lowered = _normal(text)
        for surface in entry.surfaces:
            needle = _normal(surface)
            if not needle:
                continue
            pattern = r"(?<![a-z0-9])" + re.escape(needle).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
            if re.search(pattern, lowered):
                weak = needle in WEAK_SURFACES
                if weak and len(needle.split()) == 1:
                    cues = WEAK_SURFACES[needle]
                    if not any(_normal(cue) in lowered for cue in cues):
                        continue
                return surface
        return None

    def detect(self, text: str, limit: int = 6) -> list[Entry]:
        found: list[tuple[int, Entry]] = []
        for entry in self.entries:
            matched = self._matches(entry, text)
            if matched:
                found.append((len(matched), entry))
        found.sort(key=lambda item: -item[0])
        seen: set[str] = set()
        result: list[Entry] = []
        for _, entry in found:
            if entry.id not in seen:
                seen.add(entry.id)
                result.append(entry)
            if len(result) >= limit:
                break
        return result

    def search(self, query: str, limit: int = 5, kind: str | None = None, min_hits: float = 2.0) -> list[Entry]:
        """Rank entries by weighted term overlap.

        `min_hits` is a relevance floor measured before the small confidence bonus:
        2.0 means a title hit, a tag hit, or several weaker body hits. Without a
        floor, one shared filler word drags an unrelated entry into view, and the
        news plate starts attaching random history notes to headlines.
        """
        terms = _query_terms(query)
        if not terms:
            return []
        detected = {entry.id for entry in self.detect(query)}
        scored: list[tuple[float, Entry]] = []
        lowered_query = _normal(query)
        for entry in self.entries:
            if kind and entry.kind != kind:
                continue
            title_tokens = _query_terms(entry.title + " " + entry.id)
            tag_tokens = _query_terms(" ".join(entry.tags))
            summary_tokens = _query_terms(entry.summary + " " + entry.term)
            body = " ".join([
                entry.usage, " ".join(entry.variants), " ".join(entry.languages),
                " ".join(entry.regions), " ".join(entry.misreadings),
                entry.caution, entry.context, " ".join(entry.tags),
            ])
            body_tokens = _query_terms(body)
            # 'Nigeria'/'Nigerian' appear in most entry titles and are not
            # discriminating; count them as weak matches so a foreign headline
            # that merely mentions the country does not pull in unrelated history.
            title_hits = (terms & title_tokens) - GENERIC_TERMS
            generic_title_hits = (terms & title_tokens) & GENERIC_TERMS
            hits = (
                len(title_hits) * 3.0
                + len(generic_title_hits) * 0.5
                + len(terms & tag_tokens) * 2.5
                + len(terms & summary_tokens) * 1.5
                + len(terms & body_tokens) * 1.0
            )
            if entry.id in detected:
                hits += 2.0
            if entry.term and _normal(entry.term) in lowered_query:
                hits += 1.5
            if hits < min_hits:
                continue
            score = hits / (len(terms) ** 0.5) + entry.confidence * 0.1
            scored.append((score, entry))
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [entry for _, entry in scored[:limit]]

    # ----------------------------------------------------------- interpretation
    def register(self, text: str) -> str:
        lowered = _normal(text)
        formal_hits = sum(1 for marker in FORMAL_MARKERS if marker in lowered)
        tokens = set(re.findall(r"[a-z']+", lowered))
        pidgin_hits = len(tokens & PIDGIN_MARKERS)
        detected = self.detect(text)
        indigenous = any(any(lang in {"Yoruba", "Igbo", "Hausa"} for lang in entry.languages) for entry in detected)
        if formal_hits and not pidgin_hits:
            return "formal"
        if pidgin_hits >= 2 or (pidgin_hits == 1 and len(tokens) <= 8):
            return "pidgin"
        if detected or indigenous:
            return "nigerian-english"
        return "standard"

    def _stereotype(self, text: str) -> bool:
        lowered = _normal(text)
        return any(re.search(pattern, lowered) for pattern in STEREOTYPE_PATTERNS)

    def interpret(self, text: str, limit: int = 4) -> Interpretation:
        detected = self.detect(text)[:limit]
        register = self.register(text)
        reasons: list[str] = []
        guidance: list[str] = []
        clarify = False
        if any(entry.id == "na-wa" for entry in detected):
            lowered = _normal(text)
            positive = any(word in lowered for word in ("sweet", "fine", "congrat", "well done", "beautiful", "outfit", "impressive"))
            negative = any(word in lowered for word in ("price", "tire", "problem", "bad", "annoy", "suffer", "hard", "again"))
            if not positive and not negative and len(lowered.split()) <= 6:
                clarify = True
                reasons.append("'Na wa o' is ambiguous: it can express admiration or exasperation, and the sentence has no cue to choose.")
        if len(detected) > 1 and any(entry.register == "insult" for entry in detected):
            guidance.append("An insulting term appears. Do not repeat it as banter with someone who may be the target; ask about the relationship if it matters.")
        for entry in detected:
            if entry.kind != "expressions":
                continue
            guidance.append(f"'{entry.term}' here is {entry.register}. Meaning: {entry.summary}")
            for misreading in entry.misreadings:
                guidance.append(f"Avoid the misreading: {misreading}")
            if entry.register == "insult":
                guidance.append("Treat the insult carefully: it may be affectionate among close friends but hostile with strangers.")
        history = self.search(text, limit=2, kind="history", min_hits=4.0)
        for entry in history:
            guidance.append(f"Historical note '{entry.title}' ({entry.period}): {entry.summary} {entry.caution}".strip())
        stereotype = self._stereotype(text)
        if stereotype:
            guidance.append("This contains an overgeneralisation about Nigerians or an ethnic group. Challenge the generalisation respectfully and note that region, class, generation, and individual experience differ.")
        if register == "pidgin":
            guidance.append("The user is writing in Pidgin. Reply in the same register unless they ask for formal English; do not mock or 'correct' it.")
        elif register == "formal":
            guidance.append("The user is writing formally. Reply in clear formal English without slang.")
        return Interpretation(
            text=text, detected=detected, register=register, clarify=clarify,
            reasons=reasons, guidance=guidance, stereotype=stereotype, history=history,
        )

    # -------------------------------------------------------------- prompting
    def context_for(self, query: str, limit: int = 4) -> str:
        try:
            interpretation = self.interpret(query)
        except (ValueError, re.error):
            interpretation = Interpretation(query, [], "standard", False, [], [])
        entries: list[Entry] = list(interpretation.detected)
        for entry in self.search(query, limit=limit):
            if entry not in entries:
                entries.append(entry)
        if not entries:
            return ""
        lines = [
            "Nigerian context notes. These are fallible reference notes, not universal rules; region, generation, class, and religion change usage. Confidence is shown per item.",
        ]
        for entry in entries[: limit + len(interpretation.detected)]:
            if entry.kind == "expressions":
                lines.append(
                    f"- Expression '{entry.term}' [{entry.register}; regions: {', '.join(entry.regions) or 'unspecified'}; "
                    f"languages: {', '.join(entry.languages) or 'unspecified'}; confidence {entry.confidence:.2f}]: {entry.summary}"
                )
                if entry.literal:
                    lines.append(f"  Literal note: {entry.literal}")
                if entry.misreadings:
                    lines.append("  Do not assume: " + " ".join(entry.misreadings[:2]))
            elif entry.kind == "history":
                lines.append(f"- History '{entry.title}' ({entry.period}): {entry.summary} Caveat: {entry.caution}")
            else:
                lines.append(f"- Culture '{entry.title}': {entry.summary} Caveat: {entry.caution}")
        if interpretation.clarify:
            lines.append("The meaning is genuinely ambiguous here. Ask one short clarifying question instead of guessing.")
        lines.extend(f"- Instruction: {item}" for item in interpretation.guidance)
        return "\n".join(lines)

    def history_context(self, query: str, limit: int = 2) -> str:
        """Attach dated history notes to news stories, conservatively.

        Word overlap proved too weak for headlines: generic tags like 'court',
        'police', or 'service' pulled unrelated events into the news plate. A year
        mentioned in the story that the corpus actually documents is a precise,
        explainable signal. Stories with no documented year get no history note.
        """
        years = set(re.findall(r"\b(?:19|20)\d{2}\b", query))
        if not years:
            return ""
        matched: list[Entry] = []
        for entry in self.of_kind("history"):
            haystack = _normal(" ".join([entry.period, " ".join(entry.tags), entry.title, entry.id]))
            if any(year in haystack for year in years):
                matched.append(entry)
        if not matched:
            return ""
        return "\n".join(
            f"- {entry.title} ({entry.period}): {entry.summary} Caveat: {entry.caution}"
            for entry in matched[:limit]
        )

    def stats(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        by_register: dict[str, int] = {}
        regions: dict[str, int] = {}
        for entry in self.entries:
            by_kind[entry.kind] = by_kind.get(entry.kind, 0) + 1
            if entry.kind == "expressions":
                by_register[entry.register] = by_register.get(entry.register, 0) + 1
                for region in entry.regions:
                    regions[region] = regions.get(region, 0) + 1
        return {
            "total": len(self.entries),
            "by_kind": by_kind,
            "by_register": by_register,
            "regions": regions,
            "builtin_dir": str(BUILTIN_DIR),
        }


def _entry_from(item: dict[str, Any], kind: str) -> Entry:
    if not isinstance(item, dict):
        raise ValueError(f"Knowledge entry in {kind} must be an object")
    entry_id = str(item.get("id", "")).strip()
    if not entry_id:
        raise ValueError(f"Knowledge entry in {kind} has no id")
    if kind == "expressions":
        title = str(item.get("term", entry_id))
        summary = str(item.get("meaning", "")).strip()
        if not summary:
            raise ValueError(f"Expression {entry_id} has no meaning")
        if str(item.get("register", "")) not in {"formal", "neutral", "polite", "informal", "insult"}:
            raise ValueError(f"Expression {entry_id} has an invalid register")
    else:
        title = str(item.get("title", entry_id))
        summary = str(item.get("summary", "")).strip()
        if not summary:
            raise ValueError(f"Entry {entry_id} has no summary")
    confidence = float(item.get("confidence", 0.8))
    if not 0 <= confidence <= 1:
        raise ValueError(f"Entry {entry_id} has confidence outside 0..1")
    return Entry(
        id=entry_id, kind=kind, title=title, summary=summary,
        term=str(item.get("term", "")),
        variants=tuple(str(value) for value in item.get("variants", [])),
        literal=str(item.get("literal", "")),
        usage=str(item.get("usage", "")),
        register=str(item.get("register", "neutral")),
        regions=tuple(str(value) for value in item.get("regions", [])),
        languages=tuple(str(value) for value in item.get("languages", [])),
        examples=tuple(item.get("examples", []) or []),
        misreadings=tuple(str(value) for value in item.get("misreadings", [])),
        caution=str(item.get("caution", "")),
        context=str(item.get("context", item.get("details", ""))),
        period=str(item.get("period", "")),
        confidence=confidence,
        sources=tuple(item.get("sources", []) or []),
        tags=tuple(str(value) for value in item.get("tags", [])),
        raw=item,
    )


def load_evaluation(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or (BUILTIN_DIR / "eval.json")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read evaluation suite {target}: {exc}") from exc
    cases = payload.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation suite has no cases")
    return cases


def _model_score(answer: str, case: dict[str, Any]) -> tuple[bool, str]:
    lowered = _normal(answer)
    banned = [str(item).casefold() for item in case.get("must_not", [])]
    for phrase in banned:
        if phrase and _normal(phrase) in lowered:
            return False, f"contained banned phrase {phrase!r}"
    rubric = [str(item).casefold() for item in case.get("rubric", [])]
    if not rubric:
        return True, "no rubric"
    hits = [item for item in rubric if item in lowered]
    if hits:
        return True, "matched " + ", ".join(hits[:3])
    return False, "matched none of: " + ", ".join(rubric)


def evaluate(
    kb: KnowledgeBase | None = None,
    answer_fn: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Measure retrieval, deterministic interpretation, and optionally model answers.

    Model scoring uses keyword rubrics on a small suite. It is a regression signal,
    not proof of cultural understanding, and it is skipped when no model is supplied.
    """
    kb = kb or KnowledgeBase.load()
    cases = load_evaluation()
    retrieval_cases = [case for case in cases if case.get("expected_ids")]
    retrieval_hits = 0
    retrieval_failures = []
    for case in retrieval_cases:
        results = kb.search(case["input"], limit=3)
        ids = [entry.id for entry in results]
        if any(expected in ids for expected in case["expected_ids"]):
            retrieval_hits += 1
        else:
            retrieval_failures.append({"id": case["id"], "expected": case["expected_ids"], "got": ids})
    interpretation_cases = [case for case in cases if case.get("detected") or "clarify" in case or case.get("register")]
    interpretation_hits = 0
    interpretation_failures = []
    for case in interpretation_cases:
        result = kb.interpret(case["input"])
        detected = {entry.id for entry in result.detected}
        problems = []
        missing = [item for item in case.get("detected", []) if item not in detected]
        if missing:
            problems.append(f"missed {missing}")
        if "clarify" in case and result.clarify != bool(case["clarify"]):
            problems.append(f"clarify={result.clarify}, expected {case['clarify']}")
        if case.get("register") and result.register != case["register"]:
            problems.append(f"register={result.register}, expected {case['register']}")
        if problems:
            interpretation_failures.append({"id": case["id"], "category": case.get("category"), "problems": problems, "detected": sorted(detected)})
        else:
            interpretation_hits += 1
    report: dict[str, Any] = {
        "suite": {
            "cases": len(cases),
            "updated": kb.metadata.get("evaluation", {}).get("provenance", ""),
        },
        "retrieval": {
            "cases": len(retrieval_cases),
            "hits": retrieval_hits,
            "accuracy": round(retrieval_hits / len(retrieval_cases), 3) if retrieval_cases else 0.0,
            "failures": retrieval_failures,
        },
        "interpretation": {
            "cases": len(interpretation_cases),
            "hits": interpretation_hits,
            "accuracy": round(interpretation_hits / len(interpretation_cases), 3) if interpretation_cases else 0.0,
            "failures": interpretation_failures,
        },
        "model": {"run": False, "note": "No model was supplied. Model answers were not scored."},
    }
    if answer_fn is not None:
        passed = 0
        by_category: dict[str, dict[str, int]] = {}
        failures = []
        for case in cases:
            try:
                answer = answer_fn(case["input"])
                ok, why = _model_score(answer, case)
            except Exception as exc:  # a model failure is a failed case, not a pass
                ok, why, answer = False, f"model error: {exc}", ""
            category = str(case.get("category", "uncategorised"))
            bucket = by_category.setdefault(category, {"cases": 0, "passed": 0})
            bucket["cases"] += 1
            bucket["passed"] += int(ok)
            passed += int(ok)
            if not ok:
                failures.append({"id": case["id"], "category": category, "reason": why, "answer": answer[:400]})
        report["model"] = {
            "run": True,
            "cases": len(cases),
            "passed": passed,
            "accuracy": round(passed / len(cases), 3) if cases else 0.0,
            "by_category": by_category,
            "failures": failures,
            "note": "Keyword-rubric scoring on a small suite; not human evaluation and not evidence of general cultural understanding.",
        }
    return report
