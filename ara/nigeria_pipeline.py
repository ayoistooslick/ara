"""Reproducible pipeline for Ara's Nigerian knowledge.

No external social-media corpus is bundled. This module validates the curated
seed set, versions it with hashes, and provides a guarded importer for future
licensed or consented contributions. It refuses records that look private or
contain personal identifiers.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .nigeria import BUILTIN_DIR, KnowledgeBase

PII_PATTERNS = [
    re.compile(r"\b\d{11}\b"),  # Nigerian mobile number shape
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
    re.compile(r"\b\d{10}\b"),  # bank account shape
]
PRIVATE_FLAGS = {"private", "dm", "whatsapp", "closed_group", "personal_message"}
ALLOWED_REGISTERS = {"formal", "neutral", "polite", "informal", "insult"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_corpus(directory: str | Path | None = None) -> dict[str, Any]:
    """Check structure, duplicates, ranges, and basic privacy hygiene."""
    base = Path(directory).expanduser().resolve() if directory else BUILTIN_DIR
    errors: list[str] = []
    warnings: list[str] = []
    ids: dict[str, str] = {}
    counts: dict[str, int] = {}
    files: list[dict[str, Any]] = []
    for filename, kind in (("expressions.json", "expressions"), ("history.json", "history"), ("culture.json", "culture")):
        path = base / filename
        if not path.exists():
            errors.append(f"Missing {filename}")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{filename} is not valid JSON: {exc}")
            continue
        if not isinstance(payload.get("entries"), list) or not payload["entries"]:
            errors.append(f"{filename} has no entries")
            continue
        if not payload.get("license") or not payload.get("provenance"):
            errors.append(f"{filename} is missing licence or provenance metadata")
        counts[kind] = len(payload["entries"])
        files.append({"file": filename, "sha256": _sha256(path), "entries": len(payload["entries"])})
        for item in payload["entries"]:
            entry_id = str(item.get("id", "")).strip()
            if not entry_id:
                errors.append(f"{filename}: an entry has no id")
                continue
            if entry_id in ids:
                errors.append(f"Duplicate id {entry_id!r} in {filename} and {ids[entry_id]}")
            ids[entry_id] = filename
            summary = item.get("meaning") if kind == "expressions" else item.get("summary")
            if not isinstance(summary, str) or len(summary.strip()) < 10:
                errors.append(f"{entry_id}: summary/meaning is too short")
            confidence = item.get("confidence", 0.8)
            if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
                errors.append(f"{entry_id}: confidence must be between 0 and 1")
            if float(confidence) < 0.9:
                warnings.append(f"{entry_id}: confidence {confidence} is below 0.9; mark for community review")
            if kind == "expressions":
                register = item.get("register")
                if register not in ALLOWED_REGISTERS:
                    errors.append(f"{entry_id}: invalid register {register!r}")
                if not item.get("regions") or not item.get("languages"):
                    warnings.append(f"{entry_id}: missing region or language metadata")
            blob = json.dumps(item, ensure_ascii=False)
            if any(pattern.search(blob) for pattern in PII_PATTERNS):
                errors.append(f"{entry_id}: contains something that looks like a phone number, email, or account number")
            if any(flag in blob.casefold() for flag in PRIVATE_FLAGS):
                warnings.append(f"{entry_id}: mentions private/direct-message material; confirm consent before keeping it")
    for case in _load_eval(base):
        if case.get("expected_ids") and not any(item in ids for item in case["expected_ids"]):
            errors.append(f"Evaluation {case.get('id')} references unknown ids {case['expected_ids']}")
    return {
        "ok": not errors,
        "directory": str(base),
        "files": files,
        "counts": counts,
        "total": sum(counts.values()),
        "errors": errors,
        "warnings": warnings,
    }


def _load_eval(base: Path) -> list[dict[str, Any]]:
    path = base / "eval.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return payload.get("cases", []) if isinstance(payload.get("cases"), list) else []


def write_manifest(directory: str | Path | None = None, output: str | Path | None = None) -> Path:
    """Write a deterministic manifest of the knowledge corpus."""
    base = Path(directory).expanduser().resolve() if directory else BUILTIN_DIR
    report = validate_corpus(base)
    if not report["ok"]:
        raise ValueError("Cannot version an invalid corpus: " + "; ".join(report["errors"]))
    manifest = {
        "version": 1,
        "generated_at": _now(),
        "license": "Same licence as the Ara repository",
        "provenance": "Curated seed corpus; see each file and docs/NIGERIAN_KNOWLEDGE.md",
        "files": report["files"],
        "counts": report["counts"],
        "evaluation_cases": len(_load_eval(base)),
        "warnings": len(report["warnings"]),
        "sha256": hashlib.sha256("".join(item["sha256"] for item in report["files"]).encode()).hexdigest(),
    }
    target = Path(output) if output else base / "manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return target


def ingest_jsonl(
    source: str | Path,
    kind: str,
    output_dir: str | Path,
    license_name: str,
    consent_basis: str,
) -> dict[str, Any]:
    """Import licensed entries from JSONL with privacy and provenance checks.

    Each record must carry an id and the fields required by its kind. Records that
    look like private messages, contain personal identifiers, or lack a consent
    basis are rejected rather than silently cleaned.
    """
    if kind not in {"expressions", "history", "culture"}:
        raise ValueError("kind must be expressions, history, or culture")
    if not license_name.strip() or not consent_basis.strip():
        raise ValueError("A licence and consent basis are required before importing data")
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Import file does not exist: {path}")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            rejected.append({"line": str(number), "reason": f"invalid JSON: {exc}"})
            continue
        if not isinstance(record, dict):
            rejected.append({"line": str(number), "reason": "not an object"})
            continue
        entry_id = str(record.get("id", "")).strip()
        if not entry_id:
            rejected.append({"line": str(number), "reason": "missing id"})
            continue
        if entry_id in seen:
            rejected.append({"line": str(number), "reason": f"duplicate id {entry_id}"})
            continue
        blob = json.dumps(record, ensure_ascii=False)
        if any(pattern.search(blob) for pattern in PII_PATTERNS):
            rejected.append({"line": str(number), "reason": "looks like personal data"})
            continue
        if record.get("private") or any(flag in blob.casefold() for flag in PRIVATE_FLAGS):
            rejected.append({"line": str(number), "reason": "private or consent-unclear material"})
            continue
        record.setdefault("confidence", 0.7)
        record["provenance"] = {
            "license": license_name.strip(),
            "consent_basis": consent_basis.strip(),
            "imported_at": _now(),
            "source_file": path.name,
        }
        seen.add(entry_id)
        accepted.append(record)
    existing_path = destination / f"{kind}.json"
    existing: list[dict[str, Any]] = []
    if existing_path.exists():
        try:
            existing = json.loads(existing_path.read_text(encoding="utf-8")).get("entries", [])
        except (OSError, json.JSONDecodeError):
            raise ValueError(f"Existing {existing_path} is not valid JSON")
    by_id = {str(item.get("id")): item for item in existing}
    added = 0
    for item in accepted:
        if item["id"] not in by_id:
            by_id[item["id"]] = item
            added += 1
    payload = {
        "version": 1,
        "kind": kind,
        "license": license_name.strip(),
        "provenance": f"Imported from {path.name}; consent basis: {consent_basis.strip()}",
        "entries": sorted(by_id.values(), key=lambda item: str(item.get("id"))),
    }
    existing_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "accepted": len(accepted),
        "added": added,
        "duplicates_skipped": len(accepted) - added,
        "rejected": rejected,
        "output": str(existing_path),
    }


def check_loaded(directory: str | Path | None = None) -> tuple[bool, str]:
    try:
        kb = KnowledgeBase.load(directory)
    except ValueError as exc:
        return False, str(exc)
    return True, f"Loaded {len(kb.entries)} entries ({kb.stats()['by_kind']})"
