from __future__ import annotations

import re

# Redact the rest of the line after an explicit credential assignment. Values can
# contain spaces, so stopping at the first whitespace would leak passphrases.
_ASSIGNMENT = re.compile(
    r"\b(?:password|passwd|api[ _-]?key|access[ _-]?token|secret[ _-]?key|private[ _-]?key)\b\s*(?:is|:|=)\s*.+$",
    re.IGNORECASE | re.DOTALL,
)
_BEARER = re.compile(r"\b(bearer\s+)\S+", re.IGNORECASE)


def redact_sensitive(text: str) -> str:
    text = _ASSIGNMENT.sub("[credential details redacted]", text)
    return _BEARER.sub(r"\1[redacted]", text)


def contains_credential(text: str) -> bool:
    return redact_sensitive(text) != text
