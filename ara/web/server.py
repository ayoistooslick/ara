"""Ara's web server.

Deliberately dependency-free: Python's standard library serves the app and a small
JSON/SSE API. The web interface and terminal share the same engine, store, model
adapter, Nigerian knowledge, and memory. Filesystem tools are not exposed here.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import mimetypes
import os
import posixpath
import re
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .. import __version__
from ..brand import static_frog
from ..config import Config, effective_profile
from ..embeddings import Embedder, get_embedder, retrieval_report
from ..engine import ConversationEngine
from ..model import ModelError, backend_name, get_model
from ..news import news_report
from ..nigeria import KnowledgeBase
from ..perception import (
    PerceptionError,
    PerceptionUnavailable,
    Transcriber,
    VisionClient,
    perception_status,
)
from ..store import Store, now


class ApiError(RuntimeError):
    def __init__(self, status: int, message: str, code: str = "error"):
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass
class Response:
    body: bytes
    content_type: str = "application/json; charset=utf-8"
    status: int = 200
    headers: dict[str, str] | None = None


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _cookie(token: str, max_age: int, secure: bool) -> str:
    parts = [f"ara_session={token}", "Path=/", "HttpOnly", "SameSite=Strict", f"Max-Age={max_age}"]
    if secure:
        parts.append("Secure")
    return "; ".join(parts)


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _decode_data_url(value: str, allowed: tuple[str, ...], max_bytes: int) -> tuple[bytes, str]:
    if not isinstance(value, str) or not value:
        raise ApiError(400, "Missing file data", "missing_file")
    match = re.match(r"^data:([a-z0-9.+-]+/[a-z0-9.+-]+);base64,(.*)$", value, re.IGNORECASE | re.DOTALL)
    if match:
        mime, payload = match.group(1).lower(), match.group(2)
    else:
        mime, payload = "application/octet-stream", value
    if mime != "application/octet-stream" and mime not in allowed:
        raise ApiError(415, f"Unsupported file type {mime}", "bad_type")
    try:
        data = base64.b64decode(payload, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ApiError(400, "File data is not valid base64", "bad_base64") from exc
    if len(data) > max_bytes:
        raise ApiError(413, "File is too large", "too_large")
    if not data:
        raise ApiError(400, "File is empty", "empty_file")
    return data, mime


class AraServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], config: Config, store: Store | None = None):
        self.config = config
        self.store = store or Store(config.db_path)
        self.knowledge = KnowledgeBase.load(config.knowledge_dir)
        self.embedder: Embedder = get_embedder(config)
        self.session_hours = config.web_session_hours
        self._lock = threading.Lock()
        self._attempts: dict[str, deque[float]] = defaultdict(deque)
        self._login_attempts: dict[str, deque[float]] = defaultdict(deque)
        self._model_cache: tuple[float, dict[str, Any]] | None = None
        self._news_refresh_at = 0.0
        super().__init__(address, AraRequestHandler)

    @property
    def password_required(self) -> bool:
        return bool(self.config.web_password)

    def is_loopback(self) -> bool:
        return self.server_address[0] in {"127.0.0.1", "::1", "localhost"}

    def rate_limited(self, key: str, limit: int | None = None) -> bool:
        limit = limit or self.config.web_rate_limit
        current = time.time()
        with self._lock:
            bucket = self._attempts[key]
            while bucket and current - bucket[0] > 60:
                bucket.popleft()
            if len(bucket) >= limit:
                return True
            bucket.append(current)
            return False

    def login_limited(self, key: str) -> bool:
        current = time.time()
        with self._lock:
            bucket = self._login_attempts[key]
            while bucket and current - bucket[0] > 300:
                bucket.popleft()
            if len(bucket) >= 10:
                return True
            bucket.append(current)
            return False

    def model_state(self) -> dict[str, Any]:
        with self._lock:
            cached = self._model_cache
            if cached and time.time() - cached[0] < 15:
                return cached[1]
        try:
            model = get_model(self.config)
            if model is None:
                state = {
                    "backend": "retrieval-only",
                    "active": "retrieval-only",
                    "reachable": False,
                    "message": "No model is configured. Replies use stored notes and the Nigerian knowledge base, and are labelled retrieval-only.",
                    "models": [],
                    "vision": False,
                    "streaming": False,
                    "configured_model": "none",
                }
            else:
                state = model.capabilities()
                state["active"] = backend_name(model)
                if not state.get("reachable"):
                    state["message"] = str(state.get("message", "")) + " Replies fall back to labelled retrieval until the model is reachable."
        except ModelError as exc:
            state = {
                "backend": self.config.active_backend,
                "active": "retrieval-only",
                "reachable": False,
                "message": str(exc),
                "models": [], "vision": False, "streaming": False,
                "configured_model": self.config.hosted_model,
            }
        with self._lock:
            self._model_cache = (time.time(), state)
        return state

    def new_session(self, label: str = "") -> tuple[str, dict[str, Any]]:
        token = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(24)
        conversation_id = str(uuid.uuid4())
        self.store.create_web_session(_hash_token(token), csrf, conversation_id, self.session_hours, label)
        max_age = self.session_hours * 3600
        return token, {
            "csrf": csrf,
            "conversation_id": conversation_id,
            "expires_in": max_age,
        }


class AraRequestHandler(BaseHTTPRequestHandler):
    server: AraServer
    protocol_version = "HTTP/1.1"
    server_version = "Ara/" + __version__
    sys_version = ""

    # ------------------------------------------------------------------ plumbing
    def log_message(self, fmt: str, *args: Any) -> None:
        if os.environ.get("ARA_WEB_ACCESS_LOG"):
            super().log_message(fmt, *args)

    def _security_headers(self) -> dict[str, str]:
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Referrer-Policy": "no-referrer",
            "Permissions-Policy": "camera=(self), microphone=(self), geolocation=()",
            "Cross-Origin-Opener-Policy": "same-origin",
            "Content-Security-Policy": (
                "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
                "connect-src 'self'; style-src 'self'; script-src 'self'; "
                "font-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
            ),
        }

    def _send(self, response: Response) -> None:
        self.send_response(response.status)
        headers = self._security_headers()
        headers.update(response.headers or {})
        if response.content_type:
            headers.setdefault("Content-Type", response.content_type)
        headers.setdefault("Cache-Control", "no-store")
        body = response.body
        headers.setdefault("Content-Length", str(len(body)))
        if response.status >= 400:
            self.close_connection = True
            headers.setdefault("Connection", "close")
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

    def _json(self, payload: Any, status: int = 200, headers: dict[str, str] | None = None) -> None:
        self._send(Response(_json_bytes(payload), status=status, headers=headers))

    def _error(self, status: int, message: str, code: str = "error") -> None:
        self._json({"error": message, "code": code}, status=status)

    def _read_body(self, limit: int) -> dict[str, Any]:
        length_header = self.headers.get("Content-Length")
        if not length_header:
            raise ApiError(411, "A Content-Length header is required", "length_required")
        try:
            length = int(length_header)
        except ValueError as exc:
            raise ApiError(400, "Invalid Content-Length", "bad_length") from exc
        if length < 0 or length > limit:
            self.close_connection = True
            raise ApiError(413, "Request body is too large", "too_large")
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError(400, "Request body must be valid JSON", "bad_json") from exc
        if not isinstance(payload, dict):
            raise ApiError(400, "Request body must be a JSON object", "bad_json")
        return payload

    def _cookie_token(self) -> str | None:
        header = self.headers.get("Cookie", "")
        for part in header.split(";"):
            name, _, value = part.strip().partition("=")
            if name == "ara_session" and value:
                return value
        return None

    def _session(self, required: bool = True) -> dict[str, Any] | None:
        token = self._cookie_token()
        if not token:
            if required:
                raise ApiError(401, "No Ara session. Start one from the app.", "no_session")
            return None
        session = self.server.store.web_session(_hash_token(token))
        if not session:
            if required:
                raise ApiError(401, "Your Ara session expired. Reload to start a new one.", "expired_session")
            return None
        session["_hash"] = _hash_token(token)
        return session

    def _same_origin(self) -> None:
        origin = self.headers.get("Origin")
        if not origin:
            return
        parsed = urlparse(origin)
        host = self.headers.get("Host", "")
        if parsed.netloc != host:
            raise ApiError(403, "Cross-origin request refused", "bad_origin")

    def _csrf(self, session: dict[str, Any]) -> None:
        self._same_origin()
        token = self.headers.get("X-Ara-CSRF", "")
        if not token or not hmac.compare_digest(token, session["csrf"]):
            raise ApiError(403, "Missing or invalid CSRF token", "bad_csrf")

    def _rate(self, session: dict[str, Any], bucket: str = "api", limit: int | None = None) -> None:
        key = f"{session['_hash'][:16]}:{bucket}"
        if self.server.rate_limited(key, limit):
            raise ApiError(429, "Too many requests. Wait a minute and try again.", "rate_limited")

    def _route(self) -> tuple[str, dict[str, list[str]]]:
        parsed = urlparse(self.path)
        return parsed.path, parse_qs(parsed.query)

    def _dispatch(self) -> None:
        path, query = self._route()
        if path == "/api/health":
            return self._health()
        if path == "/api/session" and self.command == "POST":
            return self._session_create()
        if path == "/api/state" and self.command == "GET":
            return self._state()
        if path == "/api/logout" and self.command == "POST":
            return self._logout()
        if path == "/api/chat" and self.command == "POST":
            return self._chat()
        if path == "/api/news" and self.command == "GET":
            return self._news(query)
        if path == "/api/nigeria" and self.command == "GET":
            return self._nigeria(query)
        if path == "/api/vision" and self.command == "POST":
            return self._vision()
        if path == "/api/transcribe" and self.command == "POST":
            return self._transcribe()
        if path == "/api/conversation" and self.command == "GET":
            return self._conversation()
        if path == "/api/conversation/new" and self.command == "POST":
            return self._new_conversation()
        if path == "/api/memory" and self.command in {"GET", "POST"}:
            return self._memory_collection()
        if path == "/api/memory/export" and self.command == "GET":
            return self._memory_export()
        if path == "/api/memory/forget-all" and self.command == "POST":
            return self._memory_forget_all()
        memory_match = re.fullmatch(r"/api/memory/(\d+)(/correct)?", path)
        if memory_match:
            memory_id = int(memory_match.group(1))
            if memory_match.group(2) and self.command == "POST":
                return self._memory_correct(memory_id)
            if not memory_match.group(2) and self.command == "DELETE":
                return self._memory_delete(memory_id)
        raise ApiError(404, "Not found", "not_found")

    def do_GET(self) -> None:  # noqa: N802
        try:
            path, _ = self._route()
            if path.startswith("/api/"):
                return self._dispatch()
            return self._static(path)
        except ApiError as exc:
            self._error(exc.status, str(exc), exc.code)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            self._error(500, "Ara hit an internal error on this request.", "internal")

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_POST(self) -> None:  # noqa: N802
        try:
            return self._dispatch()
        except ApiError as exc:
            self._error(exc.status, str(exc), exc.code)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            self._error(500, "Ara hit an internal error on this request.", "internal")

    def do_DELETE(self) -> None:  # noqa: N802
        try:
            return self._dispatch()
        except ApiError as exc:
            self._error(exc.status, str(exc), exc.code)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            self._error(500, "Ara hit an internal error on this request.", "internal")

    # -------------------------------------------------------------------- static
    def _static(self, path: str) -> None:
        static_root = (Path(__file__).resolve().parent / "static").resolve()
        relative = "index.html" if path in {"", "/", "/index.html"} else path.lstrip("/")
        if relative.startswith("static/"):
            relative = relative[len("static/"):]
        normalized = posixpath.normpath(relative)
        if normalized.startswith("..") or normalized.startswith("/") or "\\" in normalized:
            raise ApiError(404, "Not found", "not_found")
        target = (static_root / normalized).resolve()
        if static_root not in target.parents and target != static_root:
            raise ApiError(404, "Not found", "not_found")
        if not target.is_file():
            raise ApiError(404, "Not found", "not_found")
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if target.suffix == ".svg":
            content_type = "image/svg+xml"
        body = target.read_bytes()
        headers = {"Cache-Control": "no-cache" if target.suffix in {".html", ".js", ".css"} else "public, max-age=3600"}
        self._send(Response(body, content_type=content_type, headers=headers))

    # -------------------------------------------------------------------- routes
    def _health(self) -> None:
        model = self.server.model_state()
        self._json({
            "ok": True,
            "version": __version__,
            "time": now(),
            "database": "ok",
            "model_reachable": bool(model.get("reachable")),
            "configured_model": self.server.config.model_name,
        })

    def _session_create(self) -> None:
        payload = self._read_body(4096)
        host = self.client_address[0]
        if self.server.login_limited(host):
            raise ApiError(429, "Too many session attempts. Wait five minutes.", "login_limited")
        if self.server.password_required:
            supplied = str(payload.get("password", ""))
            if not hmac.compare_digest(supplied, self.server.config.web_password or ""):
                time.sleep(0.4)
                raise ApiError(401, "Incorrect password", "bad_password")
        token, data = self.server.new_session()
        secure = (self.headers.get("X-Forwarded-Proto") or "").lower() == "https"
        max_age = self.server.session_hours * 3600
        cookie = _cookie(token, max_age, secure=secure)
        self.server.store.log_web_event(_hash_token(token), "session_created")
        self._json({
            "authenticated": True,
            "csrf": data["csrf"],
            "conversation_id": data["conversation_id"],
            "expires_in": data["expires_in"],
        }, headers={"Set-Cookie": cookie})

    def _logout(self) -> None:
        session = self._session()
        self._csrf(session)
        self.server.store.delete_web_session(session["_hash"])
        self._json({"ok": True}, headers={"Set-Cookie": _cookie("", 0, secure=False)})

    def _state(self) -> None:
        session = self._session()
        model = self.server.model_state()
        memory_rows = self.server.store.list_memories(limit=1)
        self._json({
            "authenticated": True,
            "password_required": self.server.password_required,
            "version": __version__,
            "conversation_id": session["conversation_id"],
            "model": model,
            "profile": {
                "requested": self.server.config.model_profile,
                "effective": effective_profile(self.server.config),
                "max_tokens": self.server.config.max_tokens,
                "context_messages": self.server.config.context_messages,
            },
            "counts": self.server.store.counts(),
            "memory": {
                "count": len(self.server.store.list_memories(limit=10_000)),
                "retrieval": retrieval_report(self.server.store, self.server.embedder),
                "shared_note": "Memory is shared between Terminal Ara and Web Ara on this installation. Conversation threads are isolated per web session.",
            },
            "perception": perception_status(self.server.config),
            "knowledge": self.server.knowledge.stats(),
            "news": {"cached": self.server.store.news_items(1) != [], "last_fetch": self.server.store.news_last_fetch()},
            "privacy": {
                "local_by_default": not (self.server.config.hosted_url or self.server.config.allow_remote_model),
                "hosted_configured": bool(self.server.config.hosted_url),
                "uploads_stored": False,
                "vision_endpoint": "configured" if self.server.config.vision_url else "not configured",
                "audio_endpoint": "configured" if self.server.config.audio_url else "not configured",
            },
        })

    def _conversation(self) -> None:
        session = self._session()
        messages = self.server.store.conversation_messages(session["conversation_id"], limit=200)
        self._json({"conversation_id": session["conversation_id"], "messages": messages})

    def _new_conversation(self) -> None:
        session = self._session()
        self._csrf(session)
        new_id = str(uuid.uuid4())
        self.server.store.set_web_session_conversation(session["_hash"], new_id)
        self._json({"conversation_id": new_id}, headers={"Cache-Control": "no-store"})

    # ---------------------------------------------------------------------- chat
    def _chat(self) -> None:
        session = self._session()
        self._csrf(session)
        self._rate(session, "chat", limit=self.server.config.web_rate_limit)
        payload = self._read_body(64_000)
        message = str(payload.get("message", "")).strip()
        if not message:
            raise ApiError(400, "Message cannot be empty", "empty_message")
        if len(message) > 8000:
            raise ApiError(413, "Message is too long (8000 characters max)", "too_long")
        mode = str(payload.get("mode", "chat"))
        if mode not in {"chat", "learn"}:
            raise ApiError(400, "Unknown mode", "bad_mode")
        try:
            model = get_model(self.server.config)
            engine = ConversationEngine(
                self.server.config, self.server.store, model,
                conversation_id=session["conversation_id"],
                knowledge=self.server.knowledge, embedder=self.server.embedder,
                activate=False,
            )
        except ModelError as exc:
            raise ApiError(503, f"The model configuration is invalid: {exc}", "model_unavailable") from exc
        except Exception as exc:
            raise ApiError(500, f"Ara could not prepare this conversation: {exc}", "startup_error") from exc
        backend = backend_name(model)

        self.send_response(200)
        headers = self._security_headers()
        headers.update({
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
            "Connection": "close",
        })
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.close_connection = True

        def event(payload: dict[str, Any]) -> None:
            self.wfile.write(b"data: " + _json_bytes(payload) + b"\n\n")
            self.wfile.flush()

        full = []
        try:
            event({"type": "start", "conversation_id": session["conversation_id"], "backend": backend})
            for piece in engine.stream_response(message, mode=mode):
                full.append(piece)
                event({"type": "delta", "text": piece})
            learning = engine.last_learning
            interpretation = engine.last_interpretation.as_dict() if engine.last_interpretation else None
            event({
                "type": "done",
                "text": "".join(full),
                "backend": engine.last_backend,
                "backend_error": engine.last_backend_error,
                "answer_kind": "retrieval" if engine.last_backend == "retrieval-only" else "model",
                "learning": {
                    "saved": bool(learning.saved_id),
                    "kind": learning.kind,
                    "id": learning.saved_id,
                    "sensitive_rejected": learning.sensitive_rejected,
                    "conflicts": [item.get("id") for item in (learning.conflicts or [])],
                },
                "interpretation": interpretation,
            })
            self.server.store.log_web_event(session["_hash"], "chat", message[:80])
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except ModelError as exc:
            try:
                event({"type": "error", "message": str(exc), "code": "model_error"})
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
        except Exception:
            try:
                event({"type": "error", "message": "Ara could not finish this reply.", "code": "internal"})
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

    # ---------------------------------------------------------------------- news
    def _news(self, query: dict[str, list[str]]) -> None:
        session = self._session()
        refresh = query.get("refresh", ["0"])[0] in {"1", "true"}
        offline = query.get("offline", ["0"])[0] in {"1", "true"}
        topic = query.get("topic", [None])[0]
        if refresh and not offline:
            self._rate(session, "news", limit=6)
            if time.time() - self.server._news_refresh_at < 60:
                refresh = False
            else:
                self.server._news_refresh_at = time.time()
        report = news_report(
            self.server.store, self.server.config,
            refresh=refresh, offline=offline, topic=topic,
            knowledge=self.server.knowledge,
        )
        self._json(report)

    # ------------------------------------------------------------------- nigeria
    def _nigeria(self, query: dict[str, list[str]]) -> None:
        self._session()
        text = (query.get("q", [""])[0] or "").strip()
        if not text:
            raise ApiError(400, "Add a query, for example /api/nigeria?q=no%20wahala", "missing_query")
        if len(text) > 500:
            raise ApiError(413, "Query is too long", "too_long")
        interpretation = self.server.knowledge.interpret(text)
        payload = interpretation.as_dict()
        payload["context"] = self.server.knowledge.context_for(text)
        payload["provenance"] = {
            "corpus": "curated seed corpus",
            "license": "same licence as Ara",
            "note": "Fallible reference notes with confidence scores; not universal rules.",
        }
        self._json(payload)

    # -------------------------------------------------------------------- vision
    def _vision(self) -> None:
        session = self._session()
        self._csrf(session)
        self._rate(session, "vision", limit=12)
        payload = self._read_body(12_000_000)
        data, _ = _decode_data_url(str(payload.get("image", "")), ("image/jpeg", "image/png", "image/gif", "image/webp"), 8 * 1024 * 1024)
        prompt = str(payload.get("prompt", "")).strip() or "Describe this image accurately. Point out uncertainty and do not invent details."
        if len(prompt) > 1000:
            raise ApiError(413, "Prompt is too long", "too_long")
        try:
            result = VisionClient(self.server.config).describe(data, prompt)
        except PerceptionUnavailable as exc:
            raise ApiError(503, str(exc), "vision_unavailable") from exc
        except PerceptionError as exc:
            raise ApiError(400, str(exc), "vision_error") from exc
        self.server.store.log_web_event(session["_hash"], "vision", f"{result['image']['width']}x{result['image']['height']}")
        self._json(result)

    def _transcribe(self) -> None:
        session = self._session()
        self._csrf(session)
        self._rate(session, "audio", limit=12)
        payload = self._read_body(22_000_000)
        data, _ = _decode_data_url(str(payload.get("audio", "")), ("audio/webm", "audio/ogg", "audio/wav", "audio/mpeg", "audio/mp4", "audio/x-m4a"), 15 * 1024 * 1024)
        try:
            result = Transcriber(self.server.config).transcribe(data)
        except PerceptionUnavailable as exc:
            raise ApiError(503, str(exc), "transcription_unavailable") from exc
        except PerceptionError as exc:
            raise ApiError(400, str(exc), "transcription_error") from exc
        self.server.store.log_web_event(session["_hash"], "transcribe", f"{result['bytes']} bytes")
        self._json(result)

    # -------------------------------------------------------------------- memory
    def _memory_collection(self) -> None:
        session = self._session()
        if self.command == "GET":
            self._json({"memories": self.server.store.list_memories(limit=200)})
            return
        self._csrf(session)
        self._rate(session, "memory", limit=30)
        payload = self._read_body(16_000)
        content = str(payload.get("content", "")).strip()
        kind = str(payload.get("kind", "fact")).strip()
        if not content or len(content) > 1000:
            raise ApiError(400, "Memory must be between 1 and 1000 characters", "bad_memory")
        if kind not in {"fact", "preference", "correction", "personal_history"}:
            raise ApiError(400, "Unknown memory kind", "bad_kind")
        try:
            memory_id, conflicts = self.server.store.add_memory(kind, content, source="web-user", confidence=1.0)
        except ValueError as exc:
            raise ApiError(400, str(exc), "bad_memory") from exc
        memory = self.server.store.get_memory(memory_id)
        if memory and self.server.embedder:
            try:
                self.server.store.set_memory_vector(memory_id, self.server.embedder.name, self.server.embedder.embed(content))
            except Exception:
                pass
        self._json({"memory": memory, "conflicts": [item.get("id") for item in conflicts]}, status=201)

    def _memory_correct(self, memory_id: int) -> None:
        session = self._session()
        self._csrf(session)
        self._rate(session, "memory", limit=30)
        payload = self._read_body(16_000)
        content = str(payload.get("content", "")).strip()
        if not content or len(content) > 1000:
            raise ApiError(400, "Correction must be between 1 and 1000 characters", "bad_memory")
        try:
            ok = self.server.store.correct_memory(memory_id, content)
        except ValueError as exc:
            raise ApiError(400, str(exc), "bad_memory") from exc
        if not ok:
            raise ApiError(404, "No active memory with that id", "not_found")
        if self.server.embedder:
            try:
                self.server.store.set_memory_vector(memory_id, self.server.embedder.name, self.server.embedder.embed(content))
            except Exception:
                pass
        self._json({"memory": self.server.store.get_memory(memory_id)})

    def _memory_delete(self, memory_id: int) -> None:
        session = self._session()
        self._csrf(session)
        self._rate(session, "memory", limit=30)
        if not self.server.store.forget_memory(memory_id):
            raise ApiError(404, "No active memory with that id", "not_found")
        self._json({"ok": True})

    def _memory_forget_all(self) -> None:
        session = self._session()
        self._csrf(session)
        self._rate(session, "memory", limit=5)
        payload = self._read_body(2000)
        if payload.get("confirm") != "forget all":
            raise ApiError(400, "Type 'forget all' to confirm", "confirm_required")
        count = self.server.store.forget_all_memories()
        self._json({"ok": True, "forgotten": count})

    def _memory_export(self) -> None:
        self._session()
        payload = _json_bytes({"exported_at": now(), "memories": self.server.store.list_memories(limit=10_000)})
        filename = "ara-memories-" + datetime.now(timezone.utc).strftime("%Y%m%d") + ".json"
        self._send(Response(
            payload, content_type="application/json; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        ))


def create_server(config: Config, host: str | None = None, port: int | None = None) -> AraServer:
    bind_host = host or config.web_host
    bind_port = port or config.web_port
    loopback = bind_host in {"127.0.0.1", "::1", "localhost"}
    if not loopback and not config.web_password:
        raise ValueError(
            "Refusing to bind Ara to a non-loopback address without ARA_WEB_PASSWORD. "
            "Set a strong password in the environment, then use HTTPS through a reverse proxy."
        )
    return AraServer((bind_host, bind_port), config)


def serve(config: Config, host: str | None = None, port: int | None = None) -> int:
    server = create_server(config, host, port)
    address, actual_port = server.server_address[:2]
    print(static_frog())
    print(f"\nAra web is listening on http://{address}:{actual_port}")
    print("Local-only by default. Web and terminal share memory; conversations are per session.")
    if config.web_password:
        print("Password protection is on (ARA_WEB_PASSWORD). Use HTTPS through a reverse proxy when exposed.")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        print("\nAra web stopped.")
    finally:
        server.server_close()
    return 0
