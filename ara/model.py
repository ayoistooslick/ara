from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Iterator
from urllib.parse import urlparse

from .config import Config


class ModelError(RuntimeError):
    pass


class BackendUnavailable(ModelError):
    """No configured backend answered, so Ara must not claim model inference."""


def estimate_tokens(text: str) -> int:
    """Rough token estimate for budgeting. It is not a tokenizer measurement."""
    return max(1, len(text) // 4)


def fit_messages(messages: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
    """Keep the system prompt and the newest turns within a character budget."""
    if not messages:
        return []
    system = [message for message in messages if message.get("role") == "system"]
    rest = [message for message in messages if message.get("role") != "system"]
    budget = max(500, max_chars - sum(len(str(message.get("content", ""))) for message in system[:1]))
    kept: list[dict[str, Any]] = []
    used = 0
    for message in reversed(rest):
        content = str(message.get("content", ""))
        cost = len(content)
        if kept and used + cost > budget:
            break
        if not kept and cost > budget:
            message = dict(message)
            message["content"] = content[-budget:]
            cost = budget
        kept.append(message)
        used += cost
    kept.reverse()
    return system[:1] + kept


class ModelAdapter:
    """Abstract interface for language model backends.

    Ara's identity is separate from any specific model. The adapter translates
    between Ara's prompt format and the backend's API.
    """

    def status(self) -> tuple[bool, str]:
        raise NotImplementedError

    def generate(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError

    def stream(self, messages: list[dict[str, Any]], cancel: Callable[[], bool] | None = None) -> Iterator[str]:
        raise NotImplementedError

    def capabilities(self) -> dict[str, Any]:
        return {"backend": "abstract", "streaming": False, "vision": False}


class ChatCompletionsModel(ModelAdapter):
    """OpenAI-compatible chat client used by both the hosted and local backends.

    Streaming, timeouts, context fitting, and error text live here once. The
    backend label and the endpoint policy are supplied by the subclass.
    """

    backend = "openai-compatible"

    def __init__(
        self,
        config: Config,
        url: str,
        model_name: str,
        *,
        api_key: str | None = None,
        timeout: int | None = None,
    ):
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ModelError("Model endpoint must be a valid http(s) URL")
        self.config = config
        self.url = url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key or None
        self.timeout = timeout or config.request_timeout
        self.label = f"{model_name} ({self.backend})"

    def _headers(self, accept: str) -> dict[str, str]:
        headers = {"Accept": accept}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        return headers

    def status(self) -> tuple[bool, str]:
        request = urllib.request.Request(self.url + "/models", headers=self._headers("application/json"))
        try:
            with urllib.request.urlopen(request, timeout=4) as response:
                payload = json.loads(response.read(1_000_000).decode("utf-8"))
            names = [str(item.get("id", "")) for item in payload.get("data", []) if isinstance(item, dict)]
            where = "hosted ara-ng" if self.backend == "hosted" else "local inference"
            return True, f"Connected to {where} server at {self.url}" + (f"; served model(s): {', '.join(names)}" if names else "")
        except urllib.error.HTTPError as exc:
            return False, f"The {self.backend} server at {self.url} answered HTTP {exc.code}"
        except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
            if self.backend == "hosted":
                return False, f"No hosted ara-ng server responded at {self.url}: {exc}"
            return False, f"No compatible model server responded at {self.url}: {exc}"

    def capabilities(self) -> dict[str, Any]:
        ok, message = self.status()
        models: list[str] = []
        context_length = None
        if ok:
            try:
                with urllib.request.urlopen(
                    urllib.request.Request(self.url + "/models", headers=self._headers("application/json")),
                    timeout=4,
                ) as response:
                    payload = json.loads(response.read(1_000_000).decode("utf-8"))
                models = [str(item.get("id")) for item in payload.get("data", []) if isinstance(item, dict)]
            except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
                pass
            if self.backend == "local":
                # llama.cpp exposes context size at /props; other servers may not.
                base = self.url.rsplit("/v1", 1)[0]
                try:
                    with urllib.request.urlopen(base + "/props", timeout=3) as response:
                        props = json.loads(response.read(1_000_000).decode("utf-8"))
                    context_length = props.get("default_generation_settings", {}).get("n_ctx") or props.get("n_ctx")
                except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError, AttributeError):
                    pass
        return {
            "backend": self.backend,
            "reachable": ok,
            "message": message,
            "models": models,
            "context_length": context_length,
            "streaming": True,
            "vision": False,
            "vision_note": "This text endpoint is not assumed to understand images. Configure [vision].url for a real vision model.",
            "configured_model": self.model_name,
            "served_model": models[0] if models else None,
            "endpoint": self.url,
        }

    def _request_body(self, messages: list[dict[str, Any]], stream: bool) -> bytes:
        fitted = fit_messages(messages, self.config.max_context_chars)
        return json.dumps({
            "model": self.model_name,
            "messages": fitted,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": stream,
        }).encode("utf-8")

    def _open(self, body: bytes, stream: bool):
        request = urllib.request.Request(
            self.url + "/chat/completions",
            data=body,
            headers={
                **self._headers("text/event-stream, application/json" if stream else "application/json"),
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            return urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read(2000).decode("utf-8", "replace")
            if exc.code in {401, 403}:
                raise ModelError(f"The {self.backend} server rejected the credentials (HTTP {exc.code})") from exc
            if exc.code == 429:
                raise ModelError(f"The {self.backend} server is rate limiting requests (HTTP 429)") from exc
            raise ModelError(f"{self.backend.capitalize()} model request failed (HTTP {exc.code}): {detail}") from exc
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            raise BackendUnavailable(f"Could not reach the {self.backend} model server at {self.url}: {exc}") from exc

    def generate(self, messages: list[dict[str, str]]) -> str:
        response = self._open(self._request_body(messages, stream=False), stream=False)
        try:
            with response:
                result = json.loads(response.read(8_000_000).decode("utf-8"))
            answer = result["choices"][0]["message"]["content"]
            if not isinstance(answer, str) or not answer.strip():
                raise ModelError(f"The {self.backend} model returned an empty response")
            return answer.strip()
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ModelError(f"The {self.backend} model returned an invalid chat response: {exc}") from exc
        except (OSError, TimeoutError) as exc:
            raise ModelError(f"Reading the {self.backend} model response failed: {exc}") from exc

    def stream(self, messages: list[dict[str, Any]], cancel: Callable[[], bool] | None = None) -> Iterator[str]:
        """Yield text deltas from an OpenAI-compatible streaming response.

        A server that ignores stream=true still works: its full JSON body is parsed
        and yielded once. Cancellation is cooperative between chunks.
        """
        response = self._open(self._request_body(messages, stream=True), stream=True)
        content_type = response.headers.get("Content-Type", "")
        try:
            if "text/event-stream" not in content_type:
                raw = response.read(8_000_000).decode("utf-8")
                try:
                    payload = json.loads(raw)
                    content = payload["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                    raise ModelError(f"The {self.backend} model returned an invalid chat response: {exc}") from exc
                if content and str(content).strip():
                    yield str(content)
                return
            for raw_line in response:
                if cancel and cancel():
                    return
                line = raw_line.decode("utf-8", "replace").strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    return
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    delta = payload["choices"][0].get("delta", {})
                    piece = delta.get("content")
                    if not piece:
                        piece = payload["choices"][0].get("text")
                except (KeyError, IndexError, TypeError):
                    continue
                if piece:
                    yield str(piece)
        except (OSError, TimeoutError) as exc:
            raise ModelError(f"Streaming from the {self.backend} model failed: {exc}") from exc
        finally:
            response.close()


class LocalChatModel(ChatCompletionsModel):
    """Client for a local OpenAI-compatible inference server, including llama.cpp."""

    backend = "local"

    def __init__(self, config: Config):
        parsed = urlparse(config.model_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ModelError("Model URL must be a valid http(s) URL")
        is_loopback = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        if not is_loopback and not config.allow_remote_model:
            raise ModelError("Ara permits only a local model endpoint by default. Set model.allow_remote=true in config.toml to allow a remote endpoint explicitly.")
        if parsed.scheme == "https" and is_loopback:
            raise ModelError("Use http for a local model server")
        super().__init__(config, config.model_url, config.model_name)


class HostedChatModel(ChatCompletionsModel):
    """Client for the hosted ara-ng service.

    The endpoint must be HTTPS unless it is loopback, and the key is only read
    from the environment. Nothing is sent until the user sends a message.
    """

    backend = "hosted"

    def __init__(self, config: Config):
        if not config.hosted_url:
            raise ModelError("No hosted endpoint is configured. Set model.hosted_url or ARA_HOSTED_URL, or use a local model.")
        parsed = urlparse(config.hosted_url)
        if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ModelError("The hosted ara-ng endpoint must use https (http is allowed only for a local test server).")
        super().__init__(
            config,
            config.hosted_url,
            config.hosted_model,
            api_key=config.hosted_api_key,
            timeout=min(config.request_timeout, 60),
        )


def get_model(config: Config) -> ModelAdapter | None:
    """Pick the first configured backend. None means retrieval-only, honestly."""
    if config.model_backend == "retrieval":
        return None
    if config.model_backend in {"auto", "hosted"} and config.hosted_url:
        return HostedChatModel(config)
    if config.model_backend == "hosted":
        raise ModelError("model.backend is hosted but no hosted_url is set. Add one, or set backend = \"auto\".")
    return LocalChatModel(config)


def backend_name(model: ModelAdapter | None) -> str:
    if model is None:
        return "retrieval-only"
    backend = getattr(model, "backend", "openai-compatible")
    return "hosted" if backend == "hosted" else "local"


def backend_summary(config: Config, model: ModelAdapter | None) -> str:
    """One short line for welcome screens; never a network call."""
    if model is None:
        return "retrieval-only (no model configured; answers are retrieved notes, not model text)"
    if getattr(model, "backend", "") == "hosted":
        key = "key set" if config.hosted_api_key else "no API key set"
        return f"hosted ara-ng at {config.hosted_url} ({key})"
    return f"local model at {config.model_url} (not checked yet)"


def runtime_diagnostics(config: Config) -> list[str]:
    available = None
    try:
        available = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        pass
    lines = [
        f"Backend: {config.active_backend}",
        f"Hosted ara-ng: {config.hosted_url or 'not configured'}; API key: {'set' if config.hosted_api_key else 'not set'}",
        f"Local server: {config.model_url}",
        f"Local models folder: {config.local_models_dir}",
    ]
    if available is not None:
        lines.append(f"Currently available system memory: {available / (1024**3):.2f} GiB. This snapshot is not a model-fit guarantee; context cache and runtime overhead need additional memory.")
    if config.model_path:
        if config.model_path.is_file():
            size = config.model_path.stat().st_size
            lines.append(f"Configured model file: {config.model_path} ({size / (1024**3):.2f} GiB)")
            if available is not None and size > available:
                lines.append("Warning: model file exceeds currently available memory; loading is unlikely to succeed without adequate swap or accelerator memory.")
        else:
            lines.append(f"Configured model file does not exist: {config.model_path}")
    else:
        lines.append("No local GGUF path configured; Ara does not download weights automatically.")
    return lines
