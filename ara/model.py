from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Iterable, Iterator
from urllib.parse import urlparse

from .config import Config


class ModelError(RuntimeError):
    pass


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
    between Ara's prompt format and the backend's API. Swap the backend without
    changing the rest of the application.
    """

    def status(self) -> tuple[bool, str]:
        raise NotImplementedError

    def generate(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError

    def stream(self, messages: list[dict[str, Any]], cancel: Callable[[], bool] | None = None) -> Iterator[str]:
        raise NotImplementedError

    def capabilities(self) -> dict[str, Any]:
        return {"backend": "abstract", "streaming": False, "vision": False}


class LocalChatModel(ModelAdapter):
    """Client for a local OpenAI-compatible inference server, including llama.cpp.

    The configured model_name is sent as a label to the server; the server decides
    which loaded model actually responds. Ara does not care whether the weights are
    Qwen, Mistral, Llama, or another supported model.
    """

    def __init__(self, config: Config):
        self.config = config
        parsed = urlparse(config.model_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ModelError("Model URL must be a valid http(s) URL")
        is_loopback = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        if not is_loopback and not config.allow_remote_model:
            raise ModelError("Ara permits only a local model endpoint by default. Set model.allow_remote=true in config.toml to allow a remote endpoint explicitly.")
        if parsed.scheme == "https" and is_loopback:
            raise ModelError("Use http for a local model server")

    def status(self) -> tuple[bool, str]:
        endpoint = self.config.model_url + "/models"
        req = urllib.request.Request(endpoint, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=3) as response:
                payload = json.loads(response.read(1_000_000).decode("utf-8"))
            names = [str(item.get("id", "")) for item in payload.get("data", []) if isinstance(item, dict)]
            return True, f"Connected to local inference server at {self.config.model_url}" + (f"; served model(s): {', '.join(names)}" if names else "")
        except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
            return False, f"No compatible model server responded at {self.config.model_url}: {exc}"

    def capabilities(self) -> dict[str, Any]:
        ok, message = self.status()
        models: list[str] = []
        context_length = None
        props: dict[str, Any] = {}
        if ok:
            try:
                with urllib.request.urlopen(self.config.model_url + "/models", timeout=3) as response:
                    payload = json.loads(response.read(1_000_000).decode("utf-8"))
                models = [str(item.get("id")) for item in payload.get("data", []) if isinstance(item, dict)]
            except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
                pass
            # llama.cpp exposes context size at /props; other servers may not.
            base = self.config.model_url.rsplit("/v1", 1)[0]
            try:
                with urllib.request.urlopen(base + "/props", timeout=3) as response:
                    props = json.loads(response.read(1_000_000).decode("utf-8"))
                context_length = props.get("default_generation_settings", {}).get("n_ctx") or props.get("n_ctx")
            except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError, AttributeError):
                props = {}
        return {
            "backend": "openai-compatible",
            "reachable": ok,
            "message": message,
            "models": models,
            "context_length": context_length,
            "streaming": True,
            "vision": False,
            "vision_note": "This text endpoint is not assumed to understand images. Configure [vision].url for a real vision model.",
            "configured_model": self.config.model_name,
        }

    def _request_body(self, messages: list[dict[str, Any]], stream: bool) -> bytes:
        fitted = fit_messages(messages, self.config.max_context_chars)
        return json.dumps({
            "model": self.config.model_name,
            "messages": fitted,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": stream,
        }).encode("utf-8")

    def generate(self, messages: list[dict[str, str]]) -> str:
        body = self._request_body(messages, stream=False)
        req = urllib.request.Request(
            self.config.model_url + "/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.request_timeout) as response:
                result = json.loads(response.read(8_000_000).decode("utf-8"))
            answer = result["choices"][0]["message"]["content"]
            if not isinstance(answer, str) or not answer.strip():
                raise ModelError("The local model returned an empty response")
            return answer.strip()
        except urllib.error.HTTPError as exc:
            detail = exc.read(2000).decode("utf-8", "replace")
            raise ModelError(f"Local model request failed (HTTP {exc.code}): {detail}") from exc
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            raise ModelError(f"Could not reach the local model server: {exc}") from exc
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ModelError(f"Local model returned an invalid chat response: {exc}") from exc

    def stream(self, messages: list[dict[str, Any]], cancel: Callable[[], bool] | None = None) -> Iterator[str]:
        """Yield text deltas from an OpenAI-compatible streaming response.

        A server that ignores stream=true still works: its full JSON body is parsed
        and yielded once. Cancellation is cooperative between chunks.
        """
        body = self._request_body(messages, stream=True)
        req = urllib.request.Request(
            self.config.model_url + "/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream, application/json"},
            method="POST",
        )
        try:
            response = urllib.request.urlopen(req, timeout=self.config.request_timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read(2000).decode("utf-8", "replace")
            raise ModelError(f"Local model request failed (HTTP {exc.code}): {detail}") from exc
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            raise ModelError(f"Could not reach the local model server: {exc}") from exc
        content_type = response.headers.get("Content-Type", "")
        try:
            if "text/event-stream" not in content_type:
                raw = response.read(8_000_000).decode("utf-8")
                try:
                    payload = json.loads(raw)
                    content = payload["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                    raise ModelError(f"Local model returned an invalid chat response: {exc}") from exc
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
            raise ModelError(f"Streaming from the local model failed: {exc}") from exc
        finally:
            response.close()


def runtime_diagnostics(config: Config) -> list[str]:
    available = None
    try:
        available = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        pass
    lines = [f"Python: local runtime available; configured server: {config.model_url}"]
    if available is not None:
        lines.append(f"Currently available system memory: {available / (1024**3):.2f} GiB. This snapshot is not a model-fit guarantee; context cache and runtime overhead need additional memory.")
    if config.model_path:
        if config.model_path.is_file():
            size = config.model_path.stat().st_size
            lines.append(f"Configured model file: {config.model_path} ({size / (1024**3):.2f} GiB)")
            if available is not None:
                if size > available:
                    lines.append("Warning: model file exceeds currently available memory; loading is unlikely to succeed without adequate swap or accelerator memory.")
        else:
            lines.append(f"Configured model file does not exist: {config.model_path}")
    else:
        lines.append("No local GGUF path configured; Ara does not download weights automatically.")
    return lines
