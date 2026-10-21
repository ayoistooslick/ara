from __future__ import annotations

import json
import urllib.error
import urllib.request
from urllib.parse import urlparse

from .config import Config


class ModelError(RuntimeError):
    pass


class LocalChatModel:
    """Client for a local OpenAI-compatible inference server, including llama.cpp."""

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

    def generate(self, messages: list[dict[str, str]]) -> str:
        body = json.dumps({"model": self.config.model_name, "messages": messages, "temperature": 0.7, "max_tokens": self.config.max_tokens, "stream": False}).encode("utf-8")
        req = urllib.request.Request(self.config.model_url + "/chat/completions", data=body, headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST")
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


def runtime_diagnostics(config: Config) -> list[str]:
    available = None
    try:
        import os
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
