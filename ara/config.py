from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10, common in Termux
    import tomli as tomllib


DEFAULT_MODEL_URL = "http://127.0.0.1:8080/v1"


@dataclass(frozen=True)
class Config:
    home: Path
    # auto: hosted when a hosted endpoint is configured, else the local server.
    model_backend: str = "auto"
    model_url: str = DEFAULT_MODEL_URL
    model_name: str = "Qwen3-8B"
    model_path: Path | None = None
    # Hosted OpenAI-compatible "ara-ng" service. The key is read from the
    # environment only, so it never lands in config.toml or a backup.
    hosted_url: str | None = None
    hosted_model: str = "ara-ng"
    max_tokens: int = 512
    context_messages: int = 12
    max_context_chars: int = 12000
    temperature: float = 0.7
    allowed_roots: tuple[Path, ...] = ()
    allow_remote_model: bool = False
    request_timeout: int = 120
    model_profile: str = "auto"
    # Optional local embedding server (OpenAI-compatible /embeddings).
    embedding_url: str | None = None
    embedding_model: str = "local-embeddings"
    # Optional local vision model endpoint (OpenAI-compatible chat with image parts).
    vision_url: str | None = None
    vision_model: str = "local-vision"
    # Optional local speech-to-text endpoint (OpenAI-compatible /audio/transcriptions).
    audio_url: str | None = None
    audio_model: str = "whisper-1"
    audio_language: str = "en"
    # News.
    news_timeout: int = 12
    news_max_items: int = 40
    news_feeds: tuple[str, ...] = ()
    # Web.
    web_host: str = "127.0.0.1"
    web_port: int = 8787
    web_session_hours: int = 72
    web_rate_limit: int = 60
    # Nigerian knowledge.
    knowledge_dir: Path | None = None

    @property
    def db_path(self) -> Path:
        return self.home / "ara.sqlite3"

    @property
    def backup_dir(self) -> Path:
        return self.home / "backups"

    @property
    def config_path(self) -> Path:
        return self.home / "config.toml"

    @property
    def local_models_dir(self) -> Path:
        return self.home / "models"

    @property
    def hosted_api_key(self) -> str | None:
        value = os.environ.get("ARA_HOSTED_API_KEY") or os.environ.get("ARA_API_KEY")
        return value if value and value.strip() else None

    @property
    def active_backend(self) -> str:
        """Which backend a chat turn will try first, without opening a connection."""
        if self.model_backend == "retrieval":
            return "retrieval-only"
        if self.model_backend == "hosted" or (self.model_backend == "auto" and self.hosted_url):
            return "hosted"
        return "local"

    @property
    def news_cache_path(self) -> Path:
        return self.home / "news-cache.json"

    @property
    def embedding_cache_path(self) -> Path:
        return self.home / "embeddings-cache.json"

    @property
    def web_assets(self) -> Path:
        return Path(__file__).resolve().parent / "web" / "static"

    @property
    def web_password(self) -> str | None:
        """Read only from the environment so a password never lands in config.toml."""
        value = os.environ.get("ARA_WEB_PASSWORD")
        return value if value else None

    @property
    def web_insecure_bind(self) -> bool:
        """Explicit opt-in for a non-loopback bind without a password.

        Only for isolated containers (e.g. a managed preview sandbox) where the
        platform owns the port and terminates all outside access. Never set it on
        a host that is reachable from an untrusted network.
        """
        return os.environ.get("ARA_WEB_INSECURE_BIND", "").strip().lower() in {"1", "true", "yes"}


def _env(name: str, fallback: Any) -> Any:
    value = os.environ.get(name)
    return fallback if value is None or value == "" else value


def _int(value: Any, label: str, minimum: int = 1) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a whole number") from exc
    if number < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return number


def _float(value: Any, label: str, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc
    if not low <= number <= high:
        raise ValueError(f"{label} must be between {low} and {high}")
    return number


def _optional_url(value: Any, label: str) -> str | None:
    if value is None or value == "":
        return None
    text = str(value).strip().rstrip("/")
    if not text.startswith(("http://", "https://")):
        raise ValueError(f"{label} must be an http(s) URL or empty")
    return text


def load_config(home: str | Path | None = None) -> Config:
    base = Path(home or os.environ.get("ARA_HOME", "~/.local/share/ara")).expanduser().resolve()
    raw: dict[str, Any] = {}
    config_file = base / "config.toml"
    if config_file.exists():
        try:
            with config_file.open("rb") as stream:
                raw = tomllib.load(stream)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"Cannot read {config_file}: {exc}") from exc

    model = raw.get("model", {})
    tools = raw.get("tools", {})
    runtime = raw.get("runtime", {})
    embeddings = raw.get("embeddings", {})
    vision = raw.get("vision", {})
    audio = raw.get("audio", {})
    news = raw.get("news", {})
    web = raw.get("web", {})
    knowledge = raw.get("knowledge", {})
    for section_name, section in [
        ("model", model), ("tools", tools), ("runtime", runtime),
        ("embeddings", embeddings), ("vision", vision), ("audio", audio),
        ("news", news), ("web", web), ("knowledge", knowledge),
    ]:
        if not isinstance(section, dict):
            raise ValueError(f"[{section_name}] must be a table")

    allow_remote = model.get("allow_remote", False)
    if not isinstance(allow_remote, bool):
        raise ValueError("model.allow_remote must be true or false")
    roots = tools.get("allowed_roots", [str(Path.cwd())])
    if not isinstance(roots, list) or not all(isinstance(root, str) for root in roots):
        raise ValueError("tools.allowed_roots must be a list of directory paths")
    feeds = news.get("feeds", [])
    if not isinstance(feeds, list) or not all(isinstance(feed, str) for feed in feeds):
        raise ValueError("news.feeds must be a list of feed URLs")

    model_path_value = model.get("path")
    profile = str(_env("ARA_MODEL_PROFILE", model.get("profile", "auto"))).strip().lower()
    if profile not in {"auto", "constrained", "standard", "powerful"}:
        raise ValueError("model.profile must be auto, constrained, standard, or powerful")
    backend = str(_env("ARA_BACKEND", model.get("backend", "auto"))).strip().lower()
    if backend not in {"auto", "hosted", "local", "retrieval"}:
        raise ValueError("model.backend must be auto, hosted, local, or retrieval")
    knowledge_dir = knowledge.get("dir")
    cfg = Config(
        home=base,
        model_backend=backend,
        model_url=str(_env("ARA_MODEL_URL", model.get("url", DEFAULT_MODEL_URL))).rstrip("/"),
        model_name=str(_env("ARA_MODEL_NAME", model.get("name", "Qwen3-8B"))),
        hosted_url=_optional_url(_env("ARA_HOSTED_URL", model.get("hosted_url")), "model.hosted_url"),
        hosted_model=str(_env("ARA_HOSTED_MODEL", model.get("hosted_model", "ara-ng"))).strip() or "ara-ng",
        model_path=Path(_env("ARA_MODEL_PATH", model_path_value)).expanduser().resolve() if _env("ARA_MODEL_PATH", model_path_value) else None,
        max_tokens=_int(_env("ARA_MAX_TOKENS", model.get("max_tokens", 512)), "model.max_tokens"),
        context_messages=_int(_env("ARA_CONTEXT_MESSAGES", runtime.get("context_messages", 12)), "runtime.context_messages"),
        max_context_chars=_int(runtime.get("max_context_chars", 12000), "runtime.max_context_chars", 1000),
        temperature=_float(_env("ARA_TEMPERATURE", model.get("temperature", 0.7)), "model.temperature", 0.0, 2.0),
        allowed_roots=tuple(Path(p).expanduser().resolve() for p in roots),
        allow_remote_model=allow_remote,
        request_timeout=_int(runtime.get("request_timeout", 120), "runtime.request_timeout"),
        model_profile=profile,
        embedding_url=_optional_url(_env("ARA_EMBEDDING_URL", embeddings.get("url")), "embeddings.url"),
        embedding_model=str(_env("ARA_EMBEDDING_MODEL", embeddings.get("model", "local-embeddings"))),
        vision_url=_optional_url(_env("ARA_VISION_URL", vision.get("url")), "vision.url"),
        vision_model=str(_env("ARA_VISION_MODEL", vision.get("model", "local-vision"))),
        audio_url=_optional_url(_env("ARA_AUDIO_URL", audio.get("url")), "audio.url"),
        audio_model=str(_env("ARA_AUDIO_MODEL", audio.get("model", "whisper-1"))),
        audio_language=str(audio.get("language", "en")),
        news_timeout=_int(news.get("timeout", 12), "news.timeout"),
        news_max_items=_int(news.get("max_items", 40), "news.max_items"),
        news_feeds=tuple(feeds),
        web_host=str(_env("ARA_WEB_HOST", web.get("host", "127.0.0.1"))),
        web_port=_int(_env("ARA_WEB_PORT", web.get("port", _env("PORT", 8787))), "web.port"),
        web_session_hours=_int(web.get("session_hours", 72), "web.session_hours"),
        web_rate_limit=_int(web.get("rate_limit", 60), "web.rate_limit"),
        knowledge_dir=Path(knowledge_dir).expanduser().resolve() if knowledge_dir else None,
    )
    if cfg.web_port > 65535:
        raise ValueError("web.port must be 65535 or lower")
    if not cfg.model_url.startswith(("http://", "https://")):
        raise ValueError("model.url must be an http(s) URL")
    return cfg


def write_default_config(config: Config, overwrite: bool = False) -> Path:
    config.home.mkdir(parents=True, exist_ok=True)
    if config.config_path.exists() and not overwrite:
        return config.config_path
    root = config.allowed_roots[0] if config.allowed_roots else Path.cwd()
    text = f'''# Ara configuration. Model weights and runtimes are installed separately.
# Every value can be overridden with an ARA_* environment variable; see docs/MODELS.md.
[model]
# auto uses hosted "ara-ng" when hosted_url is set, else the local server.
backend = {json.dumps(config.model_backend)}
# hosted_url = "https://your-ara-ng-endpoint.example/v1"
# The hosted API key is read from ARA_HOSTED_API_KEY, never from this file.
hosted_model = {json.dumps(config.hosted_model)}
url = {json.dumps(config.model_url)}
name = {json.dumps(config.model_name)}
# path = "/path/to/model.gguf"
allow_remote = false
profile = "auto"
max_tokens = {config.max_tokens}
temperature = {config.temperature}

[runtime]
context_messages = {config.context_messages}
max_context_chars = {config.max_context_chars}
request_timeout = {config.request_timeout}

[tools]
allowed_roots = [{json.dumps(str(root))}]

# Optional local OpenAI-compatible services. Leave unset to use the built-in
# hashed-vector memory fallback and to keep vision/transcription clearly disabled.
[embeddings]
# url = "http://127.0.0.1:8081/v1"
# model = "local-embeddings"

[vision]
# url = "http://127.0.0.1:8082/v1"
# model = "local-vision"

[audio]
# url = "http://127.0.0.1:8083/v1"
# model = "whisper-1"
language = "en"

[news]
timeout = {config.news_timeout}
max_items = {config.news_max_items}
# feeds = ["https://www.premiumtimesng.com/feed"]

[web]
host = "127.0.0.1"
port = {config.web_port}
session_hours = {config.web_session_hours}
rate_limit = {config.web_rate_limit}

[knowledge]
# dir = "/path/to/extra/nigerian-knowledge"
'''
    config.config_path.write_text(text, encoding="utf-8")
    return config.config_path


def available_memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None


def effective_profile(config: Config) -> str:
    """Resolve 'auto' to a RAM-based suggestion. This is a starting point, not a benchmark."""
    if config.model_profile != "auto":
        return config.model_profile
    memory = available_memory_bytes()
    if memory is None:
        return "standard"
    if memory < 3 * 1024**3:
        return "constrained"
    if memory < 10 * 1024**3:
        return "standard"
    return "powerful"


MODEL_PROFILES: dict[str, dict[str, object]] = {
    "constrained": {
        "label": "Constrained (about 2-4 GB available RAM)",
        "suggested_gguf": "Qwen3-0.6B Q8_0 (about 0.6 GB) or Qwen3-1.7B Q4_K_M (about 1.1 GB)",
        "max_tokens": 256,
        "context_messages": 6,
        "note": "Short context and small token budgets keep latency usable on entry-level Android phones.",
    },
    "standard": {
        "label": "Standard (about 4-10 GB available RAM)",
        "suggested_gguf": "Qwen3-4B Q4_K_M (about 2.5 GB) or Qwen3-8B Q4_K_M (about 4.7 GB)",
        "max_tokens": 512,
        "context_messages": 12,
        "note": "Balanced quality and speed for common laptops and mid-range phones with swap.",
    },
    "powerful": {
        "label": "Powerful (about 10 GB+ available RAM or a GPU)",
        "suggested_gguf": "Qwen3-8B Q5_K_M or a compatible 14B model in 4-bit",
        "max_tokens": 1024,
        "context_messages": 20,
        "note": "Larger models need proportionally more memory for the KV cache.",
    },
}


def compatibility_report(config: Config) -> list[str]:
    """Describe fit using observed file size and current free memory, without promising speed."""
    profile = effective_profile(config)
    details = MODEL_PROFILES[profile]
    memory = available_memory_bytes()
    lines = [
        f"Requested profile: {config.model_profile} (resolved: {profile})",
        f"{details['label']}: {details['suggested_gguf']}",
        f"Note: {details['note']}",
    ]
    if memory is not None:
        lines.append(f"Free memory right now: {memory / 1024**3:.2f} GiB. Leave room for the KV cache and the operating system.")
    if config.model_path and config.model_path.is_file():
        size = config.model_path.stat().st_size
        lines.append(f"Configured weights: {config.model_path.name} ({size / 1024**3:.2f} GiB)")
        if memory is not None and size > memory * 0.75:
            lines.append("Risk: the weights alone use most free memory; loading will probably fail. Choose a smaller quantization or add swap.")
    lines.append("These profiles are starting suggestions. Actual speed depends on CPU, thermal limits, and the inference backend. Measure with 'ara model status'.")
    return lines
