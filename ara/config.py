from __future__ import annotations

import os
import json
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Config:
    home: Path
    model_url: str = "http://127.0.0.1:8080/v1"
    model_name: str = "Qwen3-8B"
    model_path: Path | None = None
    max_tokens: int = 512
    context_messages: int = 12
    allowed_roots: tuple[Path, ...] = ()
    allow_remote_model: bool = False
    request_timeout: int = 120

    @property
    def db_path(self) -> Path:
        return self.home / "ara.sqlite3"

    @property
    def backup_dir(self) -> Path:
        return self.home / "backups"

    @property
    def config_path(self) -> Path:
        return self.home / "config.toml"


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
    allow_remote = model.get("allow_remote", False)
    if not isinstance(allow_remote, bool):
        raise ValueError("model.allow_remote must be true or false")
    roots = tools.get("allowed_roots", [str(Path.cwd())])
    if not isinstance(roots, list) or not all(isinstance(root, str) for root in roots):
        raise ValueError("tools.allowed_roots must be a list of directory paths")
    model_path_value = model.get("path")
    cfg = Config(
        home=base,
        model_url=os.environ.get("ARA_MODEL_URL", str(model.get("url", "http://127.0.0.1:8080/v1"))).rstrip("/"),
        model_name=os.environ.get("ARA_MODEL_NAME", str(model.get("name", "Qwen3-8B"))),
        model_path=Path(os.environ.get("ARA_MODEL_PATH", model_path_value)).expanduser().resolve() if os.environ.get("ARA_MODEL_PATH", model_path_value) else None,
        max_tokens=int(os.environ.get("ARA_MAX_TOKENS", model.get("max_tokens", 512))),
        context_messages=int(os.environ.get("ARA_CONTEXT_MESSAGES", runtime.get("context_messages", 12))),
        allowed_roots=tuple(Path(p).expanduser().resolve() for p in roots),
        allow_remote_model=allow_remote,
        request_timeout=int(runtime.get("request_timeout", 120)),
    )
    if cfg.max_tokens < 1 or cfg.context_messages < 1 or cfg.request_timeout < 1:
        raise ValueError("max_tokens, context_messages, and request_timeout must be positive")
    return cfg


def write_default_config(config: Config, overwrite: bool = False) -> Path:
    config.home.mkdir(parents=True, exist_ok=True)
    if config.config_path.exists() and not overwrite:
        return config.config_path
    root = config.allowed_roots[0] if config.allowed_roots else Path.cwd()
    text = f'''# Ara configuration. Model weights and runtimes are installed separately.\n[model]\nurl = {json.dumps(config.model_url)}\nname = {json.dumps(config.model_name)}\n# path = "/path/to/model.gguf"\nallow_remote = false\nmax_tokens = {config.max_tokens}\n\n[runtime]\ncontext_messages = {config.context_messages}\nrequest_timeout = {config.request_timeout}\n\n[tools]\nallowed_roots = [{json.dumps(str(root))}]\n'''
    config.config_path.write_text(text, encoding="utf-8")
    return config.config_path
