from __future__ import annotations

import hashlib
import os
import re
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

MODELS: dict[str, dict[str, object]] = {
    "qwen3-0.6b": {
        "label": "Qwen3 0.6B, 8-bit (610 MB)",
        "url": "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
        "bytes": 639_446_688,
        "ram": "about 1.5 GB free RAM",
        "license": "Apache-2.0",
        "note": "Smallest option, and the one measured for this README. Expect simple answers and mistakes.",
    },
    "qwen3-1.7b": {
        "label": "Qwen3 1.7B, 4-bit (about 1.1 GB)",
        "url": "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf",
        "bytes": 1_107_409_472,
        "ram": "about 2 GB free RAM",
        "license": "Apache-2.0 (quantised by unsloth)",
        "note": "Better than 0.6B while still plausible on a phone with swap.",
    },
    "qwen3-4b": {
        "label": "Qwen3 4B, 4-bit (about 2.3 GB)",
        "url": "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf",
        "bytes": 2_497_280_256,
        "ram": "about 4 GB free RAM",
        "license": "Apache-2.0",
        "note": "Better answers if your device has the memory. Slower on CPU.",
    },
}

_CHUNK = 1024 * 1024


class DownloadError(RuntimeError):
    pass


def human_size(number: int | float | None) -> str:
    if number is None:
        return "unknown size"
    if number == 0:
        return "0 B"
    value = float(number)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def _etag_sha256(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip().strip('"').removeprefix("W/").strip('"').lower()
    return cleaned if re.fullmatch(r"[0-9a-f]{64}", cleaned) else None


def _request(url: str, *, method: str = "GET", headers: dict[str, str] | None = None, timeout: int = 30):
    request = urllib.request.Request(url, method=method, headers={"User-Agent": "ara-model-downloader", **(headers or {})})
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise DownloadError(f"Server answered HTTP {exc.code} for {url}") from exc
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        raise DownloadError(f"Cannot reach {url}: {exc}") from exc


def probe(url: str, timeout: int = 30) -> dict[str, object]:
    """Read size and checksum hints without downloading the weights."""
    try:
        response = _request(url, method="HEAD", timeout=timeout)
    except DownloadError:
        response = _request(url, headers={"Range": "bytes=0-0"}, timeout=timeout)
    with response:
        length = response.headers.get("x-linked-size") or response.headers.get("Content-Length")
        content_range = response.headers.get("Content-Range", "")
        if not length and "/" in content_range:
            length = content_range.rsplit("/", 1)[-1]
        try:
            size = int(length) if length and length.isdigit() else None
        except (TypeError, ValueError):
            size = None
        sha = _etag_sha256(response.headers.get("x-linked-etag")) or _etag_sha256(response.headers.get("ETag"))
        return {"size": size, "sha256": sha, "url": response.geturl()}


def _sha256(path: Path, progress: Callable[[int, int], None] | None = None, total: int = 0) -> str:
    digest = hashlib.sha256()
    done = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK):
            digest.update(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)
    return digest.hexdigest()


def download(
    url: str,
    destination: Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    timeout: int = 60,
) -> dict[str, object]:
    """Resumable download with size and checksum verification.

    Writes ``<destination>.part`` and only renames it into place after the
    checksum and size checks pass, so a half-downloaded file is never used as
    a model. Existing bytes are resumed with an HTTP Range request.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    lock = destination.with_name(destination.name + ".lock")
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise DownloadError(f"Another download is already writing {destination}. Remove {lock} if that is not true.") from exc
    try:
        os.write(lock_fd, f"{os.getpid()} {time.time():.0f}\n".encode())
        os.close(lock_fd)
        info = probe(url, timeout=timeout)
        size = expected_size or (info["size"] if isinstance(info["size"], int) else None)
        sha_expected = (expected_sha256 or (info["sha256"] if isinstance(info["sha256"], str) else None))
        started = part.stat().st_size if part.exists() else 0
        if size and started > size:
            part.unlink()
            started = 0
        if size:
            free = shutil.disk_usage(destination.parent).free
            needed = max(0, size - started) + 50 * 1024 * 1024
            if free < needed:
                raise DownloadError(
                    f"Not enough disk space: need about {human_size(needed)} in {destination.parent}, "
                    f"only {human_size(free)} is free."
                )
        digest = hashlib.sha256()
        done = 0
        if started:
            with part.open("rb") as existing:
                while chunk := existing.read(_CHUNK):
                    digest.update(chunk)
                    done += len(chunk)
        headers = {"Range": f"bytes={started}-"} if started else {}
        response = _request(url, headers=headers, timeout=timeout)
        mode = "ab" if started else "wb"
        try:
            status = getattr(response, "status", 200)
            if started and status == 200:
                # The server ignored Range; restart rather than corrupt the file.
                mode = "wb"
                done = 0
                digest = hashlib.sha256()
            with part.open(mode) as stream:
                while chunk := response.read(_CHUNK):
                    stream.write(chunk)
                    digest.update(chunk)
                    done += len(chunk)
                    if on_progress:
                        on_progress(done, size or 0)
        finally:
            response.close()
        actual = digest.hexdigest()
        if size and done != size:
            raise DownloadError(f"Download ended at {human_size(done)} but the server said {human_size(size)}. Re-run to resume.")
        if sha_expected and actual != sha_expected:
            part.unlink(missing_ok=True)
            raise DownloadError("Checksum mismatch. The partial file was deleted; run the download again.")
        os.replace(part, destination)
        return {"path": destination, "bytes": done, "sha256": actual, "checksum_verified": bool(sha_expected)}
    finally:
        lock.unlink(missing_ok=True)


def catalog_entry(name: str) -> dict[str, object]:
    if name not in MODELS:
        raise DownloadError(f"Unknown model '{name}'. Choose one of: {', '.join(sorted(MODELS))}")
    return MODELS[name]
