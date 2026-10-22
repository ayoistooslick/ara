"""Image, audio, and sensor-adjacent perception.

Ara does not claim that a text model can see. Vision and transcription require an
explicitly configured local OpenAI-compatible service. Uploads are validated in
memory, sent to that service, and not written to Ara's database.
"""
from __future__ import annotations

import base64
import json
import struct
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any

from .config import Config


class PerceptionError(RuntimeError):
    pass


class PerceptionUnavailable(PerceptionError):
    pass


MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_AUDIO_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000


@dataclass
class ImageInfo:
    mime: str
    width: int
    height: int
    size: int


def _png_size(data: bytes) -> tuple[int, int]:
    if data[:8] != b"\x89PNG\r\n\x1a\n" or len(data) < 24:
        raise PerceptionError("Not a valid PNG")
    return struct.unpack(">II", data[16:24])


def _gif_size(data: bytes) -> tuple[int, int]:
    if data[:6] not in (b"GIF87a", b"GIF89a") or len(data) < 10:
        raise PerceptionError("Not a valid GIF")
    return struct.unpack("<HH", data[6:10])


def _jpeg_size(data: bytes) -> tuple[int, int]:
    if not data.startswith(b"\xff\xd8\xff"):
        raise PerceptionError("Not a valid JPEG")
    index = 2
    while index + 9 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            height, width = struct.unpack(">HH", data[index + 5:index + 9])
            return width, height
        if marker in {0xD8, 0xD9}:
            index += 2
            continue
        if index + 4 > len(data):
            break
        length = struct.unpack(">H", data[index + 2:index + 4])[0]
        index += 2 + max(2, length)
    raise PerceptionError("Could not read JPEG dimensions")


def _webp_size(data: bytes) -> tuple[int, int]:
    if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise PerceptionError("Not a valid WebP")
    kind = data[12:16]
    if kind == b"VP8X":
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
        return width, height
    if kind == b"VP8 ":
        return struct.unpack("<HH", data[26:30])
    if kind == b"VP8L":
        bits = int.from_bytes(data[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    raise PerceptionError("Unsupported WebP variant")


def inspect_image(data: bytes, content_type: str | None = None) -> ImageInfo:
    if not data:
        raise PerceptionError("The image is empty")
    if len(data) > MAX_IMAGE_BYTES:
        raise PerceptionError("The image is larger than 8 MB")
    if data.startswith(b"\xff\xd8\xff"):
        mime, size = "image/jpeg", _jpeg_size(data)
    elif data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime, size = "image/png", _png_size(data)
    elif data[:6] in (b"GIF87a", b"GIF89a"):
        mime, size = "image/gif", _gif_size(data)
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        mime, size = "image/webp", _webp_size(data)
    else:
        raise PerceptionError("Only JPEG, PNG, GIF, and WebP images are supported. SVG is not accepted.")
    if content_type and content_type.split(";")[0].strip() not in {mime, "application/octet-stream"}:
        raise PerceptionError(f"Image content type {content_type!r} does not match its bytes ({mime})")
    width, height = size
    if width <= 0 or height <= 0:
        raise PerceptionError("The image has invalid dimensions")
    if width * height > MAX_IMAGE_PIXELS:
        raise PerceptionError("The image has too many pixels; use one below 40 megapixels")
    return ImageInfo(mime=mime, width=width, height=height, size=len(data))


def inspect_audio(data: bytes, filename: str = "") -> str:
    if not data:
        raise PerceptionError("The audio is empty")
    if len(data) > MAX_AUDIO_BYTES:
        raise PerceptionError("The audio is larger than 15 MB")
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        if len(data) >= 44 and data[20:22] == b"\x01\x00":
            channels, rate = struct.unpack("<HI", data[22:28])
            byte_rate = struct.unpack("<I", data[28:32])[0] or 1
            seconds = max(0, len(data) - 44) / byte_rate
            if seconds > 120:
                raise PerceptionError("Audio is longer than two minutes; send a shorter clip")
        return "audio/wav"
    if data[:4] == b"OggS":
        return "audio/ogg"
    if data[:4] == b"\x1aE\xdf\xa3":
        return "audio/webm"
    if data[:3] == b"ID3" or data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio/mpeg"
    if data[4:8] == b"ftyp":
        return "audio/mp4"
    raise PerceptionError("Unsupported audio format. Use WAV, OGG, WebM, MP3, or M4A.")


class VisionClient:
    def __init__(self, config: Config):
        self.config = config

    @property
    def available(self) -> bool:
        return bool(self.config.vision_url)

    def describe(self, data: bytes, prompt: str, content_type: str | None = None) -> dict[str, Any]:
        if not self.available:
            raise PerceptionUnavailable(
                "No vision model is configured. Set [vision] url (and model) in config.toml to a local "
                "OpenAI-compatible multimodal endpoint. Ara will not pretend a text-only model can see."
            )
        info = inspect_image(data, content_type)
        encoded = base64.b64encode(data).decode("ascii")
        body = json.dumps({
            "model": self.config.vision_model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{info.mime};base64,{encoded}"}},
                ],
            }],
            "max_tokens": min(self.config.max_tokens, 512),
            "temperature": 0.2,
        }).encode("utf-8")
        request = urllib.request.Request(
            self.config.vision_url + "/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.request_timeout) as response:
                payload = json.loads(response.read(8_000_000).decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read(1000).decode("utf-8", "replace")
            raise PerceptionError(f"Vision model failed (HTTP {exc.code}): {detail}") from exc
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise PerceptionError(f"Cannot reach the vision model: {exc}") from exc
        try:
            answer = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise PerceptionError("Vision model returned an unexpected response") from exc
        if not isinstance(answer, str) or not answer.strip():
            raise PerceptionError("Vision model returned an empty description")
        return {"text": answer.strip(), "image": {"mime": info.mime, "width": info.width, "height": info.height, "bytes": info.size}}


class Transcriber:
    def __init__(self, config: Config):
        self.config = config

    @property
    def available(self) -> bool:
        return bool(self.config.audio_url)

    def transcribe(self, data: bytes, filename: str = "speech.webm") -> dict[str, Any]:
        if not self.available:
            raise PerceptionUnavailable(
                "Speech recognition is not configured. Set [audio] url to a local OpenAI-compatible "
                "transcription endpoint (for example a whisper.cpp server). The browser can still use its "
                "own speech synthesis for spoken replies."
            )
        mime = inspect_audio(data, filename)
        boundary = "----ara" + uuid.uuid4().hex
        parts: list[bytes] = []

        def field(name: str, value: str) -> None:
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())

        field("model", self.config.audio_model)
        if self.config.audio_language:
            field("language", self.config.audio_language)
        extension = {"audio/wav": "wav", "audio/ogg": "ogg", "audio/webm": "webm", "audio/mpeg": "mp3", "audio/mp4": "m4a"}[mime]
        safe_name = "speech." + extension
        parts.append(
            (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{safe_name}\"\r\n"
             f"Content-Type: {mime}\r\n\r\n").encode() + data + b"\r\n"
        )
        parts.append(f"--{boundary}--\r\n".encode())
        body = b"".join(parts)
        request = urllib.request.Request(
            self.config.audio_url + "/audio/transcriptions",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.request_timeout) as response:
                payload = json.loads(response.read(4_000_000).decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read(1000).decode("utf-8", "replace")
            raise PerceptionError(f"Transcription failed (HTTP {exc.code}): {detail}") from exc
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise PerceptionError(f"Cannot reach the transcription service: {exc}") from exc
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise PerceptionError("The transcription service returned no text")
        return {"text": text.strip(), "mime": mime, "bytes": len(data)}


def perception_status(config: Config) -> dict[str, Any]:
    return {
        "vision": {
            "available": bool(config.vision_url),
            "endpoint": config.vision_url or "",
            "model": config.vision_model if config.vision_url else "",
            "note": "Images are sent only to the configured endpoint and are not stored by Ara." if config.vision_url else "No vision model configured; image descriptions are unavailable.",
        },
        "transcription": {
            "available": bool(config.audio_url),
            "endpoint": config.audio_url or "",
            "model": config.audio_model if config.audio_url else "",
            "note": "Audio is sent only to the configured endpoint and is not stored by Ara." if config.audio_url else "No speech-to-text endpoint configured; type instead or configure [audio].url.",
        },
        "device_sensors": {
            "terminal": "No direct phone sensor access. Files supplied by the user can be analysed.",
            "browser": "Camera and microphone require explicit browser permission and run only while the user has them active.",
        },
        "speech_synthesis": {
            "available": True,
            "backend": "browser SpeechSynthesis",
            "note": "Quality and language support depend on the device's installed voices; not verified on your hardware.",
        },
    }
