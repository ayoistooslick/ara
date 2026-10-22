from __future__ import annotations

import base64
import json
import struct
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ara.config import Config
from ara.perception import (
    MAX_IMAGE_BYTES,
    PerceptionError,
    PerceptionUnavailable,
    Transcriber,
    VisionClient,
    inspect_audio,
    inspect_image,
    perception_status,
)


def png(width: int = 64, height: int = 48) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
        + b"\x08\x06\x00\x00\x00"
        + b"\x00\x00\x00\x00"
    )


def jpeg(width: int = 640, height: int = 480) -> bytes:
    body = b"\xff\xd8\xff\xe0\x00\x10" + b"0" * 14
    body += b"\xff\xc0\x00\x11\x08" + struct.pack(">HH", height, width) + b"\x03" + b"\x00" * 9
    return body


class StubPerception(BaseHTTPRequestHandler):
    requests: list[tuple[str, bytes]] = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        StubPerception.requests.append((self.path, raw))
        if self.path.endswith("/chat/completions"):
            payload = json.loads(raw)
            assert "image_url" in payload["messages"][0]["content"][1]
            body = json.dumps({"choices": [{"message": {"content": "A green frog on a white background."}}]}).encode()
        elif self.path.endswith("/audio/transcriptions"):
            body = json.dumps({"text": "How far?"}).encode()
        else:
            body = b"{}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


class PerceptionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.config = Config(home=self.home, allowed_roots=(self.home,))
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), StubPerception)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}/v1"
        StubPerception.requests = []

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def test_png_and_jpeg_dimensions(self):
        info = inspect_image(png(320, 200))
        self.assertEqual((info.mime, info.width, info.height), ("image/png", 320, 200))
        info = inspect_image(jpeg(1280, 720))
        self.assertEqual((info.mime, info.width, info.height), ("image/jpeg", 1280, 720))

    def test_svg_upload_is_rejected(self):
        with self.assertRaises(PerceptionError):
            inspect_image(b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>', "image/svg+xml")

    def test_content_type_must_match_bytes(self):
        with self.assertRaises(PerceptionError):
            inspect_image(png(), "image/jpeg")

    def test_oversized_and_empty_images_rejected(self):
        with self.assertRaises(PerceptionError):
            inspect_image(b"")
        with self.assertRaises(PerceptionError):
            inspect_image(b"\x89PNG\r\n\x1a\n" + b"0" * (MAX_IMAGE_BYTES + 1))

    def test_absurd_pixel_count_rejected(self):
        with self.assertRaises(PerceptionError):
            inspect_image(png(20_000, 20_000))

    def test_audio_magic_and_duration(self):
        self.assertEqual(inspect_audio(b"\x1aE\xdf\xa3" + b"0" * 40), "audio/webm")
        self.assertEqual(inspect_audio(b"RIFF\x00\x00\x00\x00WAVE" + b"0" * 40), "audio/wav")
        with self.assertRaises(PerceptionError):
            inspect_audio(b"not audio at all")
        with self.assertRaises(PerceptionError):
            inspect_audio(b"")

    def test_vision_unconfigured_is_explicit(self):
        with self.assertRaises(PerceptionUnavailable):
            VisionClient(self.config).describe(png(), "Describe")
        status = perception_status(self.config)
        self.assertFalse(status["vision"]["available"])
        self.assertIn("No vision model", status["vision"]["note"])

    def test_vision_posts_real_data_url_and_returns_text(self):
        config = Config(home=self.home, allowed_roots=(self.home,), vision_url=self.base, vision_model="stub-vision")
        result = VisionClient(config).describe(png(100, 80), "What is this?")
        self.assertEqual(result["text"], "A green frog on a white background.")
        self.assertEqual(result["image"]["width"], 100)
        path, raw = StubPerception.requests[-1]
        self.assertEqual(path, "/v1/chat/completions")
        payload = json.loads(raw)
        self.assertEqual(payload["model"], "stub-vision")
        image_url = payload["messages"][0]["content"][1]["image_url"]["url"]
        self.assertTrue(image_url.startswith("data:image/png;base64,"))
        self.assertEqual(base64.b64decode(image_url.split(",", 1)[1])[:8], b"\x89PNG\r\n\x1a\n")

    def test_transcription_unconfigured_is_explicit(self):
        with self.assertRaises(PerceptionUnavailable):
            Transcriber(self.config).transcribe(b"\x1aE\xdf\xa3" + b"0" * 20)

    def test_transcription_multipart_contract(self):
        config = Config(home=self.home, allowed_roots=(self.home,), audio_url=self.base, audio_model="stub-whisper")
        result = Transcriber(config).transcribe(b"\x1aE\xdf\xa3" + b"0" * 64)
        self.assertEqual(result["text"], "How far?")
        path, raw = StubPerception.requests[-1]
        self.assertEqual(path, "/v1/audio/transcriptions")
        self.assertIn(b'name="model"', raw)
        self.assertIn(b"stub-whisper", raw)
        self.assertIn(b'filename="speech.webm"', raw)
        self.assertIn(b"Content-Type: audio/webm", raw)

    def test_perception_status_does_not_claim_unconfigured_capabilities(self):
        status = perception_status(self.config)
        self.assertFalse(status["transcription"]["available"])
        self.assertIn("browser", status["device_sensors"]["browser"].casefold())
        self.assertIn("not verified", status["speech_synthesis"]["note"].casefold())


if __name__ == "__main__":
    unittest.main()
