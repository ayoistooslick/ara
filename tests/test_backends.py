from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ara.config import Config, load_config
from ara.downloads import MODELS, DownloadError, download, human_size, probe
from ara.engine import ConversationEngine
from ara.model import HostedChatModel, LocalChatModel, ModelError, backend_name, get_model
from ara.store import Store

PAYLOAD = b"ara-model-bytes-" * 500
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


class HostedHandler(BaseHTTPRequestHandler):
    calls: list[dict] = []

    def do_GET(self):
        body = json.dumps({"data": [{"id": "ara-ng"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        HostedHandler.calls.append({"payload": payload, "authorization": self.headers.get("Authorization")})
        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for piece in ("Kedu", " from ", "ara-ng"):
                self.wfile.write(("data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        body = json.dumps({"choices": [{"message": {"content": "Kedu from ara-ng"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


class RangeHandler(BaseHTTPRequestHandler):
    requests: list[str] = []

    def _head(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(PAYLOAD)))
        self.send_header("x-linked-etag", PAYLOAD_SHA)
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_HEAD(self):
        RangeHandler.requests.append("HEAD")
        self._head()

    def do_GET(self):
        header = self.headers.get("Range", "")
        RangeHandler.requests.append(header or "GET")
        start = 0
        if header.startswith("bytes="):
            start = int(header[6:].split("-", 1)[0] or 0)
        body = PAYLOAD[start:]
        self.send_response(206 if header else 200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("x-linked-etag", PAYLOAD_SHA)
        if header:
            self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


def start(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


class FakeFailing:
    def generate(self, messages):
        raise ModelError("stub backend is down")

    def stream(self, messages, cancel=None):
        raise ModelError("stub backend is down")


class FakeEmptyStream:
    def stream(self, messages, cancel=None):
        return iter(())


class HostedBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), HostedHandler)
        self.thread = start(self.server)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def test_auto_prefers_hosted_when_configured(self):
        local = Config(home=self.home)
        self.assertEqual(local.active_backend, "local")
        self.assertIsInstance(get_model(local), LocalChatModel)
        hosted = Config(home=self.home, hosted_url=self.url)
        self.assertEqual(hosted.active_backend, "hosted")
        model = get_model(hosted)
        self.assertIsInstance(model, HostedChatModel)
        self.assertEqual(model.model_name, "ara-ng")
        self.assertEqual(backend_name(model), "hosted")
        self.assertIsNone(get_model(Config(home=self.home, model_backend="retrieval")))

    def test_hosted_requires_https_unless_loopback(self):
        with self.assertRaises(ModelError):
            HostedChatModel(Config(home=self.home, hosted_url="http://api.example.com/v1"))
        self.assertIsInstance(HostedChatModel(Config(home=self.home, hosted_url="https://api.example.com/v1")), HostedChatModel)
        with self.assertRaises(ModelError):
            get_model(Config(home=self.home, model_backend="hosted"))

    def test_hosted_stream_sends_key_and_model_name(self):
        HostedHandler.calls = []
        with patch.dict(os.environ, {"ARA_HOSTED_API_KEY": "secret-key"}, clear=False):
            config = Config(home=self.home, hosted_url=self.url)
            model = HostedChatModel(config)
            self.assertTrue(model.status()[0])
            text = "".join(model.stream([{"role": "user", "content": "How far?"}]))
        self.assertEqual(text, "Kedu from ara-ng")
        call = HostedHandler.calls[-1]
        self.assertEqual(call["authorization"], "Bearer secret-key")
        self.assertEqual(call["payload"]["model"], "ara-ng")

    def test_engine_labels_retrieval_and_keeps_history(self):
        store = Store(self.home / "ara.sqlite3")
        engine = ConversationEngine(Config(home=self.home), store, FakeFailing())
        answer, _ = engine.respond("How far?")
        self.assertTrue(answer.startswith("[retrieval-only]"))
        self.assertEqual(engine.last_backend, "retrieval-only")
        self.assertIn("stub backend is down", answer)
        messages = store.recent_messages(engine.conversation_id, 5)
        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])

    def test_empty_model_stream_is_labelled_retrieval(self):
        store = Store(self.home / "ara.sqlite3")
        engine = ConversationEngine(Config(home=self.home), store, FakeEmptyStream())
        text = "".join(engine.stream_response("How far?"))
        self.assertTrue(text.startswith("[retrieval-only]"))
        self.assertEqual(engine.last_backend, "retrieval-only")
        self.assertEqual(engine.last_backend_error, "The model returned no text.")

    def test_retrieval_only_without_any_model(self):
        store = Store(self.home / "ara.sqlite3")
        engine = ConversationEngine(Config(home=self.home, model_backend="retrieval"), store, None)
        answer, _ = engine.respond("Explain no wahala")
        self.assertTrue(answer.startswith("[retrieval-only]"))
        self.assertIn("no wahala", answer.lower())

    def test_cli_ask_uses_hosted_ara_ng(self):
        env = dict(
            os.environ,
            ARA_HOME=str(self.home),
            ARA_HOSTED_URL=self.url,
            ARA_HOSTED_API_KEY="secret-key",
            PYTHONPATH=str(ROOT),
        )
        result = subprocess.run(
            [sys.executable, "-m", "ara", "ask", "How far?", "--json"],
            cwd=ROOT, env=env, text=True, capture_output=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["backend"], "hosted")
        self.assertEqual(payload["answer"], "Kedu from ara-ng")

    def test_cli_ask_without_backend_is_labelled(self):
        env = dict(os.environ, ARA_HOME=str(self.home), ARA_MODEL_URL="http://127.0.0.1:19997/v1", PYTHONPATH=str(ROOT))
        env.pop("ARA_HOSTED_URL", None)
        result = subprocess.run(
            [sys.executable, "-m", "ara", "ask", "How far?", "--json"],
            cwd=ROOT, env=env, text=True, capture_output=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["backend"], "retrieval-only")
        self.assertTrue(payload["answer"].startswith("[retrieval-only]"))


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
        self.thread = start(self.server)
        self.url = f"http://127.0.0.1:{self.server.server_port}/model.gguf"
        RangeHandler.requests = []

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def test_probe_reports_size_and_checksum(self):
        info = probe(self.url)
        self.assertEqual(info["size"], len(PAYLOAD))
        self.assertEqual(info["sha256"], PAYLOAD_SHA)

    def test_download_verifies_and_saves(self):
        target = self.root / "model.gguf"
        result = download(self.url, target)
        self.assertEqual(target.read_bytes(), PAYLOAD)
        self.assertTrue(result["checksum_verified"])
        self.assertFalse((self.root / "model.gguf.part").exists())
        self.assertFalse((self.root / "model.gguf.lock").exists())

    def test_resume_uses_partial_file(self):
        target = self.root / "model.gguf"
        part = self.root / "model.gguf.part"
        part.write_bytes(PAYLOAD[:1234])
        result = download(self.url, target)
        self.assertEqual(target.read_bytes(), PAYLOAD)
        self.assertEqual(result["bytes"], len(PAYLOAD))
        self.assertIn(f"bytes=1234-", RangeHandler.requests)

    def test_uppercase_expected_checksum_is_accepted(self):
        target = self.root / "model.gguf"
        result = download(self.url, target, expected_sha256=PAYLOAD_SHA.upper())
        self.assertTrue(result["checksum_verified"])
        self.assertEqual(target.read_bytes(), PAYLOAD)

    def test_bad_checksum_deletes_partial(self):
        target = self.root / "model.gguf"
        with self.assertRaises(DownloadError):
            download(self.url, target, expected_sha256="0" * 64)
        self.assertFalse(target.exists())
        self.assertFalse((self.root / "model.gguf.part").exists())

    def test_concurrent_download_is_refused(self):
        target = self.root / "model.gguf"
        (self.root / "model.gguf.lock").write_text("1 0\n")
        with self.assertRaises(DownloadError) as ctx:
            download(self.url, target)
        self.assertIn("already", str(ctx.exception))

    def test_cli_download_refuses_without_confirmation(self):
        env = dict(os.environ, ARA_HOME=str(self.root / "home"), PYTHONPATH=str(ROOT))
        result = subprocess.run(
            [sys.executable, "-m", "ara", "model", "download", self.url, "--out", str(self.root / "cli.gguf")],
            cwd=ROOT, env=env, text=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=60,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--yes", result.stderr)
        self.assertFalse((self.root / "cli.gguf").exists())

    def test_cli_download_with_yes_saves_verified_file(self):
        env = dict(os.environ, ARA_HOME=str(self.root / "home"), PYTHONPATH=str(ROOT))
        target = self.root / "cli.gguf"
        result = subprocess.run(
            [sys.executable, "-m", "ara", "model", "download", self.url, "--out", str(target), "--yes"],
            cwd=ROOT, env=env, text=True, capture_output=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(target.read_bytes(), PAYLOAD)
        self.assertIn("SHA-256", result.stdout)
        self.assertIn(human_size(len(PAYLOAD)), result.stdout)


class CatalogTests(unittest.TestCase):
    def test_catalog_entries_are_https_and_sized(self):
        names = set()
        for name, entry in MODELS.items():
            self.assertTrue(str(entry["url"]).startswith("https://"), name)
            self.assertIsInstance(entry["bytes"], int)
            self.assertGreater(entry["bytes"], 100_000_000)
            self.assertTrue(str(entry["ram"]))
            self.assertTrue(str(entry["license"]))
            self.assertNotIn(name, names)
            names.add(name)


class ConfigBackendTests(unittest.TestCase):
    def test_env_selects_backend_and_hosted_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "ARA_BACKEND": "retrieval",
                "ARA_HOSTED_URL": "https://ara.example/v1",
                "ARA_HOSTED_MODEL": "ara-ng",
            }
            with patch.dict(os.environ, env, clear=False):
                config = load_config(tmp)
            self.assertEqual(config.model_backend, "retrieval")
            self.assertEqual(config.active_backend, "retrieval-only")
            self.assertEqual(config.hosted_url, "https://ara.example/v1")

    def test_default_config_mentions_hosted_without_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            config = load_config(home)
            from ara.config import write_default_config

            path = write_default_config(config)
            text = path.read_text()
            self.assertIn("hosted_url", text)
            self.assertIn("ARA_HOSTED_API_KEY", text)
            self.assertNotIn("secret", text.casefold().replace("secret-key", ""))


if __name__ == "__main__":
    unittest.main()
