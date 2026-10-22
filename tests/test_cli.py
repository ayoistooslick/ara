"""End-to-end checks that drive the real CLI against a stub local model server."""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ara.cli import main
from ara.store import Store


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "ara-test-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length))
        self.server.requests.append(request)
        body = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": "I hear you. Let us work through it."}}]
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class CLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.server.requests = []
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.env = mock.patch.dict(os.environ, {
            "ARA_HOME": str(self.home),
            "ARA_MODEL_URL": self.url,
            "NO_COLOR": "1",
        })
        self.env.start()
        self.server.requests.clear()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = main(list(argv))
            except SystemExit as exc:  # argparse --help
                code = int(exc.code or 0)
        return code, out.getvalue(), err.getvalue()

    def test_ask_persists_the_conversation(self):
        code, out, _ = self.run_cli("ask", "How do I start a small business?")
        self.assertEqual(code, 0)
        self.assertIn("Let us work through it", out)
        self.assertTrue(self.server.requests)
        prompt = json.dumps(self.server.requests[0], ensure_ascii=False)
        self.assertIn("Nigerian", prompt)
        code, history, _ = self.run_cli("history")
        self.assertEqual(code, 0)
        self.assertIn("How do I start a small business?", history)
        self.assertIn("I hear you", history)

    def test_status_reports_schema_and_live_model(self):
        code, out, _ = self.run_cli("status")
        self.assertEqual(code, 0)
        self.assertIn("Schema: 2", out)
        self.assertIn("Connected to local inference server", out)
        self.assertIn("ara-test-model", out)

    def test_nigeria_explain_and_eval(self):
        code, out, _ = self.run_cli("nigeria", "explain", "No wahala", "--json")
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["register"], "pidgin")
        self.assertTrue(payload["detected"])
        code, out, _ = self.run_cli("nigeria", "eval", "--json")
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertGreater(report["retrieval"]["cases"], 0)
        self.assertGreaterEqual(report["retrieval"]["accuracy"], 0.8)

    def test_news_offline_returns_structured_empty_cache(self):
        code, out, _ = self.run_cli("news", "--offline", "--json")
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertEqual(report["items"], [])
        self.assertTrue(report["offline"])

    def test_memory_embed_and_forget_all(self):
        self.run_cli("memory", "add", "User is learning Yoruba")
        code, out, _ = self.run_cli("memory", "embed")
        self.assertEqual(code, 0)
        self.assertIn("hashed-lexical", out.lower())
        self.assertIn('"semantic": false', out.lower())
        code, out, _ = self.run_cli("memory", "forget-all")
        self.assertEqual(code, 2)
        code, out, _ = self.run_cli("memory", "forget-all", "--yes")
        self.assertEqual(code, 0)
        self.assertIn("Forgot 1", out)

    def test_plain_animation_never_writes_control_sequences(self):
        code, out, _ = self.run_cli("animate", "--plain")
        self.assertEqual(code, 0)
        self.assertNotIn("\x1b", out)
        self.assertIn('.-\"\"\"-.', out)
        self.assertIn("~~~~~~~~~~~", out)

    def test_vision_without_endpoint_fails_clearly(self):
        image = self.home / "sample.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40)
        code, _, err = self.run_cli("vision", str(image))
        self.assertEqual(code, 2)
        self.assertIn("vision", err.lower())

    def test_restore_backs_up_current_data_first(self):
        self.run_cli("memory", "add", "Memory from the snapshot")
        snapshot = self.home / "snapshot.sqlite3"
        Store(self.home / "ara.sqlite3").backup_to(snapshot)
        self.run_cli("memory", "add", "Memory written after the snapshot")

        class TTY(io.StringIO):
            def isatty(self):
                return True

        with mock.patch("ara.cli.sys.stdin", new=TTY()), mock.patch("builtins.input", return_value="restore"):
            code, out, err = self.run_cli("restore", str(snapshot))
        self.assertEqual(code, 0, err)
        self.assertIn("Saved the current database", out)
        backups = list((self.home / "backups").glob("ara-pre-restore-*.sqlite3"))
        self.assertEqual(len(backups), 1)
        self.assertTrue(Store.check_backup(backups[0]))
        memories = Store(backups[0]).list_memories()
        self.assertIn("Memory written after the snapshot", [item["content"] for item in memories])
        code, out, _ = self.run_cli("memory", "inspect")
        self.assertEqual(code, 0)
        self.assertIn("Memory from the snapshot", out)
        self.assertNotIn("Memory written after the snapshot", out)

    def test_help_lists_web_and_news(self):
        code, out, _ = self.run_cli("--help")
        self.assertEqual(code, 0)
        self.assertIn("web", out)
        self.assertIn("news", out)
        self.assertIn("nigeria", out)


if __name__ == "__main__":
    unittest.main()
