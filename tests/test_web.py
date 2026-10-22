from __future__ import annotations

import base64
import http.cookiejar
import json
import os
import struct
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ara.config import Config
from ara.web.server import AraServer, create_server

PNG = (
    b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 40, 30)
    + b"\x08\x06\x00\x00\x00" + b"\x00\x00\x00\x00"
)

RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>Stub</title>
<item><title>Super Eagles train in Uyo</title><link>https://example.ng/eagles</link>
<description>Preparations continue.</description><pubDate>Wed, 21 Oct 2026 08:00:00 +0000</pubDate></item>
</channel></rss>"""


class ModelHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"data": [{"id": "stub-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for piece in ("Kedu", "! ", "How far?"):
                self.wfile.write(("data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        body = json.dumps({"choices": [{"message": {"content": "Kedu! I dey."}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


class VisionHandler(BaseHTTPRequestHandler):
    calls: list[dict] = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        VisionHandler.calls.append(payload)
        body = json.dumps({"choices": [{"message": {"content": "A green frog."}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


class AudioHandler(BaseHTTPRequestHandler):
    calls: list[bytes] = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        AudioHandler.calls.append(self.rfile.read(length))
        body = json.dumps({"text": "How far?"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


class FeedHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/rss+xml")
        self.send_header("Content-Length", str(len(RSS)))
        self.end_headers()
        self.wfile.write(RSS)

    def log_message(self, *_):
        pass


def start(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


class Harness:
    def __init__(self, **overrides):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.model = ThreadingHTTPServer(("127.0.0.1", 0), ModelHandler)
        self.vision = ThreadingHTTPServer(("127.0.0.1", 0), VisionHandler)
        self.audio = ThreadingHTTPServer(("127.0.0.1", 0), AudioHandler)
        self.feed = ThreadingHTTPServer(("127.0.0.1", 0), FeedHandler)
        self.threads = [start(self.model), start(self.vision), start(self.audio), start(self.feed)]
        dead = overrides.pop("dead_model", False)
        vision_enabled = overrides.pop("vision_enabled", False)
        audio_enabled = overrides.pop("audio_enabled", False)
        news_feed = overrides.pop("news_feed", False)
        model_url = f"http://127.0.0.1:19997/v1" if dead else f"http://127.0.0.1:{self.model.server_port}/v1"
        self.config = Config(
            home=self.home,
            allowed_roots=(self.home,),
            model_url=model_url,
            model_name="stub-model",
            vision_url=f"http://127.0.0.1:{self.vision.server_port}/v1" if vision_enabled else None,
            vision_model="stub-vision",
            audio_url=f"http://127.0.0.1:{self.audio.server_port}/v1" if audio_enabled else None,
            audio_model="stub-whisper",
            news_feeds=(f"http://127.0.0.1:{self.feed.server_port}/rss",) if news_feed else (),
            **overrides,
        )
        self.server = AraServer(("127.0.0.1", 0), self.config)
        self.thread = start(self.server)
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))
        self.csrf = None
        self.conversation_id = None

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        for server, thread in ((self.model, self.threads[0]), (self.vision, self.threads[1]), (self.audio, self.threads[2]), (self.feed, self.threads[3])):
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def request(self, path, method="GET", data=None, headers=None, csrf=True, opener=None, timeout=20):
        opener = opener or self.opener
        body = json.dumps(data).encode() if data is not None else None
        all_headers = {}
        if data is not None:
            all_headers["Content-Type"] = "application/json"
        if csrf and self.csrf and method in {"POST", "DELETE"}:
            all_headers["X-Ara-CSRF"] = self.csrf
        all_headers.update(headers or {})
        request = urllib.request.Request(self.base + path, data=body, headers=all_headers, method=method)
        try:
            with opener.open(request, timeout=timeout) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as error:
            return error.code, dict(error.headers), error.read()

    def session(self, password=None, forwarded_proto=None, opener=None):
        opener = opener or self.opener
        headers = {"X-Forwarded-Proto": forwarded_proto} if forwarded_proto else {}
        status, response_headers, body = self.request(
            "/api/session", "POST", {} if password is None else {"password": password},
            headers=headers, csrf=False, opener=opener,
        )
        if status == 200:
            data = json.loads(body)
            self.csrf = data["csrf"]
            self.conversation_id = data["conversation_id"]
        return status, response_headers, body


def events(raw: bytes) -> list[dict]:
    result = []
    for block in raw.decode("utf-8").split("\n\n"):
        data = "\n".join(line[5:].strip() for line in block.splitlines() if line.startswith("data:"))
        if data:
            try:
                result.append(json.loads(data))
            except json.JSONDecodeError:
                pass
    return result


class WebTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness()

    def tearDown(self):
        self.h.close()

    def test_health_static_and_security_headers(self):
        status, headers, body = self.h.request("/api/health", csrf=False)
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["ok"])
        self.assertEqual(headers.get("X-Frame-Options"), "DENY")
        self.assertIn("default-src 'self'", headers.get("Content-Security-Policy", ""))
        self.assertNotIn("Access-Control-Allow-Origin", headers)
        status, _, page = self.h.request("/", csrf=False)
        self.assertEqual(status, 200)
        self.assertIn(b"The AI that", page)
        status, headers, script = self.h.request("/app.js", csrf=False)
        self.assertEqual(status, 200)
        self.assertIn("javascript", headers.get("Content-Type", ""))
        self.assertIn(b"getUserMedia", script)
        status, headers, svg = self.h.request("/frog.svg", csrf=False)
        self.assertEqual(headers.get("Content-Type"), "image/svg+xml")
        status, headers, body = self.h.request("/", method="HEAD", csrf=False)
        self.assertEqual(status, 200)
        self.assertEqual(body, b"")

    def test_state_requires_session_then_returns_shared_core(self):
        status, _, _ = self.h.request("/api/state", csrf=False)
        self.assertEqual(status, 401)
        status, headers, body = self.h.session()
        self.assertEqual(status, 200)
        cookie = headers.get("Set-Cookie", "")
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Strict", cookie)
        data = json.loads(body)
        self.assertTrue(data["csrf"])
        status, _, body = self.h.request("/api/state")
        self.assertEqual(status, 200)
        state = json.loads(body)
        self.assertEqual(state["model"]["configured_model"], "stub-model")
        self.assertIn("knowledge", state)
        self.assertIn("perception", state)
        self.assertIn("retrieval", state["memory"])

    def test_csrf_and_origin_are_enforced(self):
        self.h.session()
        status, _, _ = self.h.request("/api/memory", "POST", {"content": "test", "kind": "fact"}, csrf=False)
        self.assertEqual(status, 403)
        status, _, _ = self.h.request("/api/memory", "POST", {"content": "test", "kind": "fact"}, headers={"X-Ara-CSRF": "wrong"})
        self.assertEqual(status, 403)
        status, _, _ = self.h.request("/api/memory", "POST", {"content": "test", "kind": "fact"}, headers={"Origin": "https://evil.example"})
        self.assertEqual(status, 403)
        status, _, _ = self.h.request("/api/memory", "POST", {"content": "test", "kind": "fact"}, headers={"Origin": self.h.base})
        self.assertEqual(status, 201)

    def test_chat_streams_and_persists_per_session(self):
        self.h.session()
        status, headers, body = self.h.request("/api/chat", "POST", {"message": "How far?", "mode": "chat"}, timeout=30)
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", headers.get("Content-Type", ""))
        parsed = events(body)
        deltas = "".join(event.get("text", "") for event in parsed if event["type"] == "delta")
        self.assertEqual(deltas, "Kedu! How far?")
        done = next(event for event in parsed if event["type"] == "done")
        self.assertEqual(done["text"], "Kedu! How far?")
        self.assertEqual(done["interpretation"]["detected"][0]["id"], "how-far")
        status, _, body = self.h.request("/api/conversation")
        messages = json.loads(body)["messages"]
        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
        self.assertEqual(messages[0]["content"], "How far?")

        second = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        status, _, body = self.h.session(opener=second)
        self.assertEqual(status, 200)
        second_csrf = json.loads(body)["csrf"]
        status, _, body = self.h.request("/api/conversation", opener=second)
        self.assertEqual(json.loads(body)["messages"], [])
        # Chat in the second session must not appear in the first.
        status, _, _ = self.h.request(
            "/api/chat", "POST", {"message": "Second session", "mode": "chat"},
            headers={"X-Ara-CSRF": second_csrf}, csrf=False, opener=second, timeout=30,
        )
        self.assertEqual(status, 200)
        first_messages = json.loads(self.h.request("/api/conversation")[2])["messages"]
        self.assertNotIn("Second session", [message["content"] for message in first_messages])

    def test_model_unreachable_returns_explicit_error(self):
        harness = Harness(dead_model=True)
        try:
            harness.session()
            status, _, body = harness.request("/api/chat", "POST", {"message": "hello"}, timeout=10)
            self.assertEqual(status, 503)
            self.assertIn("model", json.loads(body)["error"].casefold())
        finally:
            harness.close()

    def test_password_protection_and_secure_cookie(self):
        with patch.dict(os.environ, {"ARA_WEB_PASSWORD": "s3cret-pass"}, clear=False):
            harness = Harness()
            try:
                status, _, _ = harness.session()
                self.assertEqual(status, 401)
                status, _, _ = harness.session(password="wrong")
                self.assertEqual(status, 401)
                status, headers, _ = harness.session(password="s3cret-pass", forwarded_proto="https")
                self.assertEqual(status, 200)
                self.assertIn("Secure", headers.get("Set-Cookie", ""))
                # A Secure cookie is deliberately not sent over plain HTTP by a browser
                # cookie jar; open an HTTP session to verify authenticated state.
                status, _, _ = harness.session(password="s3cret-pass")
                self.assertEqual(status, 200)
                self.assertEqual(harness.request("/api/state")[0], 200)
            finally:
                harness.close()

    def test_non_loopback_requires_a_password(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ARA_WEB_PASSWORD", None)
            with tempfile.TemporaryDirectory() as tmp:
                config = Config(home=Path(tmp), allowed_roots=(Path(tmp),))
                with self.assertRaises(ValueError):
                    create_server(config, host="0.0.0.0", port=0)

    def test_memory_controls_and_export(self):
        self.h.session()
        status, _, body = self.h.request("/api/memory", "POST", {"content": "User prefers plain English", "kind": "preference"})
        self.assertEqual(status, 201)
        memory = json.loads(body)["memory"]
        self.assertEqual(memory["source"], "web-user")
        status, _, body = self.h.request("/api/memory")
        self.assertEqual([item["content"] for item in json.loads(body)["memories"]], ["User prefers plain English"])
        status, _, _ = self.h.request(f"/api/memory/{memory['id']}/correct", "POST", {"content": "User prefers very plain English"})
        self.assertEqual(status, 200)
        status, headers, body = self.h.request("/api/memory/export")
        self.assertEqual(status, 200)
        self.assertIn("attachment", headers.get("Content-Disposition", ""))
        self.assertIn("very plain English", body.decode())
        status, _, _ = self.h.request("/api/memory/forget-all", "POST", {"confirm": "no"})
        self.assertEqual(status, 400)
        status, _, _ = self.h.request("/api/memory/forget-all", "POST", {"confirm": "forget all"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(self.h.request("/api/memory")[2])["memories"], [])
        status, _, _ = self.h.request("/api/memory", "POST", {"content": "my password is hunter2", "kind": "fact"})
        self.assertEqual(status, 400)

    def test_memory_is_shared_but_conversations_are_isolated(self):
        self.h.session()
        self.h.request("/api/memory", "POST", {"content": "Shared note", "kind": "fact"})
        second = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        self.h.session(opener=second)
        status, _, body = self.h.request("/api/memory", opener=second)
        self.assertEqual([item["content"] for item in json.loads(body)["memories"]], ["Shared note"])

    def test_news_offline_and_nigeria_context(self):
        self.h.session()
        status, _, body = self.h.request("/api/news?offline=1")
        self.assertEqual(status, 200)
        report = json.loads(body)
        self.assertTrue(report["offline"])
        self.assertEqual(report["items"], [])
        self.assertTrue(report["sources"])
        status, _, body = self.h.request("/api/nigeria?q=Na%20wa%20o")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertTrue(payload["clarify"])
        self.assertEqual(payload["detected"][0]["id"], "na-wa")
        self.assertEqual(self.h.request("/api/nigeria")[0], 400)

    def test_news_refresh_from_configured_feed(self):
        harness = Harness(news_feed=True)
        try:
            harness.session()
            status, _, body = harness.request("/api/news?refresh=1")
            self.assertEqual(status, 200)
            report = json.loads(body)
            self.assertEqual(report["added"], 1)
            self.assertEqual(report["items"][0]["source"], "127.0.0.1")
            self.assertTrue(report["items"][0]["link"].startswith("https://"))
        finally:
            harness.close()

    def test_vision_and_transcription_unconfigured_are_503(self):
        self.h.session()
        image = base64.b64encode(PNG).decode()
        status, _, body = self.h.request("/api/vision", "POST", {"image": "data:image/png;base64," + image})
        self.assertEqual(status, 503)
        self.assertIn("vision", json.loads(body)["error"].casefold())
        status, _, body = self.h.request("/api/transcribe", "POST", {"audio": "data:audio/webm;base64," + base64.b64encode(b"\x1aE\xdf\xa3" + b"0" * 32).decode()})
        self.assertEqual(status, 503)

    def test_vision_and_transcription_configured_contract(self):
        harness = Harness(vision_enabled=True, audio_enabled=True)
        try:
            harness.session()
            VisionHandler.calls = []
            AudioHandler.calls = []
            image = base64.b64encode(PNG).decode()
            status, _, body = harness.request("/api/vision", "POST", {"image": "data:image/png;base64," + image, "prompt": "What is this?"})
            self.assertEqual(status, 200)
            result = json.loads(body)
            self.assertEqual(result["text"], "A green frog.")
            self.assertEqual(result["image"]["width"], 40)
            self.assertEqual(VisionHandler.calls[-1]["model"], "stub-vision")
            webm = base64.b64encode(b"\x1aE\xdf\xa3" + b"0" * 32).decode()
            status, _, body = harness.request("/api/transcribe", "POST", {"audio": "data:audio/webm;base64," + webm})
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["text"], "How far?")
            self.assertTrue(AudioHandler.calls)
        finally:
            harness.close()

    def test_upload_validation(self):
        harness = Harness(vision_enabled=True)
        try:
            harness.session()
            status, _, _ = harness.request("/api/vision", "POST", {"image": "data:image/svg+xml;base64," + base64.b64encode(b"<svg/>").decode()})
            self.assertEqual(status, 415)
            status, _, _ = harness.request("/api/vision", "POST", {"image": "data:image/png;base64,not-base64"})
            self.assertEqual(status, 400)
            status, _, _ = harness.request("/api/vision", "POST", {"image": "data:image/png;base64," + base64.b64encode(b"not a png").decode()})
            self.assertEqual(status, 400)
        finally:
            harness.close()

    def test_body_and_message_limits(self):
        self.h.session()
        status, _, _ = self.h.request("/api/chat", "POST", {"message": "x" * 9000})
        self.assertEqual(status, 413)
        # Announce an oversized body without sending it: the server must refuse from
        # the header alone rather than buffering it.
        cookie = "; ".join(f"{item.name}={item.value}" for item in self.h.jar)
        import http.client
        from urllib.parse import urlparse
        parsed = urlparse(self.h.base)
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
        connection.putrequest("POST", "/api/vision")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", "13000000")
        connection.putheader("Cookie", cookie)
        connection.putheader("X-Ara-CSRF", self.h.csrf)
        connection.endheaders()
        response = connection.getresponse()
        self.assertEqual(response.status, 413)
        connection.close()

    def test_static_traversal_is_blocked(self):
        status, _, _ = self.h.request("/../pyproject.toml", csrf=False)
        self.assertEqual(status, 404)
        status, _, _ = self.h.request("/static/../../README.md", csrf=False)
        self.assertEqual(status, 404)
        status, _, _ = self.h.request("/%2e%2e/pyproject.toml", csrf=False)
        self.assertEqual(status, 404)

    def test_rate_limit_is_enforced_per_session(self):
        harness = Harness(web_rate_limit=1)
        try:
            harness.session()
            first = harness.request("/api/chat", "POST", {"message": "one"}, timeout=30)
            self.assertEqual(first[0], 200)
            second = harness.request("/api/chat", "POST", {"message": "two"}, timeout=30)
            self.assertEqual(second[0], 429)
        finally:
            harness.close()

    def test_unknown_routes_and_methods(self):
        self.h.session()
        self.assertEqual(self.h.request("/api/nope")[0], 404)
        self.assertEqual(self.h.request("/api/chat")[0], 404)
        self.assertEqual(self.h.request("/api/state", "DELETE", {})[0], 404)

    def test_logout_invalidates_session(self):
        self.h.session()
        status, headers, _ = self.h.request("/api/logout", "POST", {})
        self.assertEqual(status, 200)
        self.assertIn("Max-Age=0", headers.get("Set-Cookie", ""))
        self.assertEqual(self.h.request("/api/state")[0], 401)


if __name__ == "__main__":
    unittest.main()
