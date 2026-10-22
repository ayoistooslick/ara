from __future__ import annotations

import io
import os
import re
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ara.brand import FRAMES, animation_frames, animation_supported, hop, static_frog


class FakeTTY:
    def __init__(self):
        self.value = ""
        self.width = 100

    def isatty(self):
        return True

    def write(self, value):
        self.value += value

    def flush(self):
        pass


class BrandTests(unittest.TestCase):
    def test_static_frog_has_identity(self):
        frog = static_frog()
        self.assertIn("o", frog)
        self.assertIn("_____", frog)
        self.assertNotIn("\x1b[", frog)
        self.assertEqual(len(frog.splitlines()), 8)

    def test_animation_frames_are_distinct_and_well_formed(self):
        frames = animation_frames()
        self.assertEqual(len(frames), 6)
        self.assertEqual(len(set(frames)), 6)
        for frame in frames:
            self.assertEqual(len(frame.splitlines()), 8)
            self.assertLessEqual(max(len(line) for line in frame.splitlines()), 26)

    def test_non_interactive_output_falls_back_to_static(self):
        stream = io.StringIO()
        self.assertFalse(hop(stream=stream, loops=2, delay=0))
        self.assertEqual(stream.getvalue().count(".-\"\"\"-."), 1)
        self.assertNotIn("\x1b[?25l", stream.getvalue())
        self.assertFalse(animation_supported(stream))

    def test_interactive_animation_really_moves_and_restores_cursor(self):
        stream = FakeTTY()
        with patch.dict(os.environ, {"TERM": "xterm-256color"}, clear=False):
            os.environ.pop("NO_COLOR", None)
            os.environ.pop("ARA_NO_ANIMATION", None)
            with patch("ara.brand.shutil.get_terminal_size", return_value=os.terminal_size((100, 40))):
                animated = hop(stream=stream, loops=1, delay=0)
        self.assertTrue(animated)
        output = stream.value
        self.assertIn("\x1b[?25l", output)
        self.assertIn("\x1b[?25h", output)
        # Multiple distinct frames, not one printed picture.
        self.assertGreaterEqual(output.count("o   o"), 2)
        self.assertGreaterEqual(output.count("\x1b[8F"), 2)

    def test_reduced_or_disabled_animation_is_respected(self):
        stream = FakeTTY()
        with patch.dict(os.environ, {"TERM": "xterm", "ARA_NO_ANIMATION": "1"}):
            self.assertFalse(animation_supported(stream))
            self.assertFalse(hop(stream=stream, loops=2, delay=0))
        self.assertNotIn("\x1b[?25l", stream.value)

    def test_svg_mascot_is_original_and_nigerian(self):
        svg_path = ROOT / "ara" / "web" / "static" / "frog.svg"
        text = svg_path.read_text()
        root = ET.fromstring(text)
        self.assertTrue(root.tag.endswith("svg"))
        self.assertIn("#008751", text)
        self.assertIn("#f7f5ea", text)
        lower = text.casefold()
        for forbidden in ("astronaut", "robot", "circuit", "orb", "gradient-orb"):
            self.assertNotIn(forbidden, lower)
        self.assertIn("Ara the frog", text)
        self.assertIn("viewBox", text)

    def test_web_has_hop_animation_and_reduced_motion(self):
        css = (ROOT / "ara" / "web" / "static" / "styles.css").read_text()
        self.assertIn("@keyframes frog-hop", css)
        self.assertIn("prefers-reduced-motion", css)
        self.assertIn("mascot-hop", css)

    def test_web_shell_is_accessible_and_csp_friendly(self):
        html = (ROOT / "ara" / "web" / "static" / "index.html").read_text()
        self.assertIn('<html lang="en">', html)
        self.assertIn('name="viewport"', html)
        self.assertIn("skip-link", html)
        self.assertIn('aria-live="polite"', html)
        self.assertGreaterEqual(html.count("<label"), 6)
        self.assertIn("frog.svg", html)
        self.assertNotRegex(html, r"<script(?![^>]*src=)")
        self.assertNotRegex(html, r"\son(click|change|submit|load)=")
        self.assertNotRegex(html, r"style=\"")

    def test_client_uses_permission_gated_sensors_and_real_cleanup(self):
        js = (ROOT / "ara" / "web" / "static" / "app.js").read_text()
        self.assertIn("getUserMedia", js)
        self.assertIn("MediaRecorder", js)
        self.assertIn("speechSynthesis", js)
        self.assertIn("getTracks().forEach((track) => track.stop())", js)
        self.assertIn("cameraLive", js)
        self.assertIn("visibilitychange", js)
        self.assertIn("prefers-reduced-motion", (ROOT / "ara" / "web" / "static" / "styles.css").read_text())
        # Model and user text must be inserted as text, not parsed as HTML.
        self.assertNotIn("innerHTML", js)
        self.assertNotIn("document.write", js)


if __name__ == "__main__":
    unittest.main()
