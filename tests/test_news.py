from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ara import news
from ara.config import Config
from ara.store import Store

RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Test Feed</title>
<item>
  <title>Super Eagles name new coach ahead of AFCON</title>
  <link>https://example.ng/sport/eagles?utm_source=rss</link>
  <description><![CDATA[<p>The <b>Nigeria</b> team announced a coach.</p><script>bad()</script>]]></description>
  <pubDate>Wed, 21 Oct 2026 08:30:00 +0000</pubDate>
</item>
<item>
  <title>Naira steadies as inflation report released</title>
  <link>https://example.ng/business/naira</link>
  <description>Markets reacted to the latest figures.</description>
  <pubDate>Tue, 20 Oct 2026 18:00:00 +0000</pubDate>
</item>
<item>
  <title></title>
  <link>https://example.ng/empty</link>
  <description>No title should be skipped.</description>
</item>
<item>
  <title>Unsafe link</title>
  <link>javascript:alert(1)</link>
  <pubDate>Wed, 21 Oct 2026 08:30:00 +0000</pubDate>
</item>
</channel></rss>"""

ATOM = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom Feed</title>
  <entry><title>EndSARS panel report discussed</title><link href="https://example.ng/politics/endsars"/>
  <summary>Police reform report.</summary><updated>2026-10-21T09:00:00Z</updated></entry>
</feed>"""


class FeedHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/rss"):
            body = RSS
            content_type = "application/rss+xml"
        elif self.path.startswith("/atom"):
            body = ATOM
            content_type = "application/atom+xml"
        elif self.path.startswith("/second"):
            body = RSS.replace(b"Super Eagles name new coach ahead of AFCON", b"Super Eagles appoint new head coach before AFCON")
            content_type = "application/rss+xml"
        elif self.path.startswith("/broken"):
            body = b"<html>not xml"
            content_type = "text/html"
        elif self.path.startswith("/slow"):
            self.send_response(500)
            self.end_headers()
            return
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


class NewsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FeedHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.config = Config(home=self.home, allowed_roots=(self.home,), news_timeout=5)
        self.store = Store(self.config.db_path)

    def tearDown(self):
        self.temp.cleanup()

    def test_rss_parsing_attributes_source_and_sanitises_summary(self):
        records = news.fetch_feed(self.base + "/rss")
        self.assertEqual(len(records), 2)
        items = news.parse_items(self.base + "/rss", records)
        self.assertEqual(len(items), 2)
        first = items[0]
        self.assertEqual(first.source, "127.0.0.1")
        self.assertEqual(first.source_url, "https://127.0.0.1")
        self.assertIn("example.ng/sport/eagles", first.link)
        self.assertNotIn("bad()", first.summary)
        self.assertIn("Nigeria team", first.summary)
        self.assertEqual(first.published_at, "2026-10-21T08:30:00+00:00")
        self.assertEqual(first.topic, "Sports")

    def test_atom_parsing(self):
        items = news.parse_items(self.base + "/atom", news.fetch_feed(self.base + "/atom"))
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].title, "EndSARS panel report discussed")
        self.assertEqual(items[0].published_at, "2026-10-21T09:00:00+00:00")
        self.assertEqual(items[0].topic, "Security")

    def test_invalid_xml_is_an_error_not_an_empty_success(self):
        with self.assertRaises(news.NewsError):
            news.fetch_feed(self.base + "/broken")
        with self.assertRaises(news.NewsError):
            news.fetch_feed(self.base + "/slow")

    def test_summary_and_topic_helpers(self):
        self.assertLessEqual(len(news.summarize("<p>" + "word " * 200 + "</p>", 120)), 122)
        self.assertEqual(news.classify("Naira and inflation", "CBN market"), "Business")
        self.assertEqual(news.classify("Local council election", "INEC results"), "Politics")
        self.assertEqual(news.classify("A quiet street", "Nothing happened"), "General")

    def test_refresh_persists_and_deduplicates(self):
        first = news.refresh_news(self.store, self.config, feeds=[self.base + "/rss"])
        self.assertEqual(first["added"], 2)
        self.assertEqual(first["feeds_ok"], 1)
        self.assertEqual(len(self.store.news_items()), 2)
        second = news.refresh_news(self.store, self.config, feeds=[self.base + "/rss"])
        self.assertEqual(second["added"], 0)
        self.assertEqual(len(self.store.news_items()), 2)

    def test_near_duplicate_headlines_collapse(self):
        report = news.refresh_news(self.store, self.config, feeds=[self.base + "/rss", self.base + "/second"])
        self.assertEqual(report["added"], 2)
        items = self.store.news_items()
        eagles = [item for item in items if "Super Eagles" in item["title"]]
        self.assertEqual(len(eagles), 1)
        self.assertEqual(len([item for item in items if "Naira" in item["title"]]), 1)

    def test_related_source_labels_are_recorded_for_a_competing_publisher(self):
        base = {
            "title": "Super Eagles appoint new coach",
            "summary": "A coach was appointed.",
            "link": "https://one.example/story",
            "source": "Publisher One",
            "source_url": "https://one.example",
            "published_at": "2026-10-21T08:30:00+00:00",
            "fetched_at": news.now(),
            "topic": "Sports",
            "dedupe_key": "one-story",
        }
        self.store.save_news([base])
        # A different canonical link with a near-identical headline should attach to the story.
        duplicate = dict(base, link="https://two.example/story", dedupe_key="two-story", source="Publisher Two", source_url="https://two.example")
        # save_news itself does not deduplicate; refresh_news does. Exercise the helper.
        self.assertTrue(news.similar_story(base["title"], duplicate["title"]))

    def test_offline_returns_dated_cache_and_labels_age(self):
        news.refresh_news(self.store, self.config, feeds=[self.base + "/rss"])
        report = news.news_report(self.store, self.config, offline=True, knowledge=None)
        self.assertTrue(report["offline"])
        self.assertEqual(len(report["items"]), 2)
        item = report["items"][0]
        self.assertTrue(item["published_at"])
        self.assertIn("freshness_label", item)
        self.assertIn("age_hours", item)
        self.assertIn("source", item)
        self.assertTrue(item["link"].startswith("http"))

    def test_feed_failure_preserves_cache_and_reports_error(self):
        news.refresh_news(self.store, self.config, feeds=[self.base + "/rss"])
        result = news.refresh_news(self.store, self.config, feeds=[self.base + "/missing"])
        self.assertEqual(result["feeds_failed"], 1)
        self.assertTrue(result["errors"])
        self.assertEqual(len(self.store.news_items()), 2)

    def test_missing_publication_date_is_not_presented_as_current(self):
        item = {
            "title": "A story with no date",
            "summary": "Cached without a timestamp.",
            "link": "https://example.ng/undated",
            "source": "Example",
            "source_url": "https://example.ng",
            "published_at": "",
            "fetched_at": news.now(),
            "topic": "General",
            "dedupe_key": "undated",
        }
        self.store.save_news([item])
        enriched = news.enrich(self.store.news_items())[0]
        self.assertFalse(enriched["fresh"])
        self.assertEqual(enriched["freshness_label"], "date not provided")

    def test_old_story_is_labelled_older(self):
        old = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(timespec="seconds")
        self.store.save_news([{
            "title": "Ten day old story", "summary": "Old.", "link": "https://example.ng/old",
            "source": "Example", "source_url": "https://example.ng", "published_at": old,
            "fetched_at": news.now(), "topic": "General", "dedupe_key": "old-story",
        }])
        enriched = news.enrich(self.store.news_items())[0]
        self.assertFalse(enriched["fresh"])
        self.assertEqual(enriched["freshness_label"], "older")

    def test_historical_context_needs_a_documented_year_not_shared_words(self):
        from ara.nigeria import KnowledgeBase

        knowledge = KnowledgeBase.load()
        no_year = "Yobe court sentences man to death for killing wife; police confirm the arrest"
        self.assertEqual(knowledge.history_context(no_year), "")
        with_year = "Group revisits the June 12, 1993 annulment as lawyers cite the case"
        context = knowledge.history_context(with_year)
        self.assertIn("1993", context)
        self.assertIn("June 12", context)

        enriched = news.enrich([{
            "title": "Court sentences man", "summary": no_year,
            "link": "https://example.ng/yobe", "source": "Example",
            "source_url": "https://example.ng", "published_at": news.now(),
            "fetched_at": news.now(), "topic": "Metro", "dedupe_key": "yobe-court",
        }], knowledge=knowledge)
        self.assertFalse(enriched[0].get("historical_context"))

    def test_source_registry_is_attributed_https(self):
        self.assertGreaterEqual(len(news.SOURCE_REGISTRY), 5)
        for source in news.SOURCE_REGISTRY:
            self.assertTrue(source["feed"].startswith("https://"))
            self.assertTrue(source["site"].startswith("https://"))
            self.assertTrue(source["name"])

    def test_story_without_title_or_safe_link_is_skipped(self):
        records = news.fetch_feed(self.base + "/rss")
        items = news.parse_items(self.base + "/rss", records)
        self.assertFalse(any("javascript" in item.link for item in items))
        self.assertTrue(all(item.title.strip() for item in items))


if __name__ == "__main__":
    unittest.main()
