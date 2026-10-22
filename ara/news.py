"""Nigerian news built from publisher RSS feeds.

Rules:
- Link and attribute every story to its publisher.
- Keep the publisher's publication timestamp; never present old news as current.
- Do not generate headlines, quotes, or claims. Summary text is trimmed feed text.
- Degrade to the last cached stories when offline, labelled with their dates.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Iterable
from urllib.parse import urlparse

from .config import Config
from .store import Store, now

VERIFIED_ON = "2026-10-22"

# Feeds verified to return XML on the date above. Publishers change endpoints;
# a failing feed is reported, never silently replaced with invented content.
SOURCE_REGISTRY: list[dict[str, str]] = [
    {"name": "Premium Times", "feed": "https://www.premiumtimesng.com/feed", "site": "https://www.premiumtimesng.com"},
    {"name": "Punch", "feed": "https://punchng.com/feed/", "site": "https://punchng.com"},
    {"name": "Vanguard", "feed": "https://www.vanguardngr.com/feed/", "site": "https://www.vanguardngr.com"},
    {"name": "Daily Trust", "feed": "https://dailytrust.com/feed/", "site": "https://dailytrust.com"},
    {"name": "Channels Television", "feed": "https://www.channelstv.com/feed/", "site": "https://www.channelstv.com"},
    {"name": "BusinessDay", "feed": "https://businessday.ng/feed/", "site": "https://businessday.ng"},
    {"name": "Nigerian Tribune", "feed": "https://tribuneonlineng.com/feed/", "site": "https://tribuneonlineng.com"},
]

FEED_BY_URL = {item["feed"]: item for item in SOURCE_REGISTRY}

TOPICS = {
    "Politics": ["election", "senate", "governor", "president", "minister", "assembly", "inec", "party", "political", "tinubu", "lawmaker", "cabinet", "policy"],
    "Business": ["market", "naira", "inflation", "bank", "economy", "investor", "trade", "oil", "business", "revenue", "tax", "cbn", "stock", "fuel price"],
    "Sports": ["football", "super eagles", "afcon", "coach", "league", "match", "player", "sport", "boxing", "basketball", "athletics"],
    "Entertainment": ["nollywood", "music", "album", "film", "movie", "actor", "actress", "afrobeats", "artist", "show", "grammy"],
    "Technology": ["tech", "startup", "digital", "internet", "phone", "software", "artificial intelligence", "data", "cyber", "telecom", "network"],
    "Health": ["health", "hospital", "doctor", "disease", "malaria", "vaccine", "medical", "outbreak", "cholera", "nutrition"],
    "Education": ["school", "student", "university", "jamb", "waec", "asuu", "education", "pupil", "teacher", "admission"],
    "Security": ["police", "military", "bandit", "kidnap", "boko haram", "attack", "security", "army", "gunmen", "terror", "robbery"],
    "Metro": ["lagos", "abuja", "road", "traffic", "market", "residents", "community", "fire", "flood", "court", "arrest"],
}
STOPWORDS = {"the", "a", "an", "and", "or", "of", "in", "on", "to", "for", "with", "at", "by", "from", "as", "is", "are", "was", "were", "be", "been", "has", "have", "had", "its", "it", "this", "that", "after", "over", "into", "says", "say", "new", "says"}


class NewsError(RuntimeError):
    pass


@dataclass
class FeedItem:
    title: str
    link: str
    summary: str
    published_at: str
    source: str
    source_url: str
    topic: str
    fetched_at: str
    related_sources: list[str]
    dedupe_key: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title, "link": self.link, "summary": self.summary,
            "published_at": self.published_at, "source": self.source,
            "source_url": self.source_url, "topic": self.topic,
            "fetched_at": self.fetched_at, "related_sources": self.related_sources,
            "dedupe_key": self.dedupe_key, "origin": "feed",
        }


def strip_html(value: str) -> str:
    value = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", value)
    value = re.sub(r"(?s)<[^>]+>", " ", value)
    value = html.unescape(value)
    return " ".join(value.split())


def summarize(text: str, limit: int = 320) -> str:
    clean = strip_html(text)
    clean = re.sub(r"(?i)\b(continue reading|read more|the post .* appeared first on .*)\b.*$", "", clean).strip()
    if len(clean) <= limit:
        return clean
    cut = clean[:limit]
    boundary = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if boundary > 120:
        return cut[:boundary + 1]
    return cut.rsplit(" ", 1)[0] + "…"


def parse_datetime(value: str | None) -> str:
    if not value:
        return ""
    value = value.strip()
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds")


def classify(title: str, summary: str) -> str:
    text = (title + " " + summary).casefold()
    scores = {topic: 0 for topic in TOPICS}
    for topic, words in TOPICS.items():
        for word in words:
            if re.search(r"(?<![a-z])" + re.escape(word) + r"(?![a-z])", text):
                scores[topic] += 1
    best = max(scores, key=lambda topic: scores[topic])
    return best if scores[best] else "General"


def _tokens(title: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", title.casefold())
    return {word for word in words if word not in STOPWORDS and len(word) > 2}


def similarity(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def similar_story(left: str, right: str) -> bool:
    """Conservative near-duplicate check for headlines.

    Rewritten headlines often lose different words, so Jaccard alone is too strict.
    Containment of most of the shorter headline's meaningful words is a better
    signal while still requiring strong overlap.
    """
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return False
    jaccard = len(a & b) / len(a | b)
    containment = len(a & b) / min(len(a), len(b))
    return jaccard >= 0.72 or (containment >= 0.66 and len(a & b) >= 3)


def _allowed_link(link: str) -> bool:
    parsed = urlparse(link)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc) and "." in parsed.netloc


def _canonical_link(link: str) -> str:
    parsed = urlparse(link)
    return f"{parsed.scheme}://{parsed.netloc.lower()}{parsed.path.rstrip('/')}"


def dedupe_key(title: str, link: str) -> str:
    basis = _canonical_link(link) if link else " ".join(sorted(_tokens(title)))
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]


def fetch_feed(url: str, timeout: int = 12, max_bytes: int = 4_000_000) -> list[dict[str, str]]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Ara/0.3 (local Nigerian news reader; respects publisher feeds)",
            "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml;q=0.9",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(max_bytes + 1)
            final_url = response.geturl()
    except urllib.error.HTTPError as exc:
        raise NewsError(f"HTTP {exc.code} from {url}") from exc
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        raise NewsError(f"Cannot reach {url}: {exc}") from exc
    if len(raw) > max_bytes:
        raise NewsError(f"{url} returned more than {max_bytes // 1_000_000} MB")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise NewsError(f"{url} did not return valid RSS or Atom XML: {exc}") from exc
    items: list[dict[str, str]] = []
    atom = "{http://www.w3.org/2005/Atom}"
    for node in root.iter():
        tag = node.tag.split("}")[-1]
        if tag not in {"item", "entry"}:
            continue
        def text(*names: str) -> str:
            for name in names:
                found = node.find(name)
                if found is not None and found.text:
                    return found.text.strip()
                found = node.find(atom + name) if not name.startswith("{") else None
                if found is not None and found.text:
                    return found.text.strip()
            return ""
        title = text("title")
        if not title:
            continue
        link = ""
        link_node = node.find("link")
        if link_node is None:
            link_node = node.find(atom + "link")
        if link_node is not None:
            link = (link_node.text or "").strip() or link_node.attrib.get("href", "").strip()
        if not link:
            link = text("guid", "id")
        if not _allowed_link(link):
            continue
        description = text("description", "summary", "encoded")
        published = text("pubDate", "published", "updated", "date")
        items.append({
            "title": strip_html(title),
            "link": link,
            "summary": summarize(description),
            "published_at": parse_datetime(published),
        })
    return items


def _source_for(feed_url: str) -> dict[str, str]:
    if feed_url in FEED_BY_URL:
        return FEED_BY_URL[feed_url]
    host = urlparse(feed_url).hostname or "Unknown"
    return {"name": host.replace("www.", ""), "feed": feed_url, "site": f"https://{host}"}


def parse_items(
    feed_url: str,
    records: Iterable[dict[str, str]],
    fetched_at: str | None = None,
) -> list[FeedItem]:
    source = _source_for(feed_url)
    fetched = fetched_at or now()
    result: list[FeedItem] = []
    for record in records:
        title = strip_html(record.get("title", ""))
        link = record.get("link", "").strip()
        if not title or not _allowed_link(link):
            continue
        published = record.get("published_at", "")
        if published:
            try:
                datetime.fromisoformat(published)
            except ValueError:
                published = ""
        result.append(FeedItem(
            title=title, link=link, summary=summarize(record.get("summary", "")),
            published_at=published, source=source["name"], source_url=source["site"],
            topic=classify(title, record.get("summary", "")), fetched_at=fetched,
            related_sources=[], dedupe_key=dedupe_key(title, link),
        ))
    return result


def refresh_news(
    store: Store,
    config: Config,
    feeds: Iterable[str] | None = None,
    offline: bool = False,
) -> dict[str, Any]:
    if offline:
        cached = store.news_items(config.news_max_items)
        return {"offline": True, "added": 0, "updated": 0, "cached": len(cached), "errors": [], "feeds_ok": 0, "feeds_failed": 0, "fetched_at": store.news_last_fetch()}
    urls = list(feeds or config.news_feeds or [item["feed"] for item in SOURCE_REGISTRY])
    existing = store.news_items(limit=500)
    known_keys = {item["dedupe_key"]: item for item in existing}
    fetched_at = now()
    collected: list[FeedItem] = []
    errors: list[str] = []
    ok = 0
    for url in urls:
        try:
            records = fetch_feed(url, timeout=config.news_timeout)
            items = parse_items(url, records, fetched_at)
            if items:
                ok += 1
            collected.extend(items)
        except NewsError as exc:
            errors.append(str(exc))
    # Deduplicate within this batch and against cached stories, keeping the
    # competing publishers visible as related sources rather than discarding them.
    unique: list[FeedItem] = []
    updates: list[dict[str, Any]] = []
    for item in collected:
        duplicate: FeedItem | dict[str, Any] | None = known_keys.get(item.dedupe_key)
        if duplicate is None:
            for other in unique:
                if similar_story(item.title, other.title):
                    duplicate = other
                    break
        if duplicate is None:
            for cached in existing:
                if similar_story(item.title, cached["title"]):
                    duplicate = cached
                    break
        if duplicate is None:
            unique.append(item)
            continue
        label = f"{item.source}: {item.link}"
        if isinstance(duplicate, FeedItem):
            if item.source != duplicate.source and label not in duplicate.related_sources:
                duplicate.related_sources.append(label)
        else:
            related = list(duplicate.get("related_sources") or [])
            if item.source != duplicate.get("source") and label not in related:
                related.append(label)
                copy = dict(duplicate)
                copy["related_sources"] = related
                updates.append(copy)
            duplicate["related_sources"] = related
    added, updated = store.save_news([item.as_dict() for item in unique[: config.news_max_items]] + updates)
    return {
        "offline": False,
        "added": added,
        "updated": updated,
        "feeds_ok": ok,
        "feeds_failed": len(errors),
        "errors": errors,
        "fetched_at": fetched_at,
        "cached": len(store.news_items(config.news_max_items)),
    }


def enrich(items: list[dict[str, Any]], knowledge=None) -> list[dict[str, Any]]:
    """Add freshness, age, and optional historical context without inventing facts."""
    current = datetime.now(timezone.utc)
    result = []
    for item in items:
        copy = dict(item)
        published = item.get("published_at") or ""
        age_hours = None
        if published:
            try:
                moment = datetime.fromisoformat(published)
                if moment.tzinfo is None:
                    moment = moment.replace(tzinfo=timezone.utc)
                age_hours = round((current - moment).total_seconds() / 3600, 1)
            except ValueError:
                age_hours = None
        copy["age_hours"] = age_hours
        copy["fresh"] = bool(age_hours is not None and age_hours <= 48)
        copy["freshness_label"] = (
            "current" if copy["fresh"] else
            ("recent" if age_hours is not None and age_hours <= 24 * 7 else
             ("older" if age_hours is not None else "date not provided"))
        )
        if knowledge is not None:
            context = knowledge.history_context(item.get("title", "") + " " + item.get("summary", ""), limit=1)
            copy["historical_context"] = context
        result.append(copy)
    return result


def news_report(store: Store, config: Config, refresh: bool = False, offline: bool = False, topic: str | None = None, knowledge=None) -> dict[str, Any]:
    state = {"offline": offline, "added": 0, "updated": 0, "errors": [], "fetched_at": store.news_last_fetch()}
    if refresh and not offline:
        try:
            state = refresh_news(store, config, offline=False)
        except NewsError as exc:
            state = {"offline": True, "added": 0, "updated": 0, "errors": [str(exc)], "fetched_at": store.news_last_fetch()}
    items = store.news_items(config.news_max_items, topic=topic)
    state["items"] = enrich(items, knowledge)
    state["topics"] = store.news_topics()
    state["sources"] = [{"name": item["name"], "site": item["site"]} for item in SOURCE_REGISTRY]
    state["registry_verified_on"] = VERIFIED_ON
    return state


def format_news(item: dict[str, Any]) -> str:
    published = item.get("published_at") or "date not provided"
    related = item.get("related_sources") or []
    lines = [
        f"{item['title']}",
        f"  {item.get('source', 'Unknown source')} · {published} · {item.get('freshness_label', '')}",
        f"  {item.get('summary', '')}",
        f"  {item.get('link', '')}",
    ]
    if related:
        lines.append(f"  Also reported by: {', '.join(str(value).split(':')[0] for value in related[:4])}")
    if item.get("historical_context"):
        lines.append(f"  Context: {item['historical_context']}")
    return "\n".join(lines)
