# Ara's News Plate

Ara's news is an attributed reader, not a headline generator. Every story comes from a publisher feed with its original title, summary, link, source, and publication time. Ara does not invent headlines, quotes, dates, or links, and it does not ask the language model to guess current events.

## Sources

Default registry:

| Publisher | Feed |
| --- | --- |
| Premium Times | `https://www.premiumtimesng.com/feed` |
| Punch | `https://punchng.com/feed/` |
| Vanguard | `https://www.vanguardngr.com/feed/` |
| Daily Trust | `https://dailytrust.com/feed/` |
| Channels Television | `https://www.channelstv.com/feed/` |
| BusinessDay | `https://businessday.ng/feed/` |
| Nigerian Tribune | `https://tribuneonlineng.com/feed/` |

The endpoints were checked for XML responses on 2026-10-22. Publishers change or block feeds; failures are reported per feed and are not replaced with fabricated content.

Override or extend feeds in `config.toml`:

```toml
[news]
timeout = 12
max_items = 40
feeds = [
  "https://www.premiumtimesng.com/feed",
  "https://punchng.com/feed/",
]
```

Use feeds you are permitted to read. Ara sends a normal user agent and does not bypass paywalls, authentication, or robots restrictions.

## Retrieval and storage

`refresh_news` fetches each feed, parses RSS or Atom, strips HTML, limits summaries, validates http(s) links, and classifies a topic from title and summary. It stores:

- title, summary, link, source name, publisher site;
- publication timestamp (UTC when parseable) and fetch timestamp;
- topic, dedupe key, related sources;
- origin (`feed`).

Stories without a valid title or link are skipped. A missing publication date is stored as `date not provided`; it is not silently replaced with the fetch time.

## Deduplication and attribution

Two levels:

1. **Canonical link/title hash** for exact repeats.
2. **Headline similarity** using Jaccard overlap and containment of meaningful words, requiring strong overlap.

When a competing publisher covers the same story, the earliest item remains the primary record and the other source plus link is stored in `related_sources`. This is transparent attribution, not a claim that one publisher copied another.

The summary is trimmed feed text, not an Ara-generated paragraph. If a feed supplies only a headline, Ara shows only the headline and link.

## Freshness

`enrich` computes age from the publication timestamp:

- `current`: no older than 48 hours;
- `recent`: no older than 7 days;
- `older`: older than 7 days;
- `date not provided`: cannot be aged, shown as such.

An old cached story is never labelled breaking or current. The CLI prints the actual ISO timestamp; the web shows the label and date.

Historical context is attached only when the story mentions a year (1900–2099) that the Nigerian corpus actually documents, such as 1993 for June 12 or 2020 for EndSARS. Word-overlap matching was tried and removed: generic tags like "court", "police", or "service" attached unrelated events to unrelated headlines. Stories with no documented year show no history note. The note is context, not a claim that the historical event caused the current story.

## Offline

- `ara news --offline`: read the local SQLite cache only.
- `ara news` without `--refresh`: read the cache and show its last fetch status.
- `ara news --refresh`: attempt each feed; failures are returned as warnings while cached items remain.
- Web `/api/news?offline=1` and the Refresh control use the same rules.

The cache survives restarts. It is not a live feed. If the cache is empty and the network is down, Ara says so.

## What it does not do

- Does not generate or rewrite headlines for entertainment.
- Does not claim a story is verified beyond "this publisher reported it".
- Does not merge conflicting facts into a synthetic article.
- Does not fetch article bodies or bypass publisher restrictions.
- Does not present opinion as verified reporting. The feed's own text and source label are preserved; Ara does not add a truth score it cannot support.
- Does not use model memory for current events. If asked about today's news without a cache, Ara should say it has no retrieved stories.

## Tests

`tests/test_news.py` covers RSS and Atom parsing, source attribution, missing dates, old-story labelling, near-duplicate collapse, feed failure preserving cache, invalid XML, and offline cache output. `tests/test_web.py` covers news endpoints. These use local stubs; they do not prove live publisher availability or the accuracy of today's reporting.
