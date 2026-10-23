# Àrà

Àrà is a Nigerian AI you can use right now, in a browser. Open the site, type or speak, and it answers with Nigerian context: expressions with their real registers and regions, history and culture notes, and news from Nigerian publishers kept with their sources, links, and dates. It remembers what you explicitly tell it to — and lets you correct, export, or delete every note.

No install and no gigabyte downloads for the user. When the model is unreachable, Àrà says so plainly and shows retrieved notes labelled `[retrieval-only]` — never dressed up as generated answers.

## Access the web app

The web interface **is** Àrà. On a deployment, open the site URL and start talking — no account is created or required (an operator may put a shared password in front of it).

To run it yourself:

```sh
git clone https://github.com/ayoistooslick/ara.git && cd ara
python3 -m pip install .      # Python 3.10+; downloads ~1.5 MB, no weights
ara                           # serves the web app at http://127.0.0.1:8787
```

Open that address in a browser. Your data lives in `~/.local/share/ara/` (set `ARA_HOME` to move it); upgrades never delete it. `ara doctor` checks the installation in plain English. `ara chat` opens a secondary terminal interface for maintenance and offline use — it is not the product.

## What ara-ng is

**ara-ng is my own model, trained by me on my own training data.** It is the model Àrà replies through and the only model name the product shows: the website, the model pill, the About panel, and the API all say `ara-ng`. The server holds the model; the user's phone downloads nothing.

Technically, the backend is an OpenAI-compatible hosted endpoint: set `ARA_HOSTED_URL` (HTTPS) and, if it needs one, `ARA_HOSTED_API_KEY` — the key is read only from the environment and never written to config or backups. The optional `ara model download` catalogue (third-party Qwen GGUF files, Apache-2.0, licences in `THIRD_PARTY_NOTICES.md`) is a local **offline fallback** for people who want inference without any server; it is not Àrà's identity and is never presented as ara-ng.

## Genuine limitations

- **A deployment with no reachable model** answers in retrieval-only mode, clearly labelled on every reply. The operator must configure `ARA_HOSTED_URL` for real ara-ng answers.
- **Vision and voice input** need operator-configured endpoints; camera and microphone work in the browser only in secure contexts (HTTPS or localhost).
- **Memory is per installation** (one SQLite file), not a cloud account. Web sessions are separate conversation threads; there are no user accounts and no telemetry.
- **The knowledge corpus is curated and fallible** — reference notes with confidence scores, not the final word on Nigeria. News is cached with real publication dates and is never invented.
- **ara-ng's weights and training data are not in this repository**; the hosted service is configured externally by whoever runs it.

## Tests

```sh
python -m unittest discover -s tests
```

The suite runs offline with no API key and no weights. It covers web sessions, CSRF, rate limits, streaming, retrieval-only labelling, memory persistence and corrections, news parsing, uploads, the model adapter against local HTTP stubs, corpus validation, and downloads.

More detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/MODELS.md](docs/MODELS.md), [docs/WEB.md](docs/WEB.md), [docs/NIGERIAN_KNOWLEDGE.md](docs/NIGERIAN_KNOWLEDGE.md), [docs/NEWS.md](docs/NEWS.md), [docs/TRAINING.md](docs/TRAINING.md), [deploy/README.md](deploy/README.md).

Àrà is built by Ayodele Ayokunle David. Green, white, and still curious.
