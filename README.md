# Ara

**The AI that belongs in Nigeria.** Ara is a local-first AI companion built by Ayodele Ayokunle David. It is designed around Nigerian languages, humour, history, and everyday context, and it gives that same AI two front doors: a terminal and a browser.

Ara is not a Nigerian skin over a generic chatbot. Its Nigerian-context layer is an inspectable, versioned knowledge corpus with retrieval, interpretation, evaluation, and provenance. Its model layer is deliberately replaceable: Ara does not assume that a Nigerian AI must be Qwen, Llama, or any other single model. The default is a local OpenAI-compatible inference server; cloud inference is opt-in, never hidden.

The project is honest about the boundary between what is built and what is external. Weights, a model server, browser camera hardware, and Android devices are not bundled. Everything that can be tested offline is tested; hardware-dependent paths are labelled as such.

## What is implemented

- **One shared core.** Terminal and web use the same `ConversationEngine`, `Store`, `KnowledgeBase`, model adapter, memory retrieval, safety rules, and configuration.
- **Terminal Ara.** Multi-turn chat, persistent conversations, user-approved memories and corrections, goals, decisions, skills, bounded file tools, image analysis, audio transcription, news, system diagnostics, and a real ASCII frog animation.
- **Web Ara.** A responsive single-page interface with session auth, streaming replies, memory controls, conversation history, Nigerian-context lookup, news, image upload, live camera capture, microphone recording with server transcription, and browser speech synthesis. No client-side secrets; uploads are processed in memory and not saved.
- **Nigerian intelligence.** 111 curated entries (65 expressions, 23 history, 23 culture), context-sensitive interpretation, ambiguity handling, regional and generational caveats, plus a 35-case retrieval and 20-case interpretation evaluation suite. Results are measured, not asserted.
- **Replaceable models.** A `ModelAdapter` interface, an OpenAI-compatible local adapter, capability discovery, context fitting, token budgets, timeouts, streaming with cancellation, and a separate vision endpoint. Optional local embedding and speech-to-text endpoints.
- **Hybrid memory.** Lexical search plus semantic retrieval when a real embedding endpoint is configured. Without one, Ara uses a deterministic hashed-vector fallback and reports `semantic: false` rather than calling it semantic search.
- **Nigerian News Plate.** Publisher RSS feeds, source attribution, publication dates, categories, near-duplicate collapse, cached offline reading, and explicit freshness labels. Ara never generates a headline or quote.
- **Learning companion.** A study mode that explains in steps, uses Nigerian examples where relevant, asks checking questions, and cites retrieved knowledge.
- **Frog identity.** An original green-and-white frog SVG used in the browser, a matching seven-line ASCII frog in the terminal, a CSS hopping loop, and a real terminal frame animation with non-interactive and reduced-motion fallbacks.
- **Migrations and recovery.** A versioned SQLite schema, automatic pre-migration backups, integrity checks, backup/restore, and user-controlled memory deletion.
- **Training pipeline.** Reproducible tokenizer training, dataset preparation, a small counted n-gram baseline trained from initialization on CPU, checkpoints, resume, evaluation, model cards, and a fine-tuning export bundle that never claims a fine-tune was run.

## One core, two interfaces

```text
Terminal CLI ─┐
              ├─ ConversationEngine ─ Store (SQLite) ─ KnowledgeBase
Web server  ──┘         │                    │
                        ├─ ModelAdapter      ├─ hybrid memory retrieval
                        ├─ perception        └─ news cache
                        └─ Nigerian interpretation
```

The web server is a thin HTTP/SSE layer over the same engine. It does not reimplement prompting, memory, or Nigerian-context logic. Memory is shared between terminal and web on one installation; web conversations are isolated per session.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for modules, schema, and request flow.

## Requirements

- Python 3.10 or newer. On 3.10, `pip install .` pulls in `tomli` automatically; 3.11+ uses the built-in `tomllib`.
- A local OpenAI-compatible model server for real conversations. Ara does not download weights or run inference itself.
- No GPU is required for Ara. Inference speed and memory fit depend on the model and runtime you choose.
- A modern browser for Web Ara. Camera and microphone require HTTPS (or `localhost`) and explicit browser permission.

### Linux

```sh
git clone https://github.com/ayoistooslick/ara.git
cd ara
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
ara --help
```

### Android with Termux

```sh
pkg update
pkg install python git
git clone https://github.com/ayoistooslick/ara.git
cd ara
python -m pip install .
ara doctor
```

Ara's code is standard-library Python and should run in Termux. A physical Android device was **not** available for verification; see [Known limitations](#known-limitations). Running a model server inside Termux is a separate hardware decision. Use `ara doctor` and the profiles in [docs/MODELS.md](docs/MODELS.md) before downloading a multi-gigabyte file.

## Connect a model

Ara defaults to `http://127.0.0.1:8080/v1`, which is what `llama-server` exposes. It refuses non-loopback endpoints unless `model.allow_remote = true` is set deliberately.

```sh
# Example: llama.cpp with a compatible GGUF you downloaded yourself
./llama-server -m /path/to/Qwen3-4B-Q4_K_M.gguf --host 127.0.0.1 --port 8080 -c 4096

# In another terminal
ara model status
ara ask "How far?"
ara
```

Ara does not bundle or auto-download weights. Model size, quantization, memory fit, and the difference between a model trained from scratch, a fine-tune, and an untouched pretrained model are documented in [docs/MODELS.md](docs/MODELS.md). The 8B profile is not promised to run on a phone.

Configuration lives at `~/.local/share/ara/config.toml` (or under `$ARA_HOME`). `ara` creates a commented default on first start. Every setting can be overridden with an `ARA_*` environment variable. Key options include the model URL and name, context budget, temperature, allowed file roots, embedding/vision/audio endpoints, news feeds, and web host/port.

Optional local services (all OpenAI-compatible, all disabled by default):

```toml
[embeddings]
url = "http://127.0.0.1:8081/v1"
model = "your-local-embedding-model"

[vision]
url = "http://127.0.0.1:8082/v1"
model = "your-local-vision-model"

[audio]
url = "http://127.0.0.1:8083/v1"
model = "your-local-whisper-model"
language = "en"
```

A text-only chat model is **not** treated as vision-capable. Image understanding is disabled until a real vision endpoint is configured; audio transcription is disabled until a real speech-to-text endpoint is configured.

## Terminal Ara

```sh
ara                              # resume the active conversation
ara ask "Explain compound interest simply"
ara ask --mode learn "Teach me quadratic equations"
ara conversations                # list saved threads
ara --conversation-id ID         # resume a thread
ara history --limit 20
ara status                       # schema, counts, knowledge, model
ara doctor                       # installation, memory, perception, news
ara model status                 # live server and capability report

ara memory inspect
ara memory add "I prefer short answers" --kind preference
ara memory correct 2 "I prefer short answers in chat"
ara memory forget 2
ara memory export ./memories.json
ara memory embed                 # compute vectors, report semantic true/false
ara memory forget-all --yes

ara nigeria search "no wahala"
ara nigeria explain "E don choke"
ara nigeria eval                 # retrieval + interpretation, offline
ara nigeria eval --model         # also keyword rubric on real model answers
ara nigeria validate
ara nigeria stats
ara nigeria manifest

ara news                         # cached stories
ara news --refresh               # fetch publisher feeds when online
ara news --offline --limit 8
ara news --topic Politics --json

ara vision photo.jpg             # requires [vision] endpoint
ara audio voice.m4a              # requires [audio] endpoint
ara animate                      # hopping ASCII frog in a TTY
ara web                          # start Web Ara on 127.0.0.1:8787
```

Inside the interactive terminal: `/help`, `/memory`, `/history`, `/new`, `/skills`, `/calc`, `/remember`, `/news`, `/explain`, `/image PATH`, `/audio PATH`, `/study`, `/embed`, `/doctor`, `/animate`, `/exit`.

Existing commands for goals, decisions, skills, tools, backups, restore, identity, and the bounded autonomous loop are retained. `ara tool write-file` still requires explicit confirmation and never overwrites; no tool runs arbitrary shell commands.

Terminal sensor access is intentionally limited. Ara can analyse files you provide. It does not pretend a terminal process can read your phone's camera or microphone. An optional permissioned Termux sensor bridge is not implemented or claimed; see [Known limitations](#known-limitations).

## Web Ara

```sh
ara web
# open http://127.0.0.1:8787
```

Web Ara starts on loopback only. Binding to a non-loopback address without `ARA_WEB_PASSWORD` is refused. For deployment, put it behind an HTTPS reverse proxy; examples are in `deploy/`. Never expose the Python server directly to the public internet.

The interface supports streaming chat (SSE), study mode, conversation history, memory add/correct/forget/export, Nigerian-context lookup, news with refresh/offline states, image upload, live camera capture, microphone recording and transcription, and spoken replies via the browser's speech synthesis.

Browser permissions are requested only after you press the microphone or camera control. A visible strip shows every active sensor, and one control releases all of them. Camera frames are captured only when you press **Analyse**, not continuously. Recordings stop automatically after 60 seconds, when the tab is hidden, or when you stop them. Uploads are processed in memory and are not written to the database. Denial, missing hardware, unsupported browsers, and transcription failures leave text chat usable.

Spoken output uses the device's installed voices. Nigerian English quality varies by browser and operating system; Ara does not claim accent accuracy that has not been measured.

See [docs/WEB.md](docs/WEB.md) for permissions, security, and a reproducible deployment setup.

## Nigerian intelligence

The corpus lives in `ara/data/nigeria/` as versioned JSON with an explicit licence and provenance statement. It covers:

- **Expressions:** Pidgin, Nigerian English, regional borrowings, Gen Z internet usage, insults and their social risk, transport, food, money, and everyday institutions.
- **History:** documented events and civic context with dates and caveats where interpretation is contested.
- **Culture:** social practices, education, religion, family, music, and regional differences, written without flattening Nigeria into one culture.

Interpretation returns a register, matched entries, meanings, misreadings to avoid, confidence, and a clarification flag when the same words can genuinely mean different things. It does not "translate slang" and stop there.

The evaluation suite reports measured retrieval and interpretation accuracy, broken down by failure, and a keyword-based rubric for model answers. The rubric is not a human cultural evaluation; the CLI labels it accordingly.

Data is curated, not scraped from private messages or restricted groups. Imports require a licence and consent basis, reject personal data and private material, deduplicate by ID, and record provenance. See [docs/NIGERIAN_KNOWLEDGE.md](docs/NIGERIAN_KNOWLEDGE.md) for sources, validation, versioning, and removal.

## Memory and learning

Ara stores conversation messages, memories, identity fields, goals, decisions, skills, news, and web sessions in SQLite. Schema upgrades run automatically with a pre-migration backup and a version check; a newer database is refused rather than silently downgraded.

Memory is retrieved per turn using a hybrid score:

1. Lexical overlap and confidence.
2. Cosine similarity against stored vectors.
3. A configurable weight for corrections and recent entries.

Vectors come from a configured local embedding server. Without one, Ara uses a deterministic hashed-vector fallback that improves lexical recall but is **not** semantic understanding; `ara memory embed` prints `"semantic": false` in that case.

Learning is conservative and rule-based:

- Explicit "remember", preference, name, and correction statements are captured.
- Credential-looking text is redacted before storage and rejected as a memory.
- Automatically detected corrections are stored as separate correction memories alongside the original; conflicts are surfaced for review. The explicit **Correct** action replaces that one note's text after the user asks for it.
- Every memory has a source and confidence. The user can inspect, correct, export, or forget it.
- Prompt changes and memory writes are not model training. Ara never retrains on a conversation automatically.

The same rules apply in terminal and web. The web adds a session boundary for conversations, not for shared memories on the same installation.

## News Plate

Ara reads configured publisher RSS feeds, keeps the publisher's title, summary, link, and publication time, classifies a topic, and stores the result locally. Source and link are always shown. Near-duplicate headlines from different publishers are collapsed with the competing source recorded. Cached stories remain readable offline and are labelled with their actual publication date and age; old stories are never presented as breaking news.

Default sources are Premium Times, Punch, Vanguard, Daily Trust, Channels Television, BusinessDay, and Nigerian Tribune. Feeds change; a failing feed is reported, never replaced with invented content. Ara does not generate headlines, quotes, or claims about current events from model memory.

`ara news --refresh` fetches; `ara news --offline` only reads the cache. Web exposes the same distinction.

## Learning companion

`/study` (terminal) or Study mode (web) changes how Ara explains, not what it knows. It is instructed to:

- Explain at the level the user demonstrates, with a concrete example first.
- Use Nigerian examples where they genuinely help, not as decoration.
- Ask a checking question and respond to mistakes without dumping the answer.
- Retrieve from the Nigerian corpus when relevant and name uncertainty.
- Refuse to invent curriculum facts or citations.

This is conversational scaffolding around a replaceable model, not a separate tutor product.

## Brand and animation

The mascot is an original SVG frog in Nigerian green (`#008751`) and white, with a wide body, domed eyes, and a white throat. The same silhouette is reproduced in seven-line ASCII frames. The browser uses CSS transforms for a crouch–hop–land loop and a separate shadow squash; `prefers-reduced-motion` stops it. The terminal draws distinct frames with cursor restoration, falls back to a static frog for pipes, dumb terminals, narrow widths, `NO_COLOR`, or `ARA_NO_ANIMATION`, and never corrupts logs.

## Privacy and safety

- Local by default. The model endpoint must be loopback unless remote access is explicitly enabled.
- No telemetry, no hidden cloud calls, no automatic model download.
- The web client contains no secrets. Session cookies are `HttpOnly` and `SameSite=Strict`; CSRF tokens and same-origin checks protect state-changing requests.
- Uploads are size- and type-validated, held in memory, and not persisted. Vision and transcription are disabled unless configured.
- Sensors are permission-gated, visibly indicated, and stopped on tab hide, page unload, completion, or the stop control.
- File tools are restricted to configured roots; the web exposes no filesystem or shell access.
- Ara avoids medical, legal, financial, and safety-critical certainty and directs users to qualified people where appropriate.
- Rate limits, login throttling, security headers, a restrictive CSP, static path traversal checks, and per-session conversation isolation are implemented and tested.

Details: [docs/WEB.md](docs/WEB.md).

## Hardware and offline behaviour

Ara is built for affordable Android phones, low memory, and intermittent connectivity:

- The model is optional at runtime; without a server, text chat reports the model as unavailable instead of pretending.
- Knowledge search, Nigerian interpretation, evaluation, memory management, news cache, and animation work offline.
- News refresh, model chat, embeddings, vision, and transcription need their configured local services.
- `ara doctor` reports free memory and the size of a configured GGUF. The profiles in [docs/MODELS.md](docs/MODELS.md) are RAM-based starting points, not benchmark results.
- Model profiles: constrained (roughly 2–4 GB free), standard (4–10 GB), powerful (10 GB+). An 8B Q4 model needs several gigabytes beyond the file size for context and runtime overhead.

## Model development

The training package is real and small enough to run on CPU:

```sh
python -m ara.training.train --out runs/baseline
python -m ara.training.train --out runs/baseline --resume
python -m ara.training.train --out runs/baseline --eval-only
python -m ara.training.train --out runs/baseline --export-finetune
```

It trains a byte-level BPE tokenizer and a counted n-gram language model from initialization on the curated corpus, writes checkpoints and a model card, and evaluates next-token loss. **This is a teaching baseline, not a chat model and not Ara's production brain.** The export command writes instruction data and a fine-tuning plan; it does not run a fine-tune and marks the run `NOT RUN`.

Ara has not trained a production model from scratch. Large-model fine-tuning requires hardware and libraries that are not part of this repository. See [docs/TRAINING.md](docs/TRAINING.md) for configuration, checkpoint recovery, evaluation, and the honest boundary.

## Tests

```sh
python -m unittest discover -s tests -v
```

The suite is offline and needs no paid API or model weights. At the time of this commit it runs **376 tests**, all passing, covering:

- existing CLI behaviour, conversations, identity, goals, decisions, skills, tools, permissions, backup/restore, configuration failures, and migrations;
- model request/response formatting, streaming, context fitting, and unreachable-server errors against local HTTP stubs;
- memory retrieval, embeddings, conflicts, deletion, and persistence;
- Nigerian corpus validation, provenance, interpretation, evaluation, and ingestion privacy checks;
- news parsing, provenance, freshness, deduplication, feed failure, offline cache, and the documented-year rule for historical context;
- web sessions, auth, CSRF, origin checks, rate limits, upload validation, traversal, SSE chat, memory controls, and session isolation;
- vision and transcription contracts against local stub services;
- tokenizer round-trips, deterministic dataset preparation, training checkpoints, resume, evaluation, model cards, and export honesty;
- frog SVG, CSS animation, reduced motion, terminal animation, and non-interactive fallback.

What the suite does **not** prove: real LLM quality, real camera or microphone capture on a physical device, real speech recognition accuracy, real Android/Termux operation, or a live public deployment. Those require the hardware and services named in the verification paths below.

## Verified and unverified

Verified in this environment:

- Python 3.10 execution and the full offline suite.
- Real HTTP requests to stub model, vision, transcription, news, and embedding servers.
- SQLite migration from the previous schema with data preserved and a backup created.
- Static assets served with security headers; browser client syntax checked with Node.
- Terminal animation under a simulated TTY and static fallback for pipes.

Not verified:

- A physical Android phone, Termux install, or mobile browser.
- Browser camera/microphone permission dialogs, a real captured frame, or playback of synthesized speech.
- A real local LLM, vision model, embedding model, or Whisper server.
- Live headline retrieval at the time you read this; publisher feeds change.
- Public deployment, TLS termination, or long-running production load.
- Any fine-tune or quantization run.

## Deployment

`ara web` is a development-grade Python HTTP server. For a real deployment:

1. Run it on loopback behind Caddy or nginx with TLS.
2. Set a strong `ARA_WEB_PASSWORD` in the service environment (not in `config.toml`).
3. Keep the model server local or on a trusted private network.
4. Give the service a writable data directory with restrictive permissions.
5. Back up the SQLite database and test restore.

Ready-to-edit systemd and Caddy examples are in [`deploy/`](deploy/). The documented setup has not been deployed to a public host from this workspace; see [docs/WEB.md](docs/WEB.md) for the exact checks to run.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Models, hardware, and adapters](docs/MODELS.md)
- [Nigerian knowledge pipeline](docs/NIGERIAN_KNOWLEDGE.md)
- [Web interface, permissions, and deployment](docs/WEB.md)
- [News sources and retrieval](docs/NEWS.md)
- [Training and evaluation](docs/TRAINING.md)

## Known limitations

- Real model inference needs separately installed weights and a running server. No model is bundled.
- The hashed-vector fallback is not semantic search. True semantic retrieval requires a configured embedding endpoint.
- Corrections are surfaced but do not automatically rewrite conflicting memories.
- Skills reuse only existing tools; there is no arbitrary shell tool.
- The autonomous loop is bounded and interactive; it does not run unattended.
- Terminal sensor access is file-based. No Termux camera/microphone bridge is implemented.
- Browser speech recognition is not used; recording is transcribed by the configured server endpoint. Browser speech synthesis voice quality varies.
- Nigerian English and Pidgin speech recognition accuracy has not been measured.
- News depends on publisher feeds and network access; cached stories are labelled, not refreshed invisibly.
- The training baseline is an n-gram model for reproducibility, not a conversational model.
- Android testing, physical camera/microphone testing, real model quality, and public deployment remain unverified.
- No claim is made that Ara is Nigeria's first, largest, or most advanced AI.

## Contributing and licence

Run the tests before submitting changes. Keep databases, model weights, secrets, and personal configuration out of commits. Ara's source is Apache-2.0; model weights and third-party components keep their own licences. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Developer

Ara is made by Ayodele Ayokunle David, a teenage hobby programmer from Nigeria, interested in building independent software systems. [Portfolio](https://ayo-portal.netlify.app).

"Build things that matter."
