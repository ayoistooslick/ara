# Àrà (`ara`)

Àrà is a Nigerian AI I'm building in the open. It lives in your terminal and your browser, keeps its own local memory, knows Nigerian expressions and history, and reads Nigerian news without inventing headlines.

It starts in well under a second and does **not** download model weights when you install it. The hosted model I'm building it around is called **ara-ng**; locally it can talk through any compatible model you run yourself. You can use hosted ara-ng, a local model, or no model at all — in which case Àrà says plainly that it is showing retrieved notes, not generated answers.

## Install

You need Python 3.10 or newer. On Android with Termux:

```sh
pkg update && pkg install python git
git clone https://github.com/ayoistooslick/ara.git && cd ara && python3 -m pip install .
```

On Linux or macOS the same clone-and-install line works if `python3` and `git` are already there:

```sh
git clone https://github.com/ayoistooslick/ara.git && cd ara && python3 -m pip install .
```

That is the whole install. It downloads this repository (about 1.5 MB) and one small dependency (`tomli`, only on Python 3.10). The installed package is 918 KB; a fresh virtualenv around it is 20 MB, mostly pip itself. **No model weights, datasets, or training dependencies are downloaded.** `pip install` took 2.5 s here; creating a fresh venv and installing took 6.7 s.

## Start chatting

```sh
ara
```

That opens chat directly, about 0.13 s to the welcome screen on this sandbox. You get the frog, one line telling you which backend is active, and a `you>` prompt. Type `/help` inside chat for commands, `/exit` to leave.

If something looks broken, run `ara doctor`. It checks Python, the database, the backend, memory, news cache, and the knowledge corpus, and tells you what to fix in plain English.

Your data lives in `~/.local/share/ara/` (SQLite database plus `config.toml`). Set `ARA_HOME` to keep it somewhere else. Upgrading the app never deletes it; the database migrates itself with a backup first.

## How inference works

Àrà picks the first backend that is set up:

1. **Hosted ara-ng.** Set `ARA_HOSTED_URL` to an OpenAI-compatible endpoint and `ARA_HOSTED_API_KEY` if it needs a key. The model name sent to it is `ara-ng`. Hosted is used automatically when configured, must be HTTPS, and the key is read only from the environment — never written to `config.toml` or a backup.
2. **A local model server.** Any OpenAI-compatible server works; the default is llama.cpp at `http://127.0.0.1:8080/v1`. Point it elsewhere with `ARA_MODEL_URL`.
3. **Retrieval-only.** With no model, replies come from your saved notes and the Nigerian knowledge corpus. Every such reply starts with `[retrieval-only]` and is never counted or styled as model output.

The active backend is shown at startup, in `/model`, in `ara status`, and in the web interface. You can force one with `ARA_BACKEND=hosted|local|retrieval`.

Me on this sandbox (2 vCPU, Qwen3-0.6B Q8, 2,048-token context): across runs, first token arrived after 23.5–37.2 s and a full answer after 32–44 s. The model process used about 1.27 GiB of RAM. That is a real local-model measurement, not a promise; a phone is slower, a laptop is faster, and a hosted endpoint skips the local compute entirely. Retrieval-only replies took about 0.13 s. No live hosted ara-ng service was available here, so hosted latency against the real service is unmeasured.

## Optional local models

Only do this if you want offline inference. Àrà works without it.

```sh
ara model download qwen3-0.6b     # 610 MB, about 1.5 GB RAM
ara model download qwen3-1.7b     # about 1.1 GB, about 2 GB RAM
ara model download qwen3-4b       # about 2.3 GB, about 4 GB RAM
```

It prints the exact download size, the free storage it needs, the RAM it wants, and the licence, then asks before starting. Downloads resume with HTTP Range requests, are checked against the publisher's SHA-256 when one is available, and only appear under their real name after the check passes. You can also pass your own `.gguf` URL with `--out` and `--sha256`. The catalogue sizes above were confirmed against Hugging Face without downloading the files.

Then start llama.cpp yourself (Àrà does not start servers or download them silently):

```sh
llama-server -m ~/.local/share/ara/models/Qwen3-0.6B-Q8_0.gguf --host 127.0.0.1 --port 8080 -c 4096
ara model status
```

`ara model status` reports the live server, the served model, and the context size. More detail, including sizing and quantisation notes, is in [docs/MODELS.md](docs/MODELS.md).

## Your stuff stays yours

- Conversations, memories, corrections, goals, and sessions are in one SQLite file under your home directory.
- Memories can be inspected, corrected, forgotten, and exported: `ara memory inspect`, `ara memory export notes.json`, `ara memory forget-all --yes`.
- `ara backup` makes a consistent copy; `ara restore FILE` verifies the backup and saves a pre-restore copy of your current data before replacing anything.
- Model weights live in `models/` (ignored by git); caches are separate. Upgrades do not touch either.
- No telemetry. Network requests happen only for a backend you configured, news you asked to refresh, or a perception endpoint you configured.

## The rest of the terminal

```sh
ara ask "How far?"               # one message
ara conversations                # saved threads
ara history --limit 20
ara news --refresh               # publisher feeds; cached reading offline
ara nigeria search "no wahala"   # curated context, with confidence
ara nigeria eval                 # offline retrieval/interpretation eval
ara web                          # browser interface on 127.0.0.1:8787
```

Inside chat: `/model`, `/memory`, `/history`, `/new`, `/clear`, `/remember`, `/study`, `/news`, `/explain`, `/image`, `/audio`, `/doctor`, `/exit`. End a line with `\` to continue on the next one. `ara status` and `ara doctor` also show how much space the database, backups, models, and caches use.

The browser version uses the same engine and the same memories. It asks for camera or microphone permission only after you press those controls, shows a strip while any sensor is active, and releases all of them with one stop control. See [docs/WEB.md](docs/WEB.md) for deployment and the security model.

## Tests and honest limits

```sh
python -m unittest discover -s tests
```

The suite runs offline and needs no API key or model weights. It covers CLI and chat persistence, memory and migrations, backup/restore, model streaming and failure against local HTTP stubs, hosted ara-ng selection and auth headers, retrieval-only labelling, resumable checksum-verified downloads, Nigerian corpus validation and evaluation, news parsing and offline reading, web auth/CSRF/rate limits/uploads, and brand rendering.

What I could not verify here: a live hosted ara-ng service, a physical Android phone or Termux install, real camera/microphone capture on a device, and real speech/vision quality. The local model number above is from a 0.6B model on this sandbox and says nothing about larger models or phones.

More detail lives in `docs/`: [architecture](docs/ARCHITECTURE.md), [models](docs/MODELS.md), [web](docs/WEB.md), [Nigerian knowledge](docs/NIGERIAN_KNOWLEDGE.md), [news](docs/NEWS.md), [training](docs/TRAINING.md), and [deployment](deploy/README.md).

## What Àrà is not

Àrà is not a model trained from scratch. It is an application around a pretrained model, with a curated Nigerian-context layer, memory, news, and two interfaces. The small training pipeline in `ara/training/` is a CPU teaching baseline for learning how training works; it is not the brain you chat with. The knowledge corpus is fallible reference material with confidence levels, not the final word on Nigeria.
