# Ara

Ara is a local-first terminal companion created by Ayodele Ayokunle David. It keeps conversation history, explicitly signalled memories, identity settings, goals, and documented procedures in a local SQLite database. The language model is replaceable: Ara currently talks to a local OpenAI-compatible endpoint, such as `llama-server` from llama.cpp. Ara does not download weights, call a cloud AI service, or upload conversations on its own.

The project exists to explore what a small, inspectable companion system can do around a pretrained model. Persistent notes and procedures are useful, but they are not model training or evidence of sentience.

## What works

The CLI supports local multi-turn conversations when a compatible model server is running, explicit memory capture and correction, relevance-based memory lookup, durable identity fields, goals, transparent score-based decisions, versioned procedure notes, basic arithmetic, root-limited text file tools, database backup/restore, and diagnostic commands. Memory learning is deliberately conservative: Ara saves clearly signalled statements such as “I prefer …” and “Remember that …”; it does not treat every sentence as fact. Tools do not run merely because a generated response asks for them. There is no default shell-execution tool.

## Requirements and installation

Ara needs Python 3.11 or newer and uses only the Python standard library at runtime. No GPU is required for Ara itself. Model inference is performed by the separate local runtime and can need substantial RAM, storage, and patience; an 8B model may not fit an Android handset. `ara doctor` reports available system memory and the configured GGUF file size when a path is set, but cannot guarantee that a model will load.

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

Install Python and Git from Termux, then:

```sh
pkg update
pkg install python git
git clone https://github.com/ayoistooslick/ara.git
cd ara
python -m pip install .
ara --help
```

A local model runtime must separately be available on the device or reachable over a local network that you explicitly configure. Ara defaults to loopback and rejects non-local model endpoints unless `model.allow_remote = true` is deliberately added to `config.toml`. Do not assume the 8B model is practical on a phone.

## Configure a local model

The first `ara` start creates a local configuration file at `~/.local/share/ara/config.toml` (or under `$ARA_HOME`). Ara does not fetch model files automatically. Downloading is a separate, user-initiated action. Qwen’s model card lists Qwen3-8B and identifies Apache-2.0 licensing. Qwen’s local-runtime guide documents official GGUF files and llama.cpp support. A Q4_K_M example download is:

```sh
# Install/use Hugging Face's CLI separately if it is not already available.
hf download Qwen/Qwen3-8B-GGUF qwen3-8b-q4_k_m.gguf --local-dir "$HOME/models/qwen3"
```

Then start a llama.cpp server built with Qwen3 support (Qwen documents support in llama.cpp from `b5092`):

```sh
llama-server -m "$HOME/models/qwen3/qwen3-8b-q4_k_m.gguf" --host 127.0.0.1 --port 8080 -c 8192
```

The exact file name and command-line flags can vary by llama.cpp release. Point the optional `model.path` setting in Ara’s config at the GGUF file for diagnostics. Ara’s default endpoint is `http://127.0.0.1:8080/v1`. Verify it with `ara model status`, then run `ara`.

Model weights are separate from Ara and are not checked into this repository. Keep the model publisher’s licence with any copy you download. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Using Ara

```sh
ara                         # resume the active conversation (or start one)
ara conversations           # list saved threads
ara --conversation-id ID    # resume a listed thread
ara --help                  # list commands
ara status                  # data and model connection
ara history                 # show recent messages in the active conversation
ara model status            # local server and weight diagnostics
ara doctor                  # installation diagnostics
ara memory inspect          # inspect stored memories
ara memory export ./memories.json
ara memory add "I prefer short answers" --kind preference
ara memory correct 1 "Updated detail"
ara memory forget 1
ara identity show
ara goal add "Finish the project" --priority 4
ara decisions
ara tools list
ara tools history
ara tool calculate "(17 + 5) * 3"
ara skill list
ara skill show "procedure-name"
ara backup
```

Ara resumes the last conversation after restarting; use `/new` to start a separate thread. `ara conversations` lists threads and `ara --conversation-id ID` resumes one. Within a conversation, `/help`, `/history`, `/memory`, `/skills`, `/calc 2 + 2`, `/remember a useful detail`, and `/exit` are available. Use `ara tool read-file PATH` and `ara tool list-dir PATH` only for paths under configured `tools.allowed_roots`. The default allowed root is the directory from which Ara first creates its configuration; review it in `config.toml`. File writes create new files only and require an exact interactive confirmation or the explicit `--yes` flag for that one command. No tool executes arbitrary shell commands.

Backups are SQLite database snapshots stored under `~/.local/share/ara/backups/` by default. `ara memory export [PATH]` exports active memories to JSON. Restore is destructive and requires the user to type `restore` interactively. Keep backups and exports private: they can contain conversation history and personal details.

## How memory, learning, and identity work

Conversations are stored locally by conversation ID and the active thread resumes across restarts. Each request includes a bounded number of recent messages. Credential-assignment patterns are redacted before conversation messages reach SQLite or the model prompt. Tool audit logs omit file contents, and known credential-assignment patterns are redacted there too. Ara retrieves a small set of matching active memories using lexical overlap and confidence, rather than adding the full database to prompts. The current learning rules accept explicit “remember”, preference, name, and correction statements; each entry has a source and confidence, and related overlapping notes are flagged for user review. Passwords, API keys, access tokens, and private keys are rejected by ordinary memory storage. This is a modest rule-based learning pipeline, not fine-tuning. `ara memory inspect`, `correct`, and `forget` let the user audit it.

Identity fields are persistent values, not an emotion simulator. The `ara identity set KEY VALUE` command changes a field used in the next prompt. Goals are user-visible records. The decision command selects from options using scores supplied by the caller and stores its input and rationale; it does not claim independent agency.

Skills are versioned, inspectable procedure notes (`ara skill show NAME`). Ara stores them but does not execute their steps automatically. Users should review a procedure before using it. Recorded decisions and sanitized tool results can be inspected with `ara decisions` and `ara tools history`.

## Architecture

`ara/config.py` loads local TOML settings and environment overrides. `ara/store.py` owns the SQLite schema and persistence. `ara/engine.py` composes bounded context and conservative learning. `ara/model.py` adapts a local chat-completions endpoint. `ara/tools.py` provides restricted file operations and a safe arithmetic evaluator. `ara/cli.py` contains the terminal interface and maintenance commands.

## Tests

Run the offline test suite without a paid API or model weights:

```sh
python -m unittest discover -s tests -v
```

The tests use a clearly identified fake model to isolate conversation orchestration and a local HTTP stub to check the request/response format. Neither demonstrates real model inference. For a real local integration check, start `llama-server` with compatible Qwen3 GGUF weights, run `ara model status`, then start `ara` and exchange a few turns. Ara does not bundle weights, so that integration check cannot run in a clean checkout.

## Limitations

Ara’s decisions are deterministic helpers, not autonomous reasoning. Learning is explicit and rule-based; memory retrieval uses simple lexical matching and can miss paraphrases or surface related-but-not-relevant notes. Skills are stored but not automatically executed. Tool paths are limited to configured roots, though operating-system file permissions still apply. No unrestricted command runner, background agent, model training, remote integration, telemetry, or automatic model download is included. The local inference server is a separate program that users install and operate.

## Contributing and licence

Run the tests before submitting changes. Keep conversation databases, model weights, secrets, and personal configuration out of commits. Ara’s source code is under Apache-2.0; third-party model files keep their own licences.

## Developer

Ara is made by Ayodele Ayokunle David, a teenage hobby programmer from Nigeria, who is interested in building independent software systems. [Portfolio](https://ayo-portal.netlify.app).

“Build things that matter.”
