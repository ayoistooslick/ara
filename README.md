# Ara

Ara is a local-first terminal companion created by Ayodele Ayokunle David. It keeps conversation history, explicitly signalled memories, identity settings, goals, and documented procedures in a local SQLite database. The language model is replaceable: Ara talks to any local OpenAI-compatible endpoint, such as `llama-server` from llama.cpp. Ara does not download weights, call a cloud AI service, or upload conversations on its own.

The project exists to explore what a small, inspectable companion system can do around a pretrained model. Persistent notes and procedures are useful, but they are not model training or evidence of sentience.

## What works

Ara runs multi-turn conversations when a compatible local model server is running. It captures explicitly signalled memories and corrections, retrieves relevant notes using lexical matching, keeps durable identity fields, tracks goals, records transparent score-based decisions, and can execute documented reusable procedures (skills) step by step within permission limits. The calculator and file tools are restricted to configured roots. Backups are consistent SQLite snapshots.

Memory learning is deliberately conservative: Ara saves clearly signalled statements such as "I prefer ..." and "Remember that ..."; it does not treat every sentence as fact. Tools do not run merely because a generated response asks for them. There is no default shell-execution tool.

## Requirements and installation

Ara needs Python 3.11 or newer at runtime, though the `tomllib` module on older Python is handled by installing `tomli`. No GPU is required for Ara itself. Model inference is performed by a separate local runtime and can need substantial RAM, storage, and patience. An 8B model may not fit an Android handset; smaller Qwen3 models (0.6B, 1.7B) are options for limited hardware. `ara doctor` reports available system memory and the configured GGUF file size when a path is set, but cannot guarantee that a model will load.

### Linux

```sh
git clone https://github.com/ayoistooslick/ara.git
cd ara
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
ara --help
```

If Python is older than 3.11, also install `tomli`:

```sh
python -m pip install tomli
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

The first `ara` start creates a local configuration file at `~/.local/share/ara/config.toml` (or under `$ARA_HOME`). Ara does not fetch model files automatically. Downloading is a separate, user-initiated action.

### Model options

Qwen3 models are the initial recommendation. Qwen publishes GGUF files for several sizes:

- **Qwen3-8B** — Q4_K_M is about 4.8 GB. Needs roughly 6-8 GB available RAM for comfortable CPU inference.
- **Qwen3-1.7B** — Q8_0 is about 1.75 GB. Usable on machines with 2-4 GB available.
- **Qwen3-0.6B** — Q8_0 is about 610 MB. Fits on very limited hardware but is noticeably less capable.

Smaller models use less memory but produce less capable responses. Pick the largest model your hardware can handle. The model backend is replaceable; Ara does not care which pretrained model you run.

Qwen's model card lists Qwen3-8B and identifies Apache-2.0 licensing. Qwen's local-runtime guide documents official GGUF files and llama.cpp support. Official GGUF repositories are at `Qwen/Qwen3-8B-GGUF`, `Qwen/Qwen3-1.7B-GGUF`, and `Qwen/Qwen3-0.6B-GGUF` on Hugging Face.

### Running a local model server

The easiest route is llama.cpp's `llama-server`, which speaks the OpenAI chat-completions protocol that Ara expects. A prebuilt Linux binary can be downloaded from the llama.cpp GitHub releases page. For example, using a release archive:

```sh
# Download a llama.cpp release binary (check the latest release on github.com/ggml-org/llama.cpp)
tar -xzf llama-bin-ubuntu-x64.tar.gz
cd llama-bin-*
./llama-server -m /path/to/model.gguf --host 127.0.0.1 --port 8080 -c 2048
```

Qwen3 models include a thinking/reasoning mode. When using Qwen3 models through llama-server, make sure `max_tokens` is large enough to accommodate both the thinking tokens and the actual reply. 512 or more is a reasonable starting point for the full Ara system prompt.

Ara's default endpoint is `http://127.0.0.1:8080/v1`. Verify it with `ara model status`, then run `ara`.

Model weights are separate from Ara and are not checked into this repository. Keep the model publisher's licence with any copy you download. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

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
ara identity set temperament "direct and concise"
ara goal add "Finish the project" --priority 4
ara goal list
ara goal status 1 done
ara decide --context "which tool" --option A --option B --score A=3 --score B=1
ara decisions
ara tools list
ara tools history
ara tool calculate "(17 + 5) * 3"
ara tool read-file PATH
ara tool list-dir PATH
ara tool write-file PATH "content" --yes
ara skill list
ara skill show "procedure-name"
ara skill add "name" "description" '[{"action":"calculate","expression":"1+1"}]'
ara skill run "name" --yes
ara backup
ara autonomous
```

Ara resumes the last conversation after restarting; use `/new` to start a separate thread. `ara conversations` lists threads and `ara --conversation-id ID` resumes one. Within a conversation, `/help`, `/history`, `/memory`, `/skills`, `/calc 2 + 2`, `/remember a useful detail`, and `/exit` are available.

Use `ara tool read-file PATH` and `ara tool list-dir PATH` only for paths under configured `tools.allowed_roots`. The default allowed root is the directory from which Ara first creates its configuration; review it in `config.toml`. File writes create new files only and require an exact interactive confirmation or the explicit `--yes` flag for that one command. No tool executes arbitrary shell commands.

`ara skill run NAME --yes` executes a documented procedure step by step. Each step must map to a known Ara action (read a file, list a directory, write a file, calculate). Steps can declare an `on_failure` policy of `stop` (default) or `continue`. Skills do not receive any permissions beyond what the execution already requires; run them with `--yes` or interactive confirmation.

`ara autonomous` starts a bounded goal-directed loop. Ara reads your active goals, proposes steps, and asks before running anything that modifies files. It stops when the goal is done, fails, or needs more information. This is not an open-ended agent; it is an inspectable loop that waits for your input.

Backups are SQLite database snapshots stored under `~/.local/share/ara/backups/` by default. `ara memory export [PATH]` exports active memories to JSON. Restore is destructive and requires the user to type `restore` interactively. Keep backups and exports private: they can contain conversation history and personal details.

## How memory, learning, and identity work

Conversations are stored locally by conversation ID and the active thread resumes across restarts. Each request includes a bounded number of recent messages. Credential-assignment patterns are redacted before conversation messages reach SQLite or the model prompt. Tool audit logs omit file contents, and known credential-assignment patterns are redacted there too.

Ara retrieves a small set of matching active memories using lexical overlap and confidence, rather than adding the full database to prompts. Recent corrections are surfaced separately so they can override older notes where they overlap.

The current learning rules accept explicit "remember", preference, name, and correction statements. Each entry has a source and confidence, and related overlapping notes are flagged for user review. Passwords, API keys, access tokens, and private keys are rejected by ordinary memory storage. Corrections are stored and included in later system prompts so the model can take them into account; they do not automatically delete or overwrite the original note, so the user can inspect the full picture with `ara memory inspect`.

This is a modest rule-based learning pipeline, not fine-tuning. `ara memory inspect`, `correct`, and `forget` let the user audit it.

Identity fields are persistent values, not an emotion simulator. The `ara identity set KEY VALUE` command changes a field used in the next prompt. Useful fields include `temperament`, `values`, and `communication`. Defaults are provided so Ara speaks sensibly before you customise it.

Goals are user-visible records that survive restarts. `ara goal add` creates one, `ara goal list` shows active goals, and `ara goal status ID done` marks one complete. Goals are context for conversation and the autonomous loop; they do not trigger automatic action by themselves.

The decision command (`ara decide`) selects from options using scores supplied by the caller and stores its input and rationale. It is a transparent helper, not an autonomous reasoner.

Skills are versioned, inspectable procedure notes. `ara skill add` saves a procedure as JSON steps, `ara skill show` displays it, `ara skill list` shows all versions, and `ara skill run` executes it with approval. Each skill step declares an action that maps to a known tool. Skills are stored but not executed automatically; the user runs them explicitly.

## Architecture

`ara/config.py` loads local TOML settings and environment overrides. `ara/store.py` owns the SQLite schema and persistence. `ara/engine.py` composes bounded context, conservative learning, corrections, and a bounded action loop. `ara/model.py` defines the model adapter interface and the local OpenAI-compatible client; the model backend is replaceable through that interface. `ara/tools.py` provides restricted file operations, a safe arithmetic evaluator, and skill execution. `ara/cli.py` contains the terminal interface and maintenance commands. `ara/safety.py` handles credential redaction.

## Tests

Run the offline test suite without a paid API or model weights:

```sh
python -m unittest discover -s tests -v
```

The suite includes unit tests for configuration, memory, identity, goals, decisions, tools, skills, backup/restore, error handling, conversation continuity, and permission boundaries. It uses a clearly identified fake model to isolate conversation orchestration and a local HTTP stub to check the request/response format. Neither demonstrates real model inference.

For a real local integration check, start `llama-server` with compatible GGUF weights, run `ara model status`, then start `ara` and exchange a few turns. Ara does not bundle weights, so that integration check cannot run in a clean checkout. An inference smoke test sends a real prompt to the configured model, receives a response, and validates the response format; it requires actual model weights and a running server.

## Known limitations

- Real model inference needs a separate local server and model weights. The integration cannot run without them.
- Qwen3 models use a thinking mode that consumes part of the token budget. Set `max_tokens` high enough for both thinking and reply.
- Memory retrieval uses simple lexical matching and can miss paraphrases or surface related-but-not-relevant notes.
- Corrections are included in prompts but do not automatically overwrite original notes. The user inspects and edits memories directly.
- Skills can only reuse existing Ara actions. They cannot invent new tools or run arbitrary shell commands.
- The autonomous loop is bounded and interactive. It does not run in the background or make independent decisions without user input.
- Tool paths are limited to configured roots, though operating-system file permissions still apply.
- No unrestricted command runner, background agent, model training, remote integration, telemetry, or automatic model download is included.
- The smaller Qwen3 models (0.6B, 1.7B) are usable on limited hardware but are less capable than 8B models.
- Android/Termux testing on a physical device has not been performed in this environment. The code uses only the Python standard library and standard POSIX interfaces, which Termux supports, but device-level verification remains outstanding.

## Contributing and licence

Run the tests before submitting changes. Keep conversation databases, model weights, secrets, and personal configuration out of commits. Ara's source code is under Apache-2.0; third-party model files keep their own licences.

## Developer

Ara is made by Ayodele Ayokunle David, a teenage hobby programmer from Nigeria, who is interested in building independent software systems. [Portfolio](https://ayo-portal.netlify.app).

"Build things that matter."
