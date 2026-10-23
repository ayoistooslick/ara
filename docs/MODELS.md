# Models, adapters, and hardware

**ara-ng is Àrà's own model** — trained by Àrà's builder on his own training data. It is the product identity: the web app, the model selector, and every user-facing label say `ara-ng`. It is reached through a hosted OpenAI-compatible endpoint (`ARA_HOSTED_URL`, model name `ara-ng`, key in `ARA_HOSTED_API_KEY`). The weights and training data are not in this repository.

Under the product identity, the engine treats the model as a replaceable component reached through `ara.model.ModelAdapter`; the adapters are OpenAI-compatible HTTP clients for the hosted ara-ng endpoint or a local fallback server. Third-party model names (e.g. the Qwen GGUF catalogue) appear in this repository only as optional local fallbacks and their legally required licence information — never as Àrà's identity.

## What Ara does and does not do

Ara:

- picks the first configured backend: hosted ara-ng, then a local server, then retrieval-only;
- discovers whether that server is reachable and which model IDs it serves;
- fits messages into a character budget and enforces `max_tokens`, a request timeout, and cooperative streaming cancellation;
- refuses non-loopback local endpoints unless `model.allow_remote = true`;
- requires HTTPS for a hosted endpoint (HTTP is allowed only on loopback, for a local test server);
- reports the configured weights file's size when `model.path` is set;
- downloads optional weights only when you run `ara model download`, with size, disk, RAM, licence, resumable transfer, and a checksum check.

Ara does not:

- load weights, run a tokenizer, or perform inference in-process;
- download anything during install or a normal chat start;
- assume a text endpoint can process images or audio;
- silently enable cloud inference.

`model.path` is used for diagnostics and fit warnings. The server owns the actual weights. If the path and the server's loaded model disagree, the server wins.

## Configuration

`config.toml` in `$ARA_HOME` (default `~/.local/share/ara`):

```toml
[model]
# auto = hosted when hosted_url is set, else the local server.
backend = "auto"          # auto | hosted | local | retrieval
# hosted_url = "https://your-ara-ng-endpoint.example/v1"
hosted_model = "ara-ng"
url = "http://127.0.0.1:8080/v1"
name = "the-model-id-your-server-serves"
# Optional: used only for size diagnostics.
# path = "/path/to/model.gguf"
allow_remote = false
profile = "auto"          # auto | constrained | standard | powerful
max_tokens = 512
temperature = 0.7

[runtime]
context_messages = 12
max_context_chars = 12000
request_timeout = 120
```

Environment overrides: `ARA_BACKEND`, `ARA_HOSTED_URL`, `ARA_HOSTED_API_KEY`, `ARA_HOSTED_MODEL`, `ARA_MODEL_URL`, `ARA_MODEL_NAME`, `ARA_MODEL_PATH`, `ARA_MAX_TOKENS`, `ARA_TEMPERATURE`, `ARA_MODEL_PROFILE`, `ARA_EMBEDDING_URL`, `ARA_EMBEDDING_MODEL`, `ARA_VISION_URL`, `ARA_VISION_MODEL`, `ARA_AUDIO_URL`, `ARA_AUDIO_MODEL`, `ARA_HOME`.

The hosted API key is deliberately environment-only. It is never written to `config.toml`, a backup, or a log line.

## Hosted ara-ng

```sh
ARA_HOSTED_URL="https://your-endpoint.example/v1" \
ARA_HOSTED_API_KEY="…" \
ara ask "How far?"
```

With `backend = "auto"`, a configured `hosted_url` wins over the local server. Force a backend with `ARA_BACKEND=hosted`, `local`, or `retrieval`. Hosted requests use the OpenAI chat-completions shape with `model: "ara-ng"`, a 60-second ceiling, and the same streaming parser as the local adapter. HTTP 401/403/429 produce short, non-secret error messages. No hosted service ships with this repository, so that path is tested against a local protocol stub, not a live ara-ng deployment.

The model `name` is sent as the API `model` field. Many single-model servers ignore it or expect a specific alias; check `ara model status` and the server's `/v1/models` response.

## Capability discovery

`ara model status` and `ara model` call:

- `GET /v1/models` for reachability and served model IDs;
- `GET /props` on the server base (llama.cpp) for `n_ctx` when available.

The resulting `context_length` is reported, not enforced as an exact tokenizer limit. Prompt fitting uses `max_context_chars`; a character budget is approximate and deliberately conservative for mixed English, Pidgin, and Nigerian-language text.

Streaming is requested with `stream: true`. A server that returns ordinary JSON still works: the adapter parses the full response and yields it once. Cancellation is checked between chunks. A request timeout applies to the HTTP call, not to each token.

## Local server example

Any server implementing OpenAI chat completions should work. This is the tested protocol shape:

```sh
llama-server -m ./Qwen3-4B-Q4_K_M.gguf --host 127.0.0.1 --port 8080 -c 4096
ara model status
ara ask "Explain a Nigerian mortgage in simple English."
```

## Optional local downloads

```sh
ara model download qwen3-0.6b     # 610 MB, about 1.5 GB free RAM
ara model download qwen3-1.7b     # about 1.1 GB, about 2 GB free RAM
ara model download qwen3-4b       # about 2.3 GB, about 4 GB free RAM
```

The catalogue URLs were probed against Hugging Face while writing this: 639,446,688 bytes (0.6B Q8_0), 1,107,409,472 (1.7B Q4_K_M, unsloth), and 2,497,280,256 (4B Q4_K_M). The 1.7B files are not published in Qwen's own GGUF repo, which is why that entry points at the unsloth quantisation.

Ara prints the size, required free storage, RAM guidance, and licence before asking. It resumes an interrupted `.part` file with a Range request, hashes as it writes, verifies against the publisher's SHA-256 (`x-linked-etag`) when available, and renames the file only after verification. A lock file prevents two downloads of the same target; a bad checksum deletes the partial file. Use `--out` and `--sha256` to supply your own URL and expected digest. Weights land in `$ARA_HOME/models` and are never committed.

On this sandbox, a real Qwen3-0.6B Q8_0 server answered NEPA questions with a first token after 23.5–37.2 s and a full answer after 32–44 s at 2,048 context on 2 vCPU, using about 1.27 GiB of RAM. This is a measurement of one small model, not a general speed or quality claim.

Ara's test suite uses local stub servers to exercise request encoding, streaming, malformed responses, timeouts, unreachable servers, hosted auth, and downloads. Those tests do **not** establish answer quality, African-language fluency, or speed on larger weights.

## Hardware profiles

`profile` is a starting suggestion based on currently available memory. It is not a benchmark and does not measure CPU, thermal throttling, or token throughput.

| Profile | Free memory guide | Suggested GGUF starting point | Budget |
| --- | --- | --- | --- |
| constrained | about 2–4 GB | Qwen3-0.6B Q8_0 (~0.6 GB) or Qwen3-1.7B Q4_K_M (~1.1 GB) | 256 tokens, 6 messages |
| standard | about 4–10 GB | Qwen3-4B Q4_K_M (~2.5 GB) or Qwen3-8B Q4_K_M (~4.7 GB) | 512 tokens, 12 messages |
| powerful | about 10 GB+ or a GPU | Qwen3-8B Q5_K_M or a compatible 14B 4-bit model | 1024 tokens, 20 messages |

The weights file is not the whole memory cost. The KV cache, runtime overhead, and operating system need additional room. An 8B Q4 model on a 4 GB phone is not a supported expectation. `ara doctor` prints free memory and warns when the configured file alone is close to it.

If a model does not fit:

1. Choose a smaller parameter count or stronger quantization.
2. Lower `max_tokens`, `context_messages`, and the server's `-c` context.
3. Use a remote machine on your own network with `allow_remote = true` only if you accept that boundary.
4. Use a cloud endpoint only by explicitly configuring it; never assume it is private.

## Embeddings

Hybrid memory needs an embedding endpoint:

```toml
[embeddings]
url = "http://127.0.0.1:8081/v1"
model = "nomic-embed-text"
```

`ara memory embed` computes missing vectors and prints `"semantic": true`. The endpoint must support OpenAI `POST /embeddings` with `{"model": ..., "input": ...}` and return `data[0].embedding`.

Without an endpoint, Ara uses a deterministic 384-dimensional hashed-vector fallback. It improves recall for related word forms but is **not** semantic understanding. `ara memory embed` prints `"semantic": false`; never describe that mode as semantic search.

## Vision

Vision is a separate endpoint, configured under `[vision]`. It accepts an OpenAI-compatible chat completion with an image data URL. The server must actually run a vision-capable model. `ara vision IMAGE` and the web camera use the same client.

If no endpoint is configured, both interfaces return an explicit unavailable error. They do not attach the image to a text model and call the result understanding. Images are validated (PNG, JPEG, GIF, or WebP magic bytes, dimensions, size), sent in memory, and not stored. SVG is rejected.

## Audio

Transcription is a separate endpoint under `[audio]`, supporting `POST /audio/transcriptions` as multipart form data. `ara audio FILE` and the browser recorder use it.

Speech synthesis uses the browser's installed `speechSynthesis` voices in Web Ara. There is no server-side TTS. Voice availability and Nigerian English accent quality vary by device and are not measured by this project.

Browser speech recognition is not used. Recording is the input path; the server endpoint is the recognition path.

## Identity and provenance

ara-ng is Àrà's model and its only user-facing model name. Nothing in the engine requires any particular architecture: `hosted_model` defaults to `ara-ng` and is what the UI shows when the hosted backend is active.

The Qwen references that remain (the `ara model download` catalogue, the local-profile suggestions) describe **optional local fallback weights** published by third parties. They carry their own Apache-2.0 licences and notices (see `THIRD_PARTY_NOTICES.md`); they are not presented as Àrà's model, and Àrà claims no authorship over them. A model card, licence, and quantization should travel with any weights you distribute.

The training baseline in `ara/training` is separate from the inference adapter. It is an n-gram teaching model, not a production chat model. See [TRAINING.md](TRAINING.md).
