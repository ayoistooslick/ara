# Models, adapters, and hardware

Ara separates its identity and behaviour from the underlying language model. The model is a replaceable component reached through `ara.model.ModelAdapter`; the current production adapter is an OpenAI-compatible HTTP client.

## What Ara does and does not do

Ara:

- sends prompts to an explicitly configured inference server;
- discovers whether that server is reachable and which model IDs it serves;
- fits messages into a character budget and enforces `max_tokens`, a request timeout, and cooperative streaming cancellation;
- refuses non-loopback endpoints unless `model.allow_remote = true`;
- reports the configured weights file's size when `model.path` is set.

Ara does not:

- load weights, run a tokenizer, or perform inference in-process;
- download models automatically;
- assume a text endpoint can process images or audio;
- expose cloud inference by default.

`model.path` is used for diagnostics and fit warnings. The server owns the actual weights. If the path and the server's loaded model disagree, the server wins.

## Configuration

`config.toml` in `$ARA_HOME` (default `~/.local/share/ara`):

```toml
[model]
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

Environment overrides: `ARA_MODEL_URL`, `ARA_MODEL_NAME`, `ARA_MODEL_PATH`, `ARA_MAX_TOKENS`, `ARA_TEMPERATURE`, `ARA_MODEL_PROFILE`, `ARA_EMBEDDING_URL`, `ARA_EMBEDDING_MODEL`, `ARA_VISION_URL`, `ARA_VISION_MODEL`, `ARA_AUDIO_URL`, `ARA_AUDIO_MODEL`, `ARA_HOME`.

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

Ara's test suite uses a local stub server to exercise request encoding, streaming, malformed responses, timeouts, and unreachable servers. Those tests do **not** establish answer quality, African-language fluency, or speed on real weights.

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

No model is Ara's permanent identity. The default configuration names Qwen because Qwen3 GGUF models are a practical local starting point, but nothing in the engine requires Qwen. A model card, licence, and quantization should travel with any weights you distribute.

The training baseline in `ara/training` is separate from the inference adapter. It is an n-gram teaching model, not a production chat model. See [TRAINING.md](TRAINING.md).
