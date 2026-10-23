# Web Ara

The web interface is Àrà's primary experience: plain `ara` starts it, and users open a browser — no install, no weight downloads.

It is a browser interface over the same `ConversationEngine`, `Store`, `KnowledgeBase`, and model adapter as the secondary terminal interface (`ara chat`). It is not a separate chatbot and has no separate memory store.

```sh
ara              # same as 'ara web'
# http://127.0.0.1:8787
```

Options: `ara web --host 127.0.0.1 --port 8787`. Configuration also lives under `[web]` in `config.toml` (`host`, `port`, `session_hours`, `rate_limit`). Managed containers (e.g. a sandboxed preview that owns the port) can bind `0.0.0.0` without a password by setting `ARA_WEB_INSECURE_BIND=1`; do that only where the platform isolates the port from untrusted networks.

## First run and sessions

On first load the client calls `POST /api/session`. On loopback, no password is required. The server creates a random 32-byte token, stores only its SHA-256 hash, and returns an HttpOnly, SameSite=Strict cookie plus a CSRF token. The CSRF token is required in `X-Ara-CSRF` for state-changing requests; same-origin checks are also enforced.

- Sessions expire after `web.session_hours` (default 72).
- Each session has its own conversation ID. Memory is shared across sessions and with the local CLI on the same installation.
- Logging out deletes the session row and clears the cookie.
- Session attempts are limited per client IP. Chat has a per-session rate limit.
- These limits are in-memory and reset when the server restarts. They are not a substitute for a reverse-proxy rate limit.

To expose Ara beyond loopback, set `ARA_WEB_PASSWORD` and use HTTPS. The server refuses a non-loopback bind without a password unless `ARA_WEB_INSECURE_BIND=1` is set for an isolated container. A `Secure` cookie is set when the request arrives with `X-Forwarded-Proto: https`.

```sh
ARA_WEB_PASSWORD='a long random password' ara web --host 127.0.0.1 --port 8787
```

Keep the password in the service environment, not in `config.toml` or client code.

## What the interface does

- **Chat:** streaming Server-Sent Events with a visible typing state, stop control, and Markdown-free safe text rendering (`textContent`, never `innerHTML`). If no model answers, the reply is still useful but begins with `[retrieval-only]` and the UI marks it as retrieved context, not generated text.
- **Study mode:** sends `mode: "learn"` to the same engine.
- **History:** loads the session's conversation; "New chat" starts a new thread without deleting the old one.
- **Memory:** add, correct, delete, forget all, and export the shared memories. Credential-looking content is rejected.
- **Nigerian context:** query an expression or sentence and see register, meanings, regions, caveats, and ambiguity.
- **News:** cached stories, refresh when online, offline label, source and publication time.
- **Image upload:** validates type/dimensions/size, sends to the configured vision endpoint, shows a result card. The file is not stored.
- **Camera:** explicit button, live preview inside a dialog, one frame captured only when **Analyse** is pressed.
- **Microphone:** explicit button, recording indicator and timer, 60-second cap, stop control, transcription into the composer for review before sending.
- **Speech output:** optional browser speech synthesis; can be stopped independently.
- **Diagnostics:** active backend (hosted, local, or retrieval-only), model reachability, configured endpoints, counts, and retrieval mode (`semantic: true/false`).

## Browser permissions

Camera and microphone use `navigator.mediaDevices.getUserMedia`. Permission is requested only after an explicit click. No sensor starts on page load.

Requirements and failure behaviour:

| Condition | Behaviour |
| --- | --- |
| Insecure origin (plain HTTP, not localhost) | Browser blocks `getUserMedia`; Ara explains and keeps text working |
| Permission denied | Clear message; text chat and uploads remain available |
| No device | Clear "not found" message; text remains |
| Unsupported browser | Controls report unavailability; text remains |
| Tab hidden or page unload | Recording and camera stop automatically; speech is cancelled |
| Recording reaches 60 seconds | Stops automatically and transcribes what was captured |
| Transcription/vision endpoint not configured | HTTP 503 with an explicit message; no fake result |
| Endpoint fails | Error shown; the recording or image is not silently retried elsewhere |

A visible strip lists every active sensor (camera, microphone, speech). The single control releases all active captures immediately. Camera frames are not analysed continuously; only the explicit **Analyse** action captures one JPEG frame at a bounded size.

Ara never claims browser permission grants access to the filesystem, phone sensors outside the browser, or background recording. There is no service worker recording audio, no hidden stream, and no upload without an explicit action.

## Audio and vision pipeline

```text
Microphone → MediaRecorder → data URL → POST /api/transcribe
          → configured local OpenAI-compatible /audio/transcriptions
          → text placed in the composer (not auto-sent)

Camera → canvas JPEG → POST /api/vision
       → configured local OpenAI-compatible multimodal chat
       → text result card (no image stored)

Reply text → speechSynthesis → device voice
```

The browser does not use `SpeechRecognition`. Speech-to-text is the configured server endpoint, so its language support and accuracy depend on that model. Browser speech synthesis voices vary by OS; Nigerian English quality is not measured here.

Uploads are validated by magic bytes, content type, pixel count, and size (8 MB images, 15 MB audio, 40 megapixels) and processed in memory. They are not written to SQLite.

## Security model

- No credentials in client bundles. The web password is an environment variable on the server.
- CSP: `default-src 'self'`, no inline scripts/styles, no object/frame embedding. Static assets are served from a fixed directory with traversal checks.
- `X-Frame-Options: DENY`, `nosniff`, `no-referrer`, `Permissions-Policy` limited to camera/microphone on self.
- CSRF token plus Origin/Host checks on mutating requests.
- Body size limits are enforced from `Content-Length` before buffering; messages are capped at 8,000 characters.
- Sessions are stored hashed; cookies are HttpOnly and SameSite=Strict.
- Per-session chat rate limit and per-IP login throttle.
- No shell tool, arbitrary URL fetch, filesystem browser, or admin endpoint is exposed to web users.
- Conversation access is scoped to the session's conversation ID. Memory is intentionally shared on one installation, not per anonymous visitor.
- The model, vision, and audio endpoints are server-side; the browser never calls them directly.

Limitations: this is a single-password installation, not a multi-tenant account system. The Python `http.server` is development-grade; use a reverse proxy with TLS, request limits, and process supervision. There is no built-in TLS. Do not expose port 8787 directly to the internet.

## Accessibility and small screens

- Semantic landmarks, a skip link, labelled controls, visible focus styles, and `aria-live` for chat status.
- Keyboard-operable controls; the camera dialog traps focus through the native `<dialog>` and restores it on close.
- Responsive single-column layouts for narrow Android screens; the composer stays reachable.
- `prefers-reduced-motion` disables the hopping animation and transitions.
- `prefers-color-scheme` is not required for use; the palette uses paper, ink, and green tokens with strong value contrast. Re-check with a contrast tool after changing theme tokens.
- Errors are text, not colour alone. Sensor state is named in text as well as shown visually.

## Deployment

See [`deploy/README.md`](../deploy/README.md) for the systemd + Caddy example. The short version:

1. Run `ara web --host 127.0.0.1` as a dedicated user.
2. Set `ARA_WEB_PASSWORD` in a 0600 environment file.
3. Put Caddy or nginx in front with TLS and SSE buffering disabled.
4. Keep model/vision/audio services on loopback or a trusted private network.
5. Back up `/var/lib/ara/ara.sqlite3` and test restore.

Health check: `GET /api/health` returns `ok`, version, database status, and model reachability. It does not require a session.

## Verification status

Automated tests cover:

- session creation, password gate, Secure cookie with forwarded HTTPS, logout, CSRF, origin checks, rate limits;
- SSE streaming and persistence, session isolation, model-unreachable retrieval fallback;
- upload validation and 503s when perception is unconfigured;
- vision and transcription request contracts against local stub servers;
- static traversal, security headers, unknown routes, body limits;
- shared memory controls and export;
- news offline/refresh and Nigerian context.

Not verified: a physical phone camera or microphone, real permission dialogs, actual audio capture quality, synthesized speech playback, real vision accuracy, and a public TLS deployment. Use a browser on the target device and check the visible sensor strip, a captured image result, transcribed text, and spoken output before claiming those work.
