# Deployment

This is a Linux deployment example for a small VPS or home server. The web server stays on loopback; Caddy terminates TLS and proxies to it.

**Status:** these files have not been deployed from this workspace. Run every check below on the target host and record what actually happened. Do not present an untested deployment as verified.

## Requirements

- A Linux host with Python 3.10+, systemd, and a public DNS name.
- One inference backend: a hosted ara-ng endpoint (`ARA_HOSTED_URL`, HTTPS, optional `ARA_HOSTED_API_KEY`) or a local OpenAI-compatible server reachable at `127.0.0.1:8080` (or a private address with `allow_remote = true`). Without either, chat still works but every reply is labelled retrieval-only.
- Caddy 2 for TLS.
- Enough RAM/disk for the model and the SQLite database. The model and Ara do not have to run as the same user.

A public deployment must not depend on the developer's phone. Run Ara on a host that stays online, or accept that it is unavailable when the host is off.

## Install

```sh
sudo useradd --system --home /var/lib/ara --shell /usr/sbin/nologin ara
sudo mkdir -p /opt/ara /var/lib/ara /etc/ara
sudo chown -R ara:ara /opt/ara /var/lib/ara
sudo chmod 700 /var/lib/ara

cd /opt/ara
sudo -u ara git clone https://github.com/ayoistooslick/ara.git .
sudo -u ara python3 -m venv .venv
sudo -u ara .venv/bin/python -m pip install --upgrade pip
sudo -u ara .venv/bin/python -m pip install .
```

Copy and edit the examples:

```sh
sudo cp deploy/ara.service /etc/systemd/system/ara.service
sudo cp deploy/ara.env.example /etc/ara/ara.env
sudo chmod 600 /etc/ara/ara.env
sudoedit /etc/ara/ara.env
```

Set a long random `ARA_WEB_PASSWORD`; keep `/etc/ara/ara.env` unreadable by other users. Set `ARA_HOME=/var/lib/ara`.

Run a loopback check before enabling the service:

```sh
sudo -u ara env ARA_HOME=/var/lib/ara ARA_WEB_PASSWORD=test-only \
  /opt/ara/.venv/bin/ara web --host 127.0.0.1 --port 8787
```

In another shell:

```sh
curl -fsS http://127.0.0.1:8787/api/health | head -c 500
curl -fsS http://127.0.0.1:8787/frog.svg | head -c 120
```

Stop it with Ctrl+C. Do not leave a test password in production.

## TLS and systemd

```sh
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile
sudoedit /etc/caddy/Caddyfile     # replace ara.example.ng
sudo systemctl daemon-reload
sudo systemctl enable --now ara
sudo systemctl reload caddy
```

Caddy obtains a certificate automatically when DNS points at the host and ports 80/443 are open.

## Health and auth checks

```sh
curl -fsS https://ara.example.ng/api/health
```

Expected: JSON with `"ok": true`, the version, and `"database": "ok"`. `model_reachable` is `false` if the model server is down; that is reported, not hidden.

Test the password gate without putting the password in shell history:

```sh
read -rsp "Ara web password: " ARA_PASS; echo
curl -fsS -c /tmp/ara-cookies -X POST https://ara.example.ng/api/session \
  -H 'Content-Type: application/json' -d "{\"password\":\"$ARA_PASS\"}"
unset ARA_PASS
```

Expected: `authenticated: true`, a CSRF token, and a `Set-Cookie` with `HttpOnly`, `SameSite=Strict`, and `Secure` when `X-Forwarded-Proto: https` reaches Ara (Caddy sets it).

Check that the database and backups are owned by `ara` and not world-readable:

```sh
sudo ls -l /var/lib/ara /var/lib/ara/backups
sudo -u ara /opt/ara/.venv/bin/ara status
```

The web password is read from the environment, not `config.toml`. The service refuses to bind a non-loopback address without it.

## Backups

```sh
sudo -u ara env ARA_HOME=/var/lib/ara /opt/ara/.venv/bin/ara backup
```

SQLite backups include conversations and memories. Store them encrypted and access-controlled. Test restore on a copy before relying on it. `ara restore` writes a pre-restore backup automatically.

## Model services

Run the model server under its own service or container, bound to loopback. Configure the model URL and name in `/etc/ara/ara.env`. Vision and transcription are separate services; leave them unset if unavailable so the web UI shows clear 503 errors instead of fake results.

Do not set `model.allow_remote = true` unless the remote endpoint is under your control and the privacy trade-off is intentional. A hosted ara-ng endpoint is configured separately through `ARA_HOSTED_URL` and is used only when set; keep its key in `/etc/ara/ara.env`, never in `config.toml`.

## What this setup does not provide

- Horizontal scaling or a shared database. Ara is a single-installation companion.
- Multi-user accounts. One password protects one installation; web sessions are separate conversation threads, not separate private accounts.
- Automatic OS hardening, firewall rules, or certificate monitoring.
- A guarantee that a low-memory VPS can run a large model. Check `ara doctor` and the profiles in `docs/MODELS.md`.

For a public service, use a reverse proxy with request limits, keep dependencies updated, and review the security notes in `docs/WEB.md`.
