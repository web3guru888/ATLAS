# Cloudflare Tunnel Setup for atlas-api (atlas.thebeastagi.com)

> Prepared by beast-atlas, 2026-07-06 (OpenRouter Track 1).
> Status: cloudflared **installed** (v2026.6.1, `/usr/local/bin/cloudflared`), config skeleton **written**.
> **Robin action required**: steps 1–3 below (browser login + DNS). ~5 minutes.

## Why
- The A100's public IP is ephemeral (changes on GCP stop/start). A named tunnel gives `atlas.thebeastagi.com` a stable TLS hostname regardless of IP.
- Port 8080 is currently reachable from the internet **unauthenticated** — we observed random scanner traffic (23.236.x.x) submitting requests. Once the tunnel is live, the GCP firewall rule for :8080 should be REMOVED so only tunnel traffic (with the API key) reaches atlas-api.

## What's already done
- `cloudflared` binary installed at `/usr/local/bin/cloudflared`
- Config skeleton at `/home/robindey/.cloudflared/config.yml` (tunnel name `atlas-api`, ingress → `http://localhost:8080`)
- atlas-api hardened with bearer-key auth + early-429 concurrency guard (see docs in repo / commit log)

## Robin's steps (one-time, needs browser)

### 1. Authenticate cloudflared with your Cloudflare account
```bash
cloudflared tunnel login
```
Opens a browser URL — pick the `thebeastagi.com` zone. Writes `~/.cloudflared/cert.pem`.

### 2. Create the named tunnel + DNS route
```bash
cloudflared tunnel create atlas-api
# note the tunnel UUID it prints; credentials JSON is written to ~/.cloudflared/<UUID>.json
ln -sf ~/.cloudflared/<UUID>.json ~/.cloudflared/atlas-api.json   # match config.yml
cloudflared tunnel route dns atlas-api atlas.thebeastagi.com
```

### 3. Run it as a service (survives reboot)
```bash
sudo cloudflared --config /home/robindey/.cloudflared/config.yml service install
sudo systemctl enable --now cloudflared
```
(Alternatively run it as a user unit next to atlas-olmo3-think.service.)

### 4. Verify
```bash
curl https://atlas.thebeastagi.com/v1/models          # 200, no auth needed (health/monitor route)
curl https://atlas.thebeastagi.com/v1/chat/completions -H "Authorization: Bearer $(cat /home/robindey/.config/atlas/api-key)" \
  -H "Content-Type: application/json" -d '{"model":"olmo3-7b","messages":[{"role":"user","content":"hi"}],"max_tokens":32}'
```

### 5. Close the raw port (after tunnel verified)
In GCP console → VPC firewall: delete/disable the rule allowing tcp:8080 from 0.0.0.0/0. The tunnel connects outbound; no inbound port needed.

## Notes
- API key lives at `/home/robindey/.config/atlas/api-key` (0600). The systemd unit passes it to atlas-api via `ATLAS_API_KEY_FILE`.
- `GET /v1/models` and `GET /health` are intentionally unauthenticated (uptime monitors). Everything else under `/v1/*` requires `Authorization: Bearer <key>`.
- Concurrency: max 2 in-flight inferences; 3rd+ concurrent request gets `429` + `Retry-After` (OpenRouter requirement — no queueing).
