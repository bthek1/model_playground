# Deployment

How to put Model Playground on the internet, and why the pieces are arranged the
way they are.

The stack is `docker-compose.prod.yml`: **Caddy** (TLS) → **nginx** (the built
SPA, and `/api` forwarding) → **gunicorn** (Django) → **Postgres** + **Redis**,
with a Celery worker and beat alongside.

---

## 1. The one constraint that shapes everything

**The SPA and the API must be served from the same origin, over HTTPS.**

That is not a deployment preference; two separate facts force it.

**`src/api/client.ts` ships an empty base URL.** The app calls `/api` on
whatever origin served the page. In development the Vite dev server proxies that
to Django. The dev server is not in production, so something else has to do the
forwarding — here, nginx. Without it, the frontend would have to call Django on
another origin, which the comment in `client.ts` already explains is blocked
three ways: mixed content, CORS, and the browser's Local Network Access gate.

**`navigator.gpu` only exists in a secure context.** Serve the app over plain
HTTP on anything but `localhost` and `detectWebGPU()` returns `unsupported` — on
hardware that supports it perfectly well, and indistinguishably from a browser
that genuinely doesn't. Every model page silently falls back to WASM. TLS is a
functional requirement of this app, not hardening.

So Caddy terminates TLS (obtaining certificates automatically) and nginx serves
the SPA and forwards `/api`, `/admin` and `/static` to Django on the same
hostname. In that shape **CORS is not involved at all**, which is why
`CORS_ALLOWED_ORIGINS` defaults to empty in `core/settings/prod.py`.

> **Splitting the frontend onto a static host** (Vercel, Netlify, S3) is
> possible, but it re-opens exactly the problem above: you must set
> `VITE_API_BASE_URL`, serve the API over HTTPS on its own domain, and maintain
> a CORS allowlist. You gain a CDN for a bundle that is already cached
> immutably. Prefer the single origin unless you have a reason.

What is *not* a deployment problem: **model weights**. The browser fetches them
straight from the Hugging Face CDN (`ModelCard.weights_url`), never through
Django, so there is no model hosting to arrange and no egress bill for
checkpoints. If you ever add a `Content-Security-Policy`, it must not block
`huggingface.co` / `cdn-lfs*.hf.co`, and response headers must not disturb the
`transformers-cache` bucket that `model/cache.ts` probes. Ship no CSP rather
than one that silently breaks every model load.

---

## 2. What you need

- A host with Docker and the Compose plugin (2 GB RAM is enough; the server runs
  no inference).
- A domain whose A/AAAA record already points at that host. Caddy proves control
  of it over HTTP, so **DNS must resolve before the first start** or certificate
  issuance fails and nothing is served over TLS.
- Ports **80** and **443** reachable. 80 is not optional — it is the ACME
  challenge and the HTTP→HTTPS redirect.

---

## 3. First deploy

```bash
git clone <repo-url> && cd model_playground

cp .env.example .env
just secret-key          # paste into SECRET_KEY
$EDITOR .env             # DOMAIN, ACME_EMAIL, ALLOWED_HOSTS,
                         # CSRF_TRUSTED_ORIGINS, POSTGRES_PASSWORD

just prod-config         # renders the compose file; fails loudly on a missing var
just up-prod             # build and start
```

`.env` sits beside `docker-compose.prod.yml` and is gitignored.
[`.env.example`](../../.env.example) is the source of truth for *which*
variables are required. Every one is declared `${VAR:?}` in the compose file, so
a missing value stops the stack rather than quietly substituting an empty
string.

Three that are easy to get wrong:

| Variable | Shape | Why |
|---|---|---|
| `ALLOWED_HOSTS` | `playground.example.com` | Hostnames, comma-separated, **no scheme** |
| `CSRF_TRUSTED_ORIGINS` | `https://playground.example.com` | **With** the scheme. Django checks the `Origin` header of unsafe requests against it, so `/admin/` logins fail without it |
| `SECRET_KEY` | 50 random characters | Changing it invalidates every session and password-reset link |

Then create the first user:

```bash
docker compose -f docker-compose.prod.yml exec backend python manage.py createsuperuser
```

### Verifying it worked

Four checks, in order of what they prove:

```bash
curl -I https://your-domain/                      # 200, and a valid certificate
curl -s https://your-domain/api/registry/models/  # JSON — nginx reached Django
```

Then in a browser:

3. Open `https://your-domain/admin/` — it must render **styled**. Unstyled CSS
   means WhiteNoise isn't serving `/static/`.
4. Open a model page (say `/image-classification`) and load a model to `ready`.
   This is the only check that proves TLS, WebGPU, the Hugging Face CDN and the
   cache bucket all survived the deploy at once.

---

## 4. Routine deploys

```bash
just deploy     # git pull --ff-only, rebuild, restart
```

Migrations run automatically, but only in one place. The `migrate` service sets
`RUN_MIGRATIONS=1` and exits; `backend`, `celery_worker` and `celery_beat` share
the same image with it unset. That is deliberate — two replicas racing `migrate`
on boot is a real failure mode, so the entrypoint never migrates unless asked.

```bash
just logs-prod          # tail everything
just down-prod          # stop (volumes kept)
just prod-shell         # Django shell in the running container
```

---

## 5. Backups

The database is the only state worth keeping — model weights live in the
browser's cache and Redis holds nothing durable.

```bash
just prod-backup        # -> backups/YYYYmmdd-HHMMSS.sql.gz
```

Restore:

```bash
gunzip -c backups/<file>.sql.gz \
  | docker compose -f docker-compose.prod.yml exec -T db \
      sh -c 'psql -U "$POSTGRES_USER" "$POSTGRES_DB"'
```

Copy the dumps off the host. A backup on the same disk as the database is not a
backup.

---

## 6. How the pieces fit

```
                    :443
                     │
              ┌──────▼──────┐
              │    Caddy    │  TLS, automatic certificates
              └──────┬──────┘  sets X-Forwarded-Proto: https
                     │
              ┌──────▼──────┐
              │    nginx    │  SPA + history fallback
              │  (frontend) │  /api, /admin, /static ─┐
              └─────────────┘                         │
                                              ┌───────▼───────┐
                                              │   gunicorn    │
                                              │   (backend)   │
                                              └───┬───────┬───┘
                                                  │       │
                                            ┌─────▼──┐ ┌──▼─────┐
                                            │Postgres│ │ Redis  │
                                            └────────┘ └───▲────┘
                                                           │
                                          celery worker + beat
```

Only Caddy publishes a port. Postgres, Redis, gunicorn and nginx are reachable
only on the compose network.

### The `X-Forwarded-Proto` chain

Caddy terminates TLS and forwards plain HTTP, so Django must be told that a
request *arrived* encrypted. Caddy sets `X-Forwarded-Proto: https`, nginx passes
it through, and `SECURE_PROXY_SSL_HEADER` in `prod.py` reads it.

Break any link and you get one of two failures: `/admin/` rejects every POST
with a CSRF origin mismatch, or — if `SECURE_SSL_REDIRECT` is on — Django sees
`http://`, redirects to HTTPS, and the proxy forwards the redirected request
straight back. An infinite loop. The production compose therefore sets
`SECURE_SSL_REDIRECT=False`, because **Caddy already redirects at the edge**;
doing it twice is how the loop starts.

This also means the backend container must never be exposed directly to the
internet — the header is only trustworthy because the proxy overwrites it.

### Caching

nginx serves Vite's fingerprinted `/assets/` as `immutable` for a year, and
`index.html` as `no-cache`. That pairing is what makes a deploy safe: a cached
`index.html` would keep pointing at chunk filenames that no longer exist.

`.wasm` is served as `application/wasm` explicitly. ONNX Runtime uses streaming
compilation, which refuses any other content type — as `application/octet-stream`
the models fall back to a slow path or fail outright.

### Static files

Django's own static assets (admin, DRF browsable API) are served by **WhiteNoise
inside the backend container**, with hashed filenames from
`CompressedManifestStaticFilesStorage`. `collectstatic` runs at *build* time, so
boot does no filesystem work and every replica is identical.

The alternative — a shared volume that nginx serves — was not chosen because it
couples the two containers' lifecycles for a few hundred kilobytes that are only
ever touched by `/admin/`.

---

## 7. Sizing and cost

The server does no inference: it stores catalogue rows and run metadata. All
compute happens in the visitor's browser on the visitor's GPU, and weights come
from the Hugging Face CDN. A small VPS serves a lot of users, and traffic scales
with page views rather than with model usage.

The Celery worker exists for the registry's background tasks, not for models.

---

## 8. Troubleshooting

| Symptom | Cause |
|---|---|
| Every model page says WebGPU unsupported | Not served over HTTPS, or a browser behind a flag (Firefox needs `dom.webgpu.enabled`). See [webgpu-inference.md](../explanations/webgpu-inference.md) |
| `/admin/` renders unstyled | WhiteNoise isn't serving `/static/` — check the nginx `location` block reached the backend |
| CSRF verification failed on admin login | `CSRF_TRUSTED_ORIGINS` missing the scheme, or the `X-Forwarded-Proto` chain broken |
| Redirect loop | `SECURE_SSL_REDIRECT` on while Caddy also redirects |
| Deep link 404s, root works | SPA history fallback missing — nginx `try_files` |
| Certificate never issued | DNS doesn't resolve to the host yet, or port 80 is blocked |
| Compose refuses to start, names a variable | A `${VAR:?}` with no value in `.env` — the guard working |
| Model downloads fail, app loads | A CSP or header change broke the CDN fetch or the cache bucket |

---

## 9. What CI does and does not cover

[`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) runs lint, the
Django test suite, `check --deploy`, the frontend unit tests, the production
build, the bundle budget, the mocked Playwright suite, and a build of both
production images.

It does **not** run the `@slow` specs — they download real ONNX weights from
Hugging Face. Those are the only tests that exercise a real ONNX Runtime
session, and both DeepFilterNet bugs of 2026-09-01 shipped past a green unit
suite. Run `just fe-e2e-slow` by hand before shipping a model change; see
[e2e-testing.md](./e2e-testing.md).

Nothing deploys automatically. `just deploy` is a person typing it.
