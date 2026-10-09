# Deployment

How to put Model Playground on the internet, and why the pieces are arranged the
way they are.

There are two deployments, and they are not alternatives to choose between
once — they answer different questions:

- **The full stack** (§1–§9) is `docker-compose.prod.yml`: **Caddy** (TLS) →
  **nginx** (the built SPA, and `/api` forwarding) → **gunicorn** (Django) →
  **Postgres** + **Redis**, with a Celery worker and beat alongside. It needs a
  VM, and it is the only one with sign-in and the registry.
- **The static frontend** (§10) is `frontend/dist` built with
  `VITE_BACKEND=off`, in S3 behind CloudFront at
  **https://playground.benedictthekkel.com**, for about $1 a month. Every model
  page works there, because every model page runs in the browser. CI deploys it
  on each push to `main`.

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

> **Splitting the frontend onto a static host while keeping a backend** re-opens
> exactly the problem above: you must set `VITE_API_BASE_URL`, serve the API
> over HTTPS on its own domain, and maintain a CORS allowlist. That objection is
> about the *backend*. With no backend at all (`VITE_BACKEND=off`) there is
> nothing cross-origin to call, and a static host is the cheaper answer — §10.
> When that deploy grows a backend, it stays single-origin by adding the API as
> a second CloudFront origin (§10.8), not by pointing the SPA elsewhere.

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

cp .env.prod.example .env.prod
just secret-key          # paste into SECRET_KEY
$EDITOR .env.prod        # DOMAIN, ACME_EMAIL, ALLOWED_HOSTS,
                         # CSRF_TRUSTED_ORIGINS, POSTGRES_PASSWORD

just prod-config         # renders the compose file; fails loudly on a missing var
just up-prod             # build and start
```

`.env.prod` sits beside `docker-compose.prod.yml` and is gitignored.
It is passed **explicitly** — every `just *-prod` / `prod-*` recipe runs
`docker compose --env-file .env.prod -f docker-compose.prod.yml …` — because the
root `.env` is the *development* file (#61) and compose reads `.env` by default:
a bare `docker compose -f docker-compose.prod.yml` would interpolate dev values,
or stop on the first `${VAR:?}` the dev file lacks. `backend/tests/test_env_config.py`
fails on any prod compose command in the justfile or these docs that drops the flag.
[`.env.prod.example`](../../.env.prod.example) is the source of truth for *which*
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
docker compose --env-file .env.prod -f docker-compose.prod.yml exec backend python manage.py createsuperuser
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

**The server tracks `main`, and that is what makes `main` meaningful.** Work is
written on `develop` and merged into `main` when an issue completes (see
[`onboarding.md`](onboarding.md)), so `git pull --ff-only` here is the only place
the distinction is cashed in. Check out anything else on the host and `just
deploy` quietly ships unfinished work.

Migrations run automatically, but only in one place. The `migrate` service sets
`RUN_MIGRATIONS=1`, applies them and exits; `backend`, `celery_worker` and
`celery_beat` run the same image with it `0` and wait on
`service_completed_successfully` before starting.

Both halves of that matter. Migrating from one service only means replicas never
race each other on boot — the entrypoint does nothing unless asked. Gating the
others on it means a first deploy, or any schema change, never has gunicorn
serving 500s and Celery hitting tables that do not exist yet.

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
  | docker compose --env-file .env.prod -f docker-compose.prod.yml exec -T db \
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
| Compose refuses to start, names a variable | A `${VAR:?}` with no value in `.env.prod` (or `--env-file .env.prod` left off) — the guard working |
| Model downloads fail, app loads | A CSP or header change broke the CDN fetch or the cache bucket |

---

## 9. What CI does and does not cover

[`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) runs lint, the
Django test suite, `check --deploy`, the frontend unit tests, the production
build, the bundle budget, the mocked Playwright suite against that build
(`e2e-static`), and a build of both production images.

It does **not** run the mocked suite against the dev server (`just fe-e2e`).
That job took ~10 minutes and mostly repeated `e2e-static`; what only it
covered — the `@api` specs (sign-in, the registry UI) and the `@devserver`
specs — is now run by hand before a release that touches them.

It does **not** run the `@slow` specs — they download real ONNX weights from
Hugging Face. Those are the only tests that exercise a real ONNX Runtime
session, and both DeepFilterNet bugs of 2026-09-01 shipped past a green unit
suite. Run `just fe-e2e-slow` by hand before shipping a model change; see
[e2e-testing.md](./e2e-testing.md).

The **full stack** never deploys automatically: `just deploy` is a person
typing it. The **static frontend** does — a push to `main` runs the `deploy`
job (§10.4), gated on the unit suite, both Playwright passes (the dev server,
and the shipped `VITE_BACKEND=off` bundle) and the infra tests, and ending in a
smoke test against the live site.

---

## 10. Static frontend on S3 + CloudFront

**https://playground.benedictthekkel.com** — the app with no backend (#57).

### 10.1 What it is, and why a backend is not needed

Every model page runs in the browser and fetches its weights straight from the
Hugging Face CDN. Only three things ever call `/api`: sign-in (and `useMe`,
which the navbar and `/` use), the registry catalogue on `/home` and
`/playground`, and the Celery `TaskTrigger`, which no route renders.
`VITE_BACKEND=off` removes all three, read in exactly one place
(`src/lib/features.ts`): `/` redirects to `/home`, `/login` and `/signup` do
too, and the catalogue is not rendered. The assertion is **no request to
`/api` on the wire** (`src/__tests__/staticBuild.test.tsx`,
`e2e/specs/static-build.spec.ts`), not "no sign-in button" — on this deploy a
request to `/api` does not fail, it is rewritten to `index.html` and answered
200, and a JSON parser gets a page of HTML.

The pieces, all in `infra/` (Pulumi, TypeScript) and `scripts/`:

| Piece | Where | Applied by |
|-------|-------|------------|
| State bucket | AWS CLI, below | an admin, once |
| CI identity: deploy + preview roles | `infra/bootstrap` | an admin, once (`just infra-bootstrap`) |
| Bucket, CloudFront, certificate, DNS | `infra/site` | CI on `main`; first time locally (`just infra-up`) |
| The files | `scripts/deploy-frontend.sh` | CI on `main`, after `pulumi up` |
| Smoke test | `scripts/smoke-frontend.sh` | CI on `main`, last |

The hostname is `playground`, not `model_playground`: certificates may not
contain `_` (CA/Browser Forum, since 2019), and no certificate means no HTTPS,
which means no `navigator.gpu`.

### 10.2 Bootstrap (once, locally, with admin credentials)

This is a person at a terminal on purpose: CI cannot create the identity it
authenticates with.

**1. The state bucket.** Pulumi keeps state in S3, not Pulumi Cloud, so there
is no long-lived `PULUMI_ACCESS_TOKEN`. The bucket cannot be managed by the
stack whose state it holds, so it is made by hand:

```bash
B=model-playground-pulumi-state-762233760445
aws s3api create-bucket --bucket $B --region ap-southeast-2 \
  --create-bucket-configuration LocationConstraint=ap-southeast-2
aws s3api put-bucket-versioning --bucket $B --versioning-configuration Status=Enabled
aws s3api put-public-access-block --bucket $B --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3api put-bucket-encryption --bucket $B --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
```

Both `Pulumi.yaml` files name it under `backend.url`, so no `pulumi login` is
needed anywhere.

**2. The GitHub OIDC provider.** It is an account-wide singleton that other
projects' roles already trust, so `infra/bootstrap` **looks it up and never
owns it** — the same rule as the Route 53 zone. It exists in this account. In
an account without one:

```bash
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com --client-id-list sts.amazonaws.com
```

**3. The stacks.** A passphrase secrets provider: neither stack holds a secret
value, so the passphrase protects nothing sensitive today. If one is ever
added, move to `awskms://` (about $1/month).

```bash
export PULUMI_CONFIG_PASSPHRASE=...        # the same value goes to GitHub below
just infra-install
(cd infra/bootstrap && pulumi stack init production --secrets-provider passphrase)
(cd infra/site      && pulumi stack init production --secrets-provider passphrase)
git add infra/*/Pulumi.production.yaml     # init added an encryptionsalt; commit it
just infra-bootstrap                       # prints deployRoleArn and previewRoleArn
```

**4. GitHub.** In the repository settings:

- an **environment** named `production`, deployment branches limited to `main`
  (and required reviewers, if a release should wait for a click);
- **repository variables** `AWS_DEPLOY_ROLE_ARN` and `AWS_PREVIEW_ROLE_ARN`,
  from the bootstrap outputs. Repository-level because a job's `if:` can only
  read those; the `deploy` and `preview` jobs are skipped while they are unset;
- a **repository secret** `PULUMI_CONFIG_PASSPHRASE`.

### 10.3 The first site deploy (locally)

```bash
just infra-up          # 5–30 minutes the first time: ACM DNS validation + CloudFront propagation
just deploy-frontend   # VITE_BACKEND=off build, upload, smoke test
```

Run the first `pulumi up` by hand rather than leaving it to CI: validation and
first propagation can outlast a patient job, and a half-created distribution is
easier to watch from a terminal. After that, CI owns it.

### 10.4 What CI does

On every push to `develop`, `preview` assumes the **read-only** role and runs
`pulumi preview --refresh`, so drift and pending infrastructure changes are
visible before a release.

On every push to `main`, `deploy` — after `frontend`, `e2e`, `e2e-static` and
`infra` pass — assumes the **deploy** role, runs `pulumi up`, runs the upload
script on the **same `dist` artifact** `frontend` built and `e2e-static`
tested (it is never rebuilt), then the smoke test. The job has
`concurrency: deploy-production` without cancellation, and the workflow no
longer cancels in-progress runs on `main`: a release queues behind the one
uploading, rather than killing it halfway.

The smoke test (`scripts/smoke-frontend.sh`) checks, against the live site:
`/home` is 200 and `no-cache`; `/` and `/ASR/` are 301s to `/home` and
`/asr`; a real `/assets/*.js` is `immutable`; a real `.wasm` is
`application/wasm`; `/asr` is its **own** page (its canonical names it, so
the generic shell fails); `robots.txt` names the sitemap and `sitemap.xml`
lists `/asr`; `http://` redirects to `https://`; every request verifies the
certificate (no `-k`); and the bucket's own URL answers 403. It retries `/home` for up to five minutes
(`SMOKE_ATTEMPTS` × `SMOKE_WAIT` seconds, 30 × 10 by default) while a new
distribution or an invalidation settles.

### 10.5 The trust model

The repository is public, and the account holds other things — other
certificates in us-east-1, another CloudFront site, and a shared zone whose
apex and mail records belong to nobody in this repo. So:

- **Trust is the OIDC `sub` claim, exact-matched.** The deploy role trusts
  `repo:bthek1/model_playground:environment:production` — an *environment*,
  not a branch, so a fork's pull request cannot claim it and the environment
  can require approval. The preview role trusts
  `repo:bthek1/model_playground:ref:refs/heads/develop`, which is why the
  `preview` job must not declare an `environment:` (that changes the subject).
  Never widen either to `repo:bthek1/model_playground:*`.
- **Route 53 writes are scoped to one zone and to this site's names.**
  `route53:ChangeResourceRecordSetsNormalizedRecordNames` allows only
  `playground.benedictthekkel.com` and `_*.playground.benedictthekkel.com`
  (the ACM validation record), in record types A, AAAA and CNAME. A bad
  `pulumi up` cannot rewrite the apex or the mail records. The stack looks the
  zone up with `getZone` and never creates, imports or deletes it.
- **Where a resource cannot be named in advance, it is scoped by tag.** ACM
  certificate and CloudFront distribution ARNs are random, so deleting or
  updating one requires `project=model-playground`, which the site stack sets
  and the other projects' resources do not carry. Certificates can be
  requested only for this domain, only in us-east-1.
- **No `*:*`.** The site bucket's `s3:*` is scoped to that bucket; reads that
  AWS cannot scope (`List*`) are the only unscoped statements.
  `infra/bootstrap/policies.test.ts` asserts all of this.

### 10.6 The upload, and why it is ordered

Files are **not** Pulumi resources. One `BucketObject` per file is hundreds of
resources, and it deletes the previous build's hashed chunks the moment the new
build lands — which breaks every open tab whose cached `index.html` still
references them, the next time it lazy-loads a route. The upload script:

1. uploads `assets/**` first, `public, max-age=31536000, immutable`;
2. uploads `*.wasm` with an **explicit** `application/wasm` — ONNX Runtime's
   streaming compilation refuses anything else, and the CLI's MIME guess is
   not a contract;
3. uploads each indexable route's own `<route>/index.html`, then `robots.txt`
   and `sitemap.xml` (§10.8c), then `site.webmanifest` and `index.html`
   **last** — all `no-cache, must-revalidate`, all with their content type
   stated. That is the switch-over; a route page names the same hashed chunks
   `index.html` does, so it goes after them too;
4. **never deletes.** An asset the new build no longer references is
   re-written in place tagged `stale=true`, once; the bucket's lifecycle rule
   expires tagged objects 30 days later (`site:staleAssetDays`). Re-writing
   resets the object's age, so the 30 days run from when it went stale, not
   from when it was built — and it is tagged only once, or every deploy would
   restart its clock. A chunk a later build brings back is re-uploaded
   untagged, which takes it out of the rule;
5. invalidates only `/index.html` and `/site.webmanifest`. A hashed name never
   changes meaning, so it never needs invalidating; the route pages, robots
   and sitemap are `no-cache`, which the edge revalidates per request, and one
   invalidation path per route would spend the free allowance in a few deploys.

SPA deep links (`/asr`) are a CloudFront Function (`infra/site/spa-rewrite.js`)
that treats any path whose last segment has no dot as a route — **not** a
distribution-wide custom error response, which would also turn a future
`/api` origin's 404s into the app shell. A route listed in its `ROUTES` is
served its own `/<route>/index.html`; anything else (sign-in, a mistyped URL)
gets the root `/index.html`. `/`, a trailing slash and an upper-case letter
are **301s** to the canonical form, query string kept (§10.8c). A route
containing a dot would bypass all of it; none does, and `spa-rewrite.test.ts`
checks that against the generated route tree. There is **no CSP**, for §1's
reason.

### 10.7 Cost

About **$0** for S3 and CloudFront inside the always-free tier — ~1 TB/month of
CloudFront transfer, which is tens of thousands of visits at ~25 MB per cold
visitor (model weights come from Hugging Face, not from here). The hosted zone
($0.50/month) and the domain were already paid for; the ACM certificate is
free. The only way this gets expensive is a traffic spike past the free tier —
a CloudWatch billing alarm is the guard, if one is wanted.

### 10.8 Adding a backend later

Add Django as a **second CloudFront origin** with ordered cache behaviours for
`/api/*`, `/admin/*` and `/static/*` ahead of the default, caching disabled and
all methods allowed. The site stays single-origin, so there is still no CORS,
and the SPA's empty `VITE_API_BASE_URL` keeps working. Then build with
`VITE_BACKEND=on`. Until then, know that `/api/anything` on this deploy is
rewritten to `index.html` and returns 200 — nothing calls it, which is what the
static-build tests assert.

### 10.8a Analytics: `/ingest/*` (#60)

The distribution has a second origin, PostHog's ingestion host, behind an
ordered `/ingest/*` behaviour that is matched before the default one —
without it, `spa-rewrite` answers `/ingest/e/` with `index.html` and a 200 and
every event vanishes. It caches nothing and carries its own function,
`ingest-strip.js`, which removes the prefix. The analytics key is the
**project** key (`phc_`, public), set as the repository variable
`VITE_POSTHOG_KEY`; unset, the site ships with analytics off. The one-time
project setup is `just posthog-setup`, run locally with the personal key — see
[analytics.md §5](../explanations/analytics.md). An EU project also needs
`pulumi config set site:posthogHost eu.i.posthog.com`.

### 10.8b Legal pages and notices (#62)

Before the first deploy that carries them, make sure **`CONTACT_EMAIL`**
(`frontend/src/legal/site.ts`) is a working inbox: every legal page publishes
it. The build writes `dist/THIRD-PARTY-NOTICES.txt` beside `index.html`, and
`check:bundle` fails if it is missing. The upload script's root-files pass
ships it with the other root files, and `spa-rewrite` leaves it alone because
it has an extension. Adding a host the browser fetches from (a CDN, a dataset
mirror) means adding a row to `src/legal/recipients.ts`. See
[legal.md](../explanations/legal.md).

### 10.8c Search engines: per-route pages, sitemap, redirects (#63–#65)

A single-page app serves every URL the same shell: one `<title>`, one
description, an empty `<div id="root">`. Google fills it in after rendering
the JavaScript, late; link-preview bots and most other crawlers never do. So
the build writes what they need (`frontend/scripts/seoPages.ts`, from the
pure `frontend/src/seo/`):

- **`dist/<route>/index.html`** for every page in `SITE_PAGES` — home, every
  taxonomy task, the four legal pages — with its own title, description,
  canonical, `og:url`, absolute `og:image`, JSON-LD (`WebApplication` on home,
  a `BreadcrumbList` elsewhere) and a static `<h1>`, description and links
  inside `#root`, which `createRoot` replaces when the app mounts. Sign-in and
  sign-up are not in it: an account form is not a search result.
- **`dist/sitemap.xml`** (no `lastmod` — it would be the build date on every
  page, and Google ignores an inaccurate one) and **`dist/robots.txt`** (allow
  all but `/ingest/`, and name the sitemap).

`SITE_PAGES` is built from the sidebar taxonomy, whose entries now carry a
one-line `description`. The function's `ROUTES` is a **copy**, because the
edge has no module system; `frontend/src/seo/edgeRoutes.test.ts` fails,
naming the difference, when they disagree. A page missing from `ROUTES` is
served the generic shell; a route listed with no file answers 403.

Canonicals are written against `SITE_URL` in `frontend/src/lib/site.ts`; a
build for another host sets `SITE_URL` in its environment. An unknown path
still answers 200 with the shell — the edge cannot know the router's routes —
so the app's not-found page adds `noindex` while it is mounted.

**Deploy order, the first time.** `pulumi up` updates the function before the
upload writes the route files, so for the minute between them a route listed
in `ROUTES` with no file yet answers 403. It happens once, when this lands, and
again only for the newest route when one is added.

**Search Console** (manual, once): verify the domain property with a DNS TXT
record in the hosted zone, submit `https://<domain>/sitemap.xml`, and do the
same in Bing Webmaster Tools. Nothing in the browser contacts either, so
`legal/recipients.ts` does not change.

### 10.9 How it is tested

None of this can be exercised by deploying it — a wrong IAM condition, a
rebuilt artifact or a cancelled upload all look fine until the day they
matter. So each property has a test that runs without AWS (`just infra-test`,
and the `infra` CI job):

| What | Test | Pins |
|------|------|------|
| No `/api` request with the flag off | `frontend/src/__tests__/staticBuild.test.tsx` | the real route tree, MSW's request log, a stale token in storage |
| …on every page, at the import | `frontend/src/__tests__/conventions.test.ts` | `VITE_BACKEND` read only in `features.ts`; every `src/api` importer on a named list with its gate; `useMe`, the redirects and the catalogue gated |
| …in a real browser, on the built bundle | `frontend/e2e/specs/static-build.spec.ts` (`E2E_STATIC=1`) | the request log again, `/` → `/home`, a deep-link reload |
| IAM documents | `infra/bootstrap/policies.test.ts` | exact `sub`, no `*:*`, Route 53 names/types, tag conditions, both stack configs agreeing |
| IAM wiring | `infra/bootstrap/bootstrap.test.ts` (Pulumi mocks) | the OIDC provider looked up not created; each role gets its own document |
| The site stack | `infra/site/site.test.ts` (Pulumi mocks) | public access blocked, one bucket-policy principal, `redirect-to-https`, no custom error responses, us-east-1 certificate, the zone never created, only `playground` records |
| Deep links and redirects | `infra/site/spa-rewrite.test.ts` | the function's exact bytes; a listed route gets its own file, anything else the shell; `/`, a trailing slash and upper case 301 with the query string kept; no route in `routeTree.gen.ts` has a dot |
| SEO output (#63–#65) | `frontend/src/seo/*.test.ts`, `frontend/e2e/specs/seo.spec.ts` (`E2E_STATIC=1`) | the function's `ROUTES` equals `SITE_PAGES`; every page's head, canonical, JSON-LD and `<h1>`; every sitemap URL fetched with no JS; the head following a client-side navigation; the not-found page's `noindex` |
| Upload order | `infra/deploy-frontend.test.ts` (stub `aws`) | assets → wasm → route pages → robots/sitemap → index last; explicit wasm, HTML, text and XML types; no delete; stale-marking once, never a live chunk; the two-path invalidation |
| Smoke test | `infra/smoke-frontend.test.ts` (stub `curl`) | passes a healthy site; fails, naming it, on each broken property |
| Analytics through `/ingest/*` (#60) | `infra/site/site.test.ts`, `infra/site/ingest-strip.test.ts`, `infra/workflow.test.ts`, `infra/posthog-setup.test.ts` | the ordered behaviour precedes the default, targets PostHog, caches nothing, carries the strip function and never `spa-rewrite`; CI takes the `phc_` key from a variable on the shipped build only; the setup script never prints the personal key — see [analytics.md](../explanations/analytics.md) |
| CD wiring | `infra/workflow.test.ts` (reads `ci.yml`) | the `off` artifact built once and never rebuilt; `deploy`'s needs, environment, OIDC, no-cancel concurrency, step order; `preview` has no environment and never applies; only those two jobs get a token |

The one thing none of these reach is AWS itself: whether a policy is
*sufficient*. That surfaces as an `AccessDenied` in `pulumi up` — loud, and
named by the statement's `Sid` — and `pulumi preview` from the preview role
on `develop` is where it shows first.
