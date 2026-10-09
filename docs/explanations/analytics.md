# Product analytics

Anonymous usage counts for the static site, through PostHog: which pages are
opened, which models are actually **loaded** (not just selected), how long loads
and runs take on real hardware, how often WebGPU is available, and where loads
fail.

Plan: [issue #60](https://github.com/bthek1/model_playground/issues/60).

---

## 1. The rule: what the app did, never what it was given

Several pages promise "nothing is uploaded", and an analytics call is an upload.
So no prompt, sentence, CSV, column name, file name, image, audio, label, model
output or error *message* ever reaches an event. Three layers enforce it, so a
bug in one is caught by the next:

| Layer | Where | What it does |
|---|---|---|
| Closed allowlist | `src/analytics/schema.ts` (`sanitize`) | Drops every property key not on `ALLOWED_PROPS`. A string value must look like an identifier (`[A-Za-z0-9_./:@-]`, no spaces) — a catalogue id, a route pattern, an enum — so a sentence cannot pass even under an allowed key. Errors become an `errorKind` category; the message is never sent. |
| The last gate | `src/analytics/client.ts` (`beforeSend`) | Rebuilds every event's properties from `SDK_PROPS` (a strict list — a property a future SDK adds is dropped until someone reads it) plus the app allowlist again. Replaces `$current_url` with origin + route **pattern**, so a query string or hash never leaves; drops `$referrer`, `$set`, `$set_once`. |
| Off unless asked | `src/lib/features.ts` | `ANALYTICS_ENABLED` is true only in a build with `VITE_POSTHOG_KEY`, and only CI's shipped build has one. Disabled, the facade imports nothing and sends nothing. |

The events (`src/analytics/schema.ts`):

| Event | Sent from | Properties |
|---|---|---|
| `$pageview` | `usePageviews` in `__root.tsx` | `route` — the matched route **id**, never `location.href` |
| `model_load_started` / `_ready` / `_failed` / `_cancelled` | `useModelWorker` (once, for every task page) | `modelId` (public catalogue id), `task`, `backend`, `loadedInMs` / `elapsedMs`, `retry`, `errorKind` |
| `model_run_completed` / `_failed` | `useModelWorker` | `modelId`, `task`, `runMs`, `errorKind` |
| `webgpu_status` | the facade, once per session | `webgpuStatus` (`unsupported` / `no-adapter` / `adapter`), `shaderF16` — adapter-level on purpose: acquiring a device to say `ready` would cost a GPU device per visit |
| `feature_used` | the bespoke pages' triggers | a fixed `feature` enum (`rl_train`, `graph_train`, `link_predict`, `graph_classify`, `tabular_classification`/`_regression`, `bfs_step`) and a `family` (algorithm or architecture) — never a row count, column, target or metric derived from the user's data |

`/time-series-forecasting` sends only its pageview: it has no trigger — the
forecast is a derivation recomputed as you edit — and an event per keystroke
would be noise.

The public description of all this is `/privacy` (#62). Its PostHog row and its
analytics section render only in a build with a key, and are written from this
table. Change an event or a property and the notice's wording in
`src/legal/recipients.ts` may need to change too. See
[legal.md](legal.md).

## 2. Phase 0's decisions

- **The SDK build is `posthog-js/dist/module.slim.no-external`** — 49 KB gzip as
  a lazy chunk, against 98 KB for the default build. It is the core with no
  extensions and **no external-script loader**, so it cannot fetch
  session-recording, surveys or toolbar code from PostHog's asset host however
  the project is configured. That also removed the plan's `/ingest/static/*`
  origin: there is nothing to proxy.
- **`persistence: "memory"`** — no cookie, no `localStorage`, no consent banner,
  and tabular's `Storage.setItem` spies stay meaningful. A returning visitor
  counts as a new one; accepted. (`cookieless_mode` would need a project-side
  setting and server-side hashing for the same outcome.)
- **`person_profiles: "never"`, `advanced_disable_flags: true`** — no person
  records, and no `/flags` round trip through which a toggle in the PostHog UI
  could switch a capture feature on without a code change.
- **The opt-out lives in its own key** (`mp.analytics.optOut`), because memory
  persistence forgets an SDK-level opt-out on reload. Opted out, the SDK chunk is
  never requested at all. The switch is in the system panel; the sentence
  *"Anonymous counts of which pages and models are used — never your inputs or
  results."* sits beside it and beside the existing privacy claims (`/home`,
  both tabular pages). A build without a key renders neither — a sentence
  describing what it sends would be false there. `/home`'s "Nothing is sent to a
  server" became "Nothing you give a model is sent to a server", which is true
  in both builds.

## 3. The SDK stays lazy

`src/analytics/client.ts` is the **only** module that imports `posthog-js`, and
only `src/analytics/index.ts` reaches it — by dynamic `import()`, on idle, after
the first call. One static import anywhere folds the SDK into the entry chunk
(the `INEFFECTIVE_DYNAMIC_IMPORT` trap in `CLAUDE.md`). Calls made before it
resolves are queued (capped at 100) and flushed in order. If the chunk is
ad-blocked or the tab is offline, the facade drops the queue and stops trying —
analytics must never be a way for a page to fail.

`conventions.test.ts` fails, naming the file, if `VITE_POSTHOG_*` is read outside
`features.ts`, `posthog-js` is imported outside `client.ts`, the client is
imported statically, or a module calls `posthog.capture` directly.
`check:bundle` fails if `posthog-js` reaches the entry chunk.

## 4. Same-origin ingestion

Events go to `/ingest/*` on the site's own origin. CloudFront has an ordered
behaviour for it (`infra/site/index.ts`), matched **before** the default one —
otherwise `spa-rewrite` would answer `/ingest/e/` with `index.html` and a 200,
and every event would vanish without an error. The behaviour:

- targets `us.i.posthog.com` (`site:posthogHost` overrides it for an EU project);
- uses the managed `CachingDisabled` cache policy and `AllViewerExceptHostHeader`
  origin request policy;
- carries **its own** viewer-request function, `ingest-strip.js`, which removes
  the `/ingest` prefix (CloudFront's origin path can only add one). It is never
  `spa-rewrite.js`; `site.test.ts` pins which function sits on which behaviour.

Why proxy at all: ad-blockers leave a first-party path alone, the site keeps
deployment.md §1's single-origin shape, and the E2E assertion "no request
leaves the origin" stays true with analytics on. The deploy role already covers
it — the function is inside the `model-playground-` name scope and managed
policies need no permission — so it lands with an ordinary `pulumi up`.

## 5. Setup (once), and the two keys

| Key | Prefix | What it can do | Where it lives |
|---|---|---|---|
| Personal API key | `phx_` | Read and delete every project in the organisation | `POSTHOG_ALL_ACCESS` in your root `.env`, used **only** by `just posthog-setup` |
| Project API key | `phc_` | Write events to one project. Public by design. | GitHub repository **variable** `VITE_POSTHOG_KEY` |

1. `just posthog-setup` (`scripts/posthog-setup.mjs`). It reports the key's
   region, finds or creates the `model-playground` project, turns IP
   anonymisation on and session recording, heatmaps, surveys, autocapture,
   exception and console capture and performance capture off server-side, and
   prints the `phc_` key. Idempotent — a second run says "no changes". It never
   prints the personal key or a response body.

   If PostHog refuses to **create** the project (HTTP 403), the organisation is
   usually on a one-project plan (the free plan is, and every organisation
   starts with one), or the key lacks a write scope. The script then lists the
   existing projects; adopt one with `just posthog-setup --project <id|name>`
   (or `POSTHOG_PROJECT`). Its privacy settings are changed to the ones above,
   which affects anything else already sending to that project — so it is an
   explicit flag, never a fallback.
2. Set `VITE_POSTHOG_KEY` as a repository **variable** (not a secret: it ships in
   the bundle). For an EU project, also
   `pulumi config set site:posthogHost eu.i.posthog.com` — the script says so.
3. The next `main` deploy builds with it; `pulumi up` adds `/ingest/*`.

If the personal key ever needs to live anywhere but your machine, rotate it and
scope a new one to `user:read`, `organization:read`, `project:read` and
`project:write`. Never give it a `VITE_` name: `check:bundle` fails the build if
any `phx_…` token reaches `dist/`, and `infra/workflow.test.ts` fails if CI ever
mentions it.

## 6. How it is tested

| What | Test |
|---|---|
| Allowlist, identifier rule, error categories | `src/analytics/schema.test.ts` |
| Every init option; `beforeSend` strips URLs, referrers, `$set*`, unknown SDK props | `src/analytics/client.test.ts` |
| Disabled imports nothing and touches no storage or network; queue and flush order; route stamped; opt-out never imports; opt-out survives a reload | `src/analytics/index.test.ts` |
| Load and run funnel from `useModelWorker`; failure sends a category, never the message | `src/model/useModelWorker.analytics.test.ts` |
| Note and switch render only with analytics; the copy; the opt-out key | `src/components/analytics/analytics.test.tsx` |
| Import rules | `src/__tests__/conventions.test.ts` |
| SDK out of the entry chunk; no `phx_` anywhere in `dist/` | `npm run check:bundle` |
| `/ingest/*` order, origin, policies, function; the strip function's bytes | `infra/site/site.test.ts`, `infra/site/ingest-strip.test.ts` |
| CI uses a variable, never a secret, on the shipped build only | `infra/workflow.test.ts` |
| Setup script: find-or-create, idempotent, region, never prints the key | `infra/posthog-setup.test.ts` |
| **The privacy assertion**: a typed sentence and a fitted CSV appear in no decoded payload; pageviews by pattern; query/hash never sent; every request same-origin; the opt-out survives a reload | `e2e/specs/analytics.spec.ts` (`just fe-e2e-static`) |

Two things the E2E spec had to learn, both silent:

- **PostHog drops bot traffic** — a `HeadlessChrome` user agent *or* brand, or
  `navigator.webdriver`. Under Playwright the SDK therefore captured nothing, and
  every privacy assertion passed on an empty payload. The spec presents an
  ordinary user agent and masks both properties, and asserts the payload is
  non-empty before asserting what is absent. (The flip side is real: headless
  crawlers on the live site produce no events.)
- **The SDK gzips its batches without saying so** — no `compression` query
  parameter on this version. `e2e/fixtures/ingest.ts` sniffs the gzip magic
  number; grepping an undecoded body for a secret passes vacuously.

Every page in every spec has `/ingest/**` answered locally by the base fixture,
so no test reaches PostHog — including the static pass, whose bundle carries
CI's real project key.

Not tested automatically, and checked by hand after the first deploy: events
appear with anonymised IPs and no recordings; a browser with Do Not Track sends
nothing (`respect_dnt`); an ad-blocker lets `/ingest` through.
