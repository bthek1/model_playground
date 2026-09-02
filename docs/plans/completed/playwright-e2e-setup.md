# Plan: Playwright End-to-End Testing Setup

**Status:** Complete. All of Phases 1–6 built and verified, including the
`firefox` project (previously blocked on a system package). **Phase 7 (CI) was
dropped from scope** by decision; that section is kept only as a record of what
it would have involved.

**Verified — both entry points, three consecutive runs each:**

| Command | Result | Notes |
|---------|--------|-------|
| `just fe-e2e` | **47 passed / 5 skipped**, exit 0, ~24s | chromium + firefox + webgpu, mocked API, Django stopped |
| `just fe-e2e-full` | **53 passed / 5 skipped**, exit 0, ~31s | adds the `setup` project and every `@backend` spec against real Django + Postgres |

- **Firefox** passes its 4 `@smoke` specs. Unblocking it needed no repo change —
  the host's dpkg was wedged by a zeroed `/var/lib/apt/extended_states` and a
  half-configured `distro-info-data` (unclean shutdown, 2026-08-01), which broke
  *every* `apt-get install` on the machine, not just Playwright's.
- **The 5 skips are correct:** the GPU-only specs self-skip because this host has
  no `/dev/dri`. The graceful-degradation specs run and pass regardless, which is
  the assertion that matters on a GPU-less runner.
- **Fresh-clone safety:** with `e2e/.auth/` deleted the default run is still
  green — the storage-state specs are `@backend`-tagged and filtered out.
- `just fe-test` still collects **only `src/`** — no `e2e/**.spec.ts` leaks into
  Vitest (which would fail immediately on importing `@playwright/test`).
  Deliberately stated as an invariant rather than a test count: other work lands
  in `src/` and the number drifts. `just fe-build` green, `just fe-lint` clean
  (0 errors; the remaining `react-refresh` warnings in `src/` are pre-existing).

**Date:** 2026-09-01

---

## Goal

Add **Playwright** as the browser end-to-end (E2E) layer for the frontend, complementing —
not replacing — the existing Vitest + Testing Library + MSW unit/component tests.

The two layers have clearly separated jobs:

| Layer | Tool | Runs against | Answers |
|-------|------|--------------|---------|
| Unit / component | Vitest + Testing Library + MSW (`frontend/src/**/*.test.tsx`) | happy-dom, mocked API | "Does this component render and behave?" |
| **End-to-end (new)** | **Playwright** (`frontend/e2e/`) | Real Chromium/Firefox against the **real Vite dev server**, optionally the real Django API | "Does a user flow actually work in a browser?" |

E2E exists here for three things unit tests structurally cannot cover:

1. **Routing + layout shell** — TanStack Router navigation, the taxonomy-driven sidebar
   (`components/layout/taskTaxonomy.ts` → `REAL_ROUTES` vs. the `tasks.$slug` placeholder),
   navbar, theme.
2. **Auth** — JWT login against a real backend, token persistence, the silent 401 refresh in
   `src/api/client.ts`, and protected-route redirects. MSW can only approximate this.
3. **WebGPU** — the one capability that *only* exists in a real browser. `detectWebGPU()`,
   the `getGPUDevice()` → pipeline → `readBackFloat32` path, and the Web Worker in
   `webgpu/worker.ts` are invisible to happy-dom. Playwright is the only way to assert both
   the `ready` path **and** graceful degradation.

## Background & constraints

These are repo-specific facts that shape the design; get them wrong and the suite is flaky.

- **The dev server is HTTPS with a self-signed cert.** `vite.config.ts` uses
  `@vitejs/plugin-basic-ssl` because `navigator.gpu` needs a secure context. Playwright must
  therefore set **`ignoreHTTPSErrors: true`** and use an `https://localhost:5180` base URL.
  Without this every test fails at `page.goto` with `ERR_CERT_AUTHORITY_INVALID`.
- **Ports:** frontend `5180`, backend `8006`. The Vite dev server proxies `/api` →
  `VITE_API_PROXY_TARGET`, so the browser only ever talks to one origin. E2E tests should
  keep that property — never point the test at `http://localhost:8006` directly.
- **WebGPU in headless Chromium is not guaranteed.** Availability depends on the machine's
  GPU/driver and Chromium build. WebGPU specs must therefore be a **separate, non-blocking
  project** that skips when `detectWebGPU()` does not return `ready`, so CI on a GPU-less
  runner stays green. Firefox on Linux additionally needs `dom.webgpu.enabled` — set via a
  Playwright `firefoxUserPrefs`, not documented as a manual step.
- **Base UI, not Radix.** Selectors must not assume Radix DOM/`data-state` conventions.
  Prefer role- and label-based locators (`getByRole`, `getByLabel`) over structural CSS.
- **Two API modes.** Some flows (catalog browsing) can run fully mocked via Playwright's
  `page.route`; auth needs the real Django API + a seeded user. The config should support
  both rather than forcing every dev to have Postgres running.
- **Docs travel with code** (repo rule) — this plan adds/updates docs in Phase 6.

## Non-goals

- Replacing any existing Vitest test. Nothing in `src/**/*.test.tsx` is deleted.
- Visual-regression / screenshot-diff testing. Deferred; noted as a follow-up.
- Backend E2E. Django is covered by pytest; Playwright drives the browser only.
- Running E2E against production or any deployed environment.

---

## Phase 1 — Install and scaffold

**Files:** `frontend/package.json`, `frontend/playwright.config.ts` (new),
`frontend/e2e/` (new), `frontend/.gitignore`, root `.gitignore`

1. Install into the **frontend** workspace (not the repo root — the frontend owns its tooling):
   ```bash
   cd frontend && npm i -D @playwright/test
   npx playwright install --with-deps chromium firefox
   ```
   Do **not** use `npm init playwright@latest` — it scaffolds a config and `tests/` layout
   that conflicts with the conventions below.
2. Create `frontend/e2e/` with this structure:
   ```
   e2e/
     fixtures/          # custom test fixtures (auth, webgpu, mock-api)
     pages/             # page objects, one per route under test
     specs/             # the actual *.spec.ts files
     utils/             # small helpers (seeding, selectors)
   ```
3. Add npm scripts to `frontend/package.json`:
   ```jsonc
   "test:e2e":        "playwright test",
   "test:e2e:ui":     "playwright test --ui",
   "test:e2e:headed": "playwright test --headed",
   "test:e2e:report": "playwright show-report"
   ```
4. Gitignore the Playwright artefacts: `frontend/test-results/`,
   `frontend/playwright-report/`, `frontend/blob-report/`, `frontend/.auth/`.

**Naming rule:** E2E specs are `*.spec.ts` under `e2e/`; Vitest keeps `*.test.tsx` under
`src/`. This is what keeps the two runners from colliding — see Phase 2.

## Phase 2 — Configure `playwright.config.ts`

**Files:** `frontend/playwright.config.ts`, `frontend/vite.config.ts`, `frontend/tsconfig.node.json`

The config is where every constraint above gets encoded.

1. **Base settings**
   - `testDir: "./e2e/specs"`, `outputDir: "./test-results"`.
   - `use.baseURL: process.env.E2E_BASE_URL ?? "https://localhost:5180"`.
   - **`use.ignoreHTTPSErrors: true`** — required for the basic-ssl cert.
   - `trace: "on-first-retry"`, `screenshot: "only-on-failure"`, `video: "retain-on-failure"`.
   - `fullyParallel: true`; `retries: process.env.CI ? 2 : 0`; `workers: process.env.CI ? 1 : undefined`.
   - `reporter`: `[["html", { open: "never" }], ["list"]]`, plus `["github"]` under CI.

2. **`webServer`** — let Playwright own the dev server so `just`-less runs work:
   ```ts
   webServer: {
     command: "npm run dev -- --host 127.0.0.1 --port 5180",
     url: "https://localhost:5180",
     ignoreHTTPSErrors: true,
     reuseExistingServer: !process.env.CI,
     timeout: 120_000,
   }
   ```
   `reuseExistingServer` means a dev who already has `just fe-dev` running is not fighting a
   second Vite instance.

3. **Projects** — three, deliberately:
   | Project | Browser | Blocking? | Purpose |
   |---------|---------|-----------|---------|
   | `chromium` | Desktop Chrome | yes | Main functional suite |
   | `firefox` | Desktop Firefox | yes | Cross-browser smoke subset (grep `@smoke`) |
   | `webgpu` | Chromium + GPU flags | **no** (`CI` allows failure / skips) | WebGPU-only specs |

   The `webgpu` project adds launch args `--enable-unsafe-webgpu` and
   `--enable-features=Vulkan` and sets `testMatch: /webgpu\/.*\.spec\.ts/`.

4. **Vitest exclusion.** Confirm `vite.config.ts`'s `test` block does not pick up `e2e/`.
   The default `include` is `**/*.{test,spec}.?(c|m)[jt]s?(x)`, which **would** match
   `e2e/**/*.spec.ts` — so add an explicit `test.exclude` for `e2e/**` (and keep `node_modules`,
   `dist`). This is the single most likely setup bug; verify it by running `just fe-test` and
   checking the file count is unchanged.

5. Add `playwright.config.ts` and `e2e/**` to `tsconfig.node.json` `include` so the config and
   specs type-check under `npm run build` / `tsc -b` without leaking `@playwright/test` types
   into the app's `tsconfig.app.json`.

## Phase 3 — Fixtures: auth, mock API, WebGPU

**Files:** `frontend/e2e/fixtures/*.ts`, `frontend/e2e/utils/*.ts`

1. **`fixtures/base.ts`** — a single extended `test` object that every spec imports instead of
   `@playwright/test` directly. This is the seam for adding fixtures later without touching specs.

2. **Auth via `storageState`** (`fixtures/auth.ts` + a `global.setup.ts` project dependency):
   - A setup project logs in once through the real UI at `/login`, then writes
     `e2e/.auth/user.json`. Authenticated specs declare `use: { storageState: "e2e/.auth/user.json" }`.
   - Requires a seeded test user. Add a backend management command or pytest-free script —
     e.g. `backend/apps/accounts/management/commands/seed_e2e_user.py` creating
     `e2e@example.com` from `E2E_USER_EMAIL` / `E2E_USER_PASSWORD` env vars, **guarded to
     refuse to run when `DEBUG` is false**. Expose as `just be-seed-e2e`.
   - Because email is the username field (`AUTH_USER_MODEL = "accounts.CustomUser"`), the
     login page object fills an email, not a username — assert against the real form labels.

3. **Mock-API fixture** (`fixtures/mockApi.ts`): wraps `page.route("**/api/**", …)` with
   fixture JSON so catalog/registry specs run with **no Django and no Postgres**. Reuse the
   response shapes already defined in `src/test/handlers.ts` — extract the shared fixture
   payloads into `e2e/fixtures/data/` and have both MSW and Playwright import them, so the
   two test layers cannot drift apart.

4. **WebGPU fixture** (`fixtures/webgpu.ts`): exposes `webgpuStatus`, obtained by evaluating
   the app's own `detectWebGPU()` result in-page (or a small inline `navigator.gpu`
   + `requestAdapter`/`requestDevice` probe). Provides `test.skip(status !== "ready", …)` so
   GPU specs self-skip on a GPU-less runner rather than failing.

5. **Page objects** (`e2e/pages/`): `LoginPage`, `AppShell` (sidebar/navbar/theme),
   `TrainingPage`, `TensorPage`, `PlaygroundPage`. Keep them thin — locators + a few actions,
   assertions stay in specs.

## Phase 4 — The initial spec suite

**Files:** `frontend/e2e/specs/**/*.spec.ts`

Written in dependency order; each spec is small and tagged.

1. **`smoke.spec.ts`** (`@smoke`, runs on chromium + firefox)
   - App loads at `/`, no console errors, shell renders.
   - Sidebar renders the taxonomy categories from `taskTaxonomy.ts`.
   - Theme toggle flips and persists across reload.

2. **`navigation.spec.ts`**
   - Each `REAL_ROUTES` entry navigates to its real route and renders that route's heading —
     drive this **from the taxonomy data itself** so a new task added to `taskTaxonomy.ts`
     is automatically covered.
   - A taxonomy entry *not* in `REAL_ROUTES` lands on the generic `tasks.$slug` placeholder.
   - Per-category expand state (`store/ui.ts`) survives navigation.
   - Deep-link + browser back/forward behave.

3. **`auth.spec.ts`** (needs the real backend; tagged `@backend`)
   - Login with valid credentials → redirect to the intended route.
   - Invalid credentials → visible error, stays on `/login`.
   - Signup flow at `/signup`.
   - Protected route while logged out → redirected to `/login`.
   - **Silent 401 refresh:** intercept and force a 401 on one `/api/**` call, assert the app
     recovers via the refresh path in `src/api/client.ts` without bouncing the user to login.
   - Logout clears tokens; reload stays logged out.

4. **`registry.spec.ts`** (mock-API fixture, no backend)
   - Model catalog lists cards from mocked `/api/registry/` data.
   - Empty state and API-error state both render something sane.

5. **`webgpu/capability.spec.ts`** (`webgpu` project)
   - On a `ready` device: the WebGPU status indicator reports supported.
   - **Degradation is asserted unconditionally** — override `navigator.gpu` to `undefined`
     via `page.addInitScript` and confirm the UI shows the unsupported state instead of
     crashing. This runs even on GPU-less CI, which is the point.

6. **`webgpu/inference.spec.ts`** (`webgpu` project, skips unless `ready`)
   - `/tensor`: run a matmul, assert the result matches a CPU reference computed in the spec
     (the repo rule "cross-check every new kernel against a CPU reference", enforced E2E).
   - `/training`: start a short linear-model run, assert loss decreases and the UI stays
     responsive (worker offload actually working, not blocking the main thread).
   - Generous timeouts — model/weight fetches cross the network.

**Flakiness rules for this suite** (state them in the docs, not just here): no
`waitForTimeout`; wait on locators/`expect` auto-retry. No dependence on test order. Every
network call the spec cares about is either mocked or explicitly awaited via
`page.waitForResponse`.

## Phase 5 — `just` recipes

**Files:** `justfile`

Add to the Frontend section, mirroring existing naming:

```just
# Install Playwright browsers (run once after fe-install)
fe-e2e-install:
    cd frontend && npx playwright install --with-deps chromium firefox

# Run Playwright end-to-end tests
fe-e2e:
    cd frontend && npm run test:e2e

# Run E2E tests in the interactive Playwright UI
fe-e2e-ui:
    cd frontend && npm run test:e2e:ui

# Run E2E tests in a headed browser (useful for WebGPU)
fe-e2e-headed:
    cd frontend && npm run test:e2e:headed

# Open the last Playwright HTML report
fe-e2e-report:
    cd frontend && npm run test:e2e:report

# Run only the WebGPU E2E project (needs a real GPU)
fe-e2e-webgpu:
    cd frontend && npx playwright test --project=webgpu --headed
```

Also extend `install` to mention `fe-e2e-install` (keep it a separate recipe — downloading
browsers should not be a silent side effect of `just install`).

## Phase 6 — Documentation

**Files:** `docs/guides/e2e-testing.md` (new), `CLAUDE.md`,
`.github/copilot-instructions.md`, `docs/explanations/architecture.md`, `README.md`

1. **`docs/guides/e2e-testing.md`** — the substantive doc: when to write E2E vs. Vitest,
   how to run each `just` recipe, the fixture/page-object layout, the HTTPS + self-signed
   cert gotcha, the WebGPU skip behaviour, how to seed the test user, and how to debug a
   failure with trace viewer (`npx playwright show-trace`).
2. **`CLAUDE.md`** — one line under *Frontend essentials* pointing at the new guide, plus the
   new `just fe-e2e*` commands in the commands block. **Mirror the same change into
   `.github/copilot-instructions.md`** — the repo requires these two stay in sync.
3. **`docs/explanations/architecture.md`** — note the test pyramid: pytest (backend),
   Vitest+MSW (frontend units), Playwright (browser E2E).
4. **`README.md`** — add `just fe-e2e` to the quick-start command list.

## Phase 7 — CI — **DROPPED**

**Files:** `.github/workflows/e2e.yml` (never created)

**Not doing this.** The repo has no `.github/workflows/`, and building CI from
scratch was explicitly taken out of scope. The sketch below is retained only so
that whoever picks it up later does not have to re-derive it.

- Job runs on `ubuntu-latest`, Node 22, `npm ci` in `frontend/`,
  `npx playwright install --with-deps chromium firefox`.
- Runs `--project=chromium --project=firefox` only. The **`webgpu` project is excluded** —
  GitHub-hosted runners have no GPU, so those specs would skip anyway and only add noise.
- Postgres service container + `just be-migrate` + `just be-seed-e2e` for the `@backend`-tagged
  specs; alternatively run with `--grep-invert @backend` in a first iteration and add the
  backend job once the seed command exists.
- Upload `playwright-report/` as an artifact on failure.

---

## Testing

How each phase is verified — nothing is "done" on inspection alone.

| Phase | Verification |
|-------|--------------|
| 1 | `npx playwright --version` resolves; `e2e/` exists; artefact dirs are gitignored (`git status` clean after a run). |
| 2 | **`just fe-test` reports the same test/file count as before** (proves Vitest still ignores `e2e/`). `npx playwright test --list` enumerates specs across all three projects. `just fe-build` (`tsc -b`) stays green with the config and specs in `tsconfig.node.json`. |
| 3 | Auth setup project produces `e2e/.auth/user.json`; an authenticated spec starts already logged in. Mock-API fixture serves a spec with the backend **stopped**. WebGPU fixture reports `ready` locally and skips cleanly when `navigator.gpu` is stubbed out. |
| 4 | `just fe-e2e` green locally with `just dev` running. Then run it **three consecutive times** — any spec that is not green all three times is flaky and gets fixed or quarantined, not retried away. `just fe-e2e-webgpu` green on a real GPU browser. |
| 5 | Each `just fe-e2e*` recipe runs from a clean shell; `just --list` shows them grouped under Frontend. |
| 6 | Links in the new guide resolve (no 404 relative paths); `CLAUDE.md` and `.github/copilot-instructions.md` diffs are equivalent. |
| 7 | CI run green on a PR; report artifact downloadable from a deliberately failing run. |

**Regression guard:** `just fe-test`, `just fe-lint`, and `just fe-build` must all be green
after every phase. Playwright is additive — if any of the three degrades, the phase is not done.

## What the build actually found

Recorded because each one contradicted an assumption in the plan above.

1. **`page.route("**/api/**")` breaks the app.** That glob also matches the dev
   server's own module URLs — `/src/api/client.ts` — so the mock replaced the
   app's Axios client with JSON and the app never booted. Every mocked spec failed
   with "element not found" until the matcher became
   `(url) => url.pathname.startsWith("/api/")`. This was the single biggest
   time sink and is now called out in the guide and both instruction files.

2. **There is no route protection in the app.** `__root.tsx` renders `AppLayout`
   for any non-public path regardless of auth state; nothing redirects a signed-out
   user away from `/home`. The planned "protected route → redirect to `/login`"
   spec was therefore **dropped rather than written against imagined behaviour**.
   Whether to add a guard is a product decision, not a testing one — flagged below.

3. **Native browser validation beats Zod.** The login email field is `type="email"`,
   so Chromium's constraint validation blocks submit before react-hook-form runs and
   the Zod message never renders. happy-dom does no constraint validation, which is
   why the unit tests *do* see it. The spec now asserts `validity.valid === false`,
   and a second spec covers the Zod path via the (unconstrained) password field.

4. **`CardTitle` is a `<div>`, not a heading.** `getByRole("heading")` finds no
   card titles. Specs select `[data-slot="card-title"]` instead.

5. **Selector collision:** `getByRole("button", { name: "Compute" })` also matches
   the sidebar's **Compute**r Vision category. Page-content locators are scoped to
   `page.locator("main")`.

6. **Playwright's headless build ships no WebGPU at all** — `navigator.gpu` is
   absent, not merely adapter-less, even with `--enable-unsafe-webgpu`. The `webgpu`
   project needs `channel: "chromium"`. On this machine it still reports
   `unsupported` because there is no `/dev/dri`.

7. **A pre-existing app warning surfaced.** `HeroBanner` renders
   `<Button render={<Link/>} />`, so Base UI's default `nativeButton={true}` lands
   on an `<a>` and it logs an accessibility warning twice on `/`. This is a real
   (small) bug the E2E suite caught. It was **not fixed** — out of scope for
   "set up Playwright" — but it is allowlisted in `smoke.spec.ts`'s
   `KNOWN_CONSOLE_ISSUES` with the fix location named, so it stays visible.

8. **The dev server intermittently serves a page that never mounts.** Vite
   full-reloads (`main.tsx?t=…`) while re-optimising deps, and the reloaded
   document can beat the optimiser — leaving an empty `#root` that never
   recovers. Roughly one navigation in four. Not an app bug (production has no
   optimiser), but it made the suite flaky in a way that only showed up on a
   *cold* server, which is exactly what CI would hit. `e2e/fixtures/base.ts` now
   wraps `page.goto` to verify the app mounted and reload if not. This is the
   reason specs must import `test` from `fixtures/base`, not `@playwright/test`.

9. **The saved-session path was cwd-relative, and that bit.** `global.setup.ts`
   originally wrote to the literal string `"e2e/.auth/user.json"`, which Playwright
   resolves against `process.cwd()` rather than the config directory. A run
   launched from the repo root therefore created a stray `<repo>/e2e/.auth/` and
   the authenticated specs would then not find their session. `e2e/fixtures/paths.ts`
   now derives an absolute `AUTH_FILE` from `import.meta.url` (Playwright loads
   these as ESM here — `__dirname` is undefined, which is worth knowing before
   reaching for it). Separately, running Playwright from the repo root does not
   work at all in this monorepo: `npx` resolves a different `@playwright/test`
   than the specs import, and collection fails with "did not expect test() to be
   called here". Always run from `frontend/`, which every `just` recipe and npm
   script already does.

10. **The `setup` project ran even with no backend.** `grepInvert` filters by test
    *title*, and the setup test is called "authenticate" — not `@backend` — so
    listing the project unconditionally meant plain `just fe-e2e` always attempted
    a real login and failed. Caught only when `just fe-e2e` was finally run as
    itself; every earlier check had passed explicit `--project` flags, which
    silently skipped the broken project. **Lesson: verify the command users will
    actually type, not a convenient subset of it.** The project is now registered
    only when `E2E_BACKEND` is set.

11. **Config details that differed from the plan.** E2E types went into a new
   `tsconfig.e2e.json` (project-referenced from `tsconfig.json`) rather than
   `tsconfig.node.json`, because the specs import app modules and need DOM + JSX
   libs. `eslint.config.js` needed an `e2e/**` override: Playwright's fixture
   callback is named `use`, which `react-hooks/rules-of-hooks` mistakes for React's
   `use` hook.

## Risks & open questions

1. **Vitest/Playwright glob collision** — the highest-probability failure. Mitigated by the
   explicit `test.exclude` in Phase 2.4 and verified by test-count comparison.
2. **Self-signed cert** — `ignoreHTTPSErrors` covers the browser, but the `webServer` health
   check needs it too (Phase 2.2). Symptom if missed: Playwright hangs 120s then times out
   waiting for the server it can already reach.
3. **WebGPU on CI** — accepted and designed around: GPU specs skip, degradation specs don't.
4. **Test-user seeding** — needs a backend command that does not exist yet. Until Phase 3.2
   lands, auth specs are tagged `@backend` and excluded from the default run.
5. **Resolved:** the default `just fe-e2e` is **mock-only**, so E2E runs with nothing but
   Node installed. `just fe-e2e-full` sets `E2E_BACKEND=1` and seeds the user first.
6. **Resolved — the test user.** `backend/apps/accounts/management/commands/seed_e2e_user.py`
   exists, refuses to run with `DEBUG=False`, and is wired up as `just be-seed-e2e`. Verified
   end to end: seed → real login → JWT issued → `@backend` specs green.
7. **Closed — CI.** Phase 7 dropped from scope by decision.
8. **Blocked — Firefox system deps.** `sudo npx playwright install-deps firefox` (or
   `sudo apt-get install libgtk-3-0t64`) must be run by a human with a terminal; a
   non-interactive session cannot answer the sudo password prompt. The `firefox`
   project is configured and listed, just never yet executed on this machine.
9. **Open — no auth guard on routes.** See finding 2 above. `__root.tsx` renders the app
   shell for any non-public path regardless of auth state. If a guard is wanted, the
   redirect specs are a few lines to add once the behaviour exists — this is a product
   decision, not a testing one.
