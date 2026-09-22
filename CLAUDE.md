# CLAUDE.md

Guidance for Claude Code (and other AI assistants) working in this repository.

This file is the Claude-native counterpart to [`.github/copilot-instructions.md`](.github/copilot-instructions.md).
The two files describe the same conventions — keep them in sync. When a convention changes,
update both.

---

## What this repo is

**Model Playground** — a web app for running ML models (LLMs, computer vision, custom
networks) **directly in the browser on the user's GPU/CPU**. Two client-side inference
paths coexist: (1) a **raw-WebGPU runtime** (`src/webgpu/`, hand-written WGSL compute
shaders) for custom kernels and teaching demos, and (2) **Transformers.js / ONNX Runtime
Web** for running *pretrained* models (the audio, vision and multimodal tasks) in the UI.
It is a decoupled monorepo:

- **`backend/`** — Django REST Framework API (Python 3.13, PostgreSQL, Celery). Acts as a
  **model registry**: catalog metadata + inference-run records. It does **not** run inference.
- **`frontend/`** — React 19 SPA (TypeScript, Vite, TanStack Router + Query). Owns the
  **WebGPU runtime** in `src/webgpu/` — inference runs client-side in a Web Worker.

The backend exposes only `/api/` endpoints. The frontend consumes them over HTTP. Model
weights are fetched by the browser from a model host/CDN (`ModelCard.weights_url`), never
proxied through Django. The two halves share no code — the API contract is the only
interface between them.

This began as a generic Django+React template, so **keep shared infrastructure generic and
reusable.** Prefer documented conventions over clever one-offs. WebGPU inference is the
domain focus — see [`docs/explanations/webgpu-inference.md`](docs/explanations/webgpu-inference.md).

---

## Where to look first

| You need… | Read |
|-----------|------|
| Full conventions (backend + frontend) | [`.github/copilot-instructions.md`](.github/copilot-instructions.md) |
| System architecture & design decisions | [`docs/explanations/architecture.md`](docs/explanations/architecture.md) |
| **In-browser inference (WebGPU)** | [`docs/explanations/webgpu-inference.md`](docs/explanations/webgpu-inference.md) |
| **Adding a model (kernel + registry entry)** | [`docs/guides/adding-a-model.md`](docs/guides/adding-a-model.md) |
| **Building a task page (Select → Load → Run → Output)** | [`docs/standards/model-page-pattern.md`](docs/standards/model-page-pattern.md) |
| **Visualizing a model & its structure (UI standard)** | [`docs/standards/model-visualization.md`](docs/standards/model-visualization.md) |
| **The system panel (GPU/memory/CPU/storage readouts)** | [`docs/explanations/telemetry-panel.md`](docs/explanations/telemetry-panel.md) |
| **Driving a model page (SELECT→LOAD→RUN→OUTPUT)** | [`docs/standards/model-page-pattern.md`](docs/standards/model-page-pattern.md) |
| Auth flow (JWT) | [`docs/explanations/auth-flow.md`](docs/explanations/auth-flow.md) |
| API endpoints & request/response shapes | [`docs/standards/api-contracts.md`](docs/standards/api-contracts.md) |
| Local dev setup | [`docs/guides/local-setup.md`](docs/guides/local-setup.md) |
| **Deploying it** (single origin, TLS, compose stack) | [`docs/guides/deployment.md`](docs/guides/deployment.md) |
| **Git guardrails & permission config** | [`docs/guides/ai-guardrails.md`](docs/guides/ai-guardrails.md) |
| **End-to-end tests (Playwright)** | [`docs/guides/e2e-testing.md`](docs/guides/e2e-testing.md) |
| **A model with no Transformers.js task (bare ONNX)** | [`docs/guides/adding-a-model.md`](docs/guides/adding-a-model.md) §9 |
| **When `pipeline()` is the wrong abstraction** (incl. chat-templated VLMs) | [`docs/guides/adding-a-model.md`](docs/guides/adding-a-model.md) §10 |
| Celery / async tasks | [`docs/guides/celery_setup.md`](docs/guides/celery_setup.md) |
| **Adding a task page (end-to-end procedure)** | [`docs/guides/adding-a-task-page.md`](docs/guides/adding-a-task-page.md) |
| Feature plans (phased) | **GitHub issues**, label [`plan`](https://github.com/bthek1/model_playground/issues?q=is%3Aissue+label%3Aplan) — open = active, closed = done |
| **Audio category roadmap** (complete, 6 of 6) | [`docs/roadmaps/audio.md`](docs/roadmaps/audio.md) |
| **Computer Vision roadmap** (14 of 20, plus the shared `src/vision/` module) | [`docs/roadmaps/vision.md`](docs/roadmaps/vision.md) |
| **Graph ML roadmap** (**complete, 4 of 4**; no checkpoint, pure WGSL) | [`docs/roadmaps/graph.md`](docs/roadmaps/graph.md) |
| **Multimodal roadmap** (2 of 9; VLMs, `q4f16`, streaming, document QA) | [`docs/roadmaps/multimodal.md`](docs/roadmaps/multimodal.md) |
| Roadmaps for categories not yet built | **GitHub issues**, label [`roadmap`](https://github.com/bthek1/model_playground/issues?q=is%3Aissue+label%3Aroadmap) — each graduates to `docs/roadmaps/` when its first route ships |

---

## Common commands

All workflows are wrapped in the [`justfile`](justfile). Run `just --list` to see everything.

```bash
just dev            # db + redis + celery + backend + frontend (full local stack)
just up             # everything via docker compose

# Backend
just be-dev         # makemigrations + migrate + runserver
just be-test        # pytest
just be-test-cov    # pytest with coverage
just be-lint        # ruff check
just be-fmt         # ruff format
just be-makemigrations [app]

# Frontend
just fe-dev         # vite dev server
just fe-build       # type-check + build
just fe-test        # vitest (unit/component)
just fe-e2e         # playwright end-to-end (mocked API, no backend needed)
just fe-e2e-slow    # @slow specs: real model downloads + real ONNX sessions (minutes)
just fe-e2e-enhance # @slow speech-enhancement specs (DeepFilterNet3, WASM + WebGPU)
just fe-e2e-vad     # @slow voice-activity-detection specs (Silero VAD, seconds)
just fe-e2e-vision  # @slow vision specs: real loads across all 14 routes (tens of minutes cold)
just fe-e2e-superres # @slow: Swin2SR vs a bicubic baseline, by PSNR
just fe-e2e-vision-one /pose   # one @slow vision route at a time
just fe-e2e-link    # @slow: link prediction, pinned by an AUC *band* (leakage pushes it up)
just fe-e2e-graphcls # @slow: graph classification, pinned above its majority baseline
just fe-e2e-vlm     # @slow: a real SmolVLM load + generation — the only chat-template guard (needs a GPU)
just fe-e2e-docvqa  # @slow: a real Donut load, pinned by a known answer on a known invoice (no GPU needed)
just fe-e2e-models  # check every model id (audio + vision + multimodal) resolves on the HF Hub (seconds)
just fe-e2e-install # download the playwright browsers (once)
just fe-e2e-ui      # playwright interactive UI
just fe-lint        # eslint

# Celery
just celery-up      # redis + worker + beat (docker)
just celery-worker  # run a worker locally for debugging
just flower         # Flower monitoring UI (port 5555)
```

Run backend commands from `backend/` with `uv run …` if you need them outside `just`.

---

## House rules

These mirror the "General Rules" and "Absolute Don'ts" in the Copilot instructions:

- **Docs travel with code.** A code change that affects behaviour, endpoints, or setup must
  update the relevant file under `docs/`. Add new endpoints to `docs/standards/api-contracts.md`.
- **Plans are GitHub issues, not files.** Any feature touching more than one file gets a plan
  first — phased, with a Testing section — opened as an issue with `gh issue create --label plan`
  (the plan is the issue body; see the template in the Copilot instructions). **Never add a plan
  markdown file to the repo.** Tick the phase checkboxes as work progresses, and **close the issue
  once its work has been merged into `main`** (`gh issue close <n> --comment "..."`) — the closed
  issue is the record. Reference the issue number in the commit message (`Closes #12`).
- **Never commit `.env` files.** `.env.example` is the source of truth for required vars.
- **Backend ↔ frontend communicate only via the API contract** — never mix their concerns.
- **All work lands on `develop`; `main` is what has shipped.** `develop` is the integration
  branch — commit the issue's work straight onto it. **Do not open a branch per issue**: a branch
  per issue means a merge per issue, and the issues in this repo are phased plans that touch the
  same files (the taxonomy, the model catalogue, the page-pattern shell) days apart, so the
  branches conflict with each other rather than with `main`. Check `git branch --show-current`
  before committing; if it isn't `develop`, switch (`git switch develop`) rather than branching.
  Create a `<type>/<topic>` branch **only when the user asks for one** — a spike to be thrown
  away, or work that has to be reviewed as a PR in its own right — and branch it from `develop`.
- **Merging into `main` is the completion step of an issue, not a step inside it.** When an
  issue's phases are all ticked and its tests are green: `git switch main`, merge `develop`
  (`git merge --no-ff develop`, so the issue's commits stay legible as one landing), push, then
  `gh issue close <n>`. Merge only whole issues — `develop` holding half an issue is why the
  merge waits, not a reason to cherry-pick. Never commit directly on `main`.
- **Commits are cheap; pushes are not.** Committing and switching run unattended — they are
  reversible, and `git reflog` recovers almost anything local. **Ask before anything
  outward-facing or unrecoverable:** `git push`, `git rebase`/`git merge` (the merge into `main`
  included), `gh pr create`/`gh pr merge`, create/edit/close a GitHub issue (`gh issue …`),
  `docker compose down -v`, deleting migrations, or modifying shared `.env` files. **Never**
  force-push, `git reset --hard`, `git clean`, `git branch -D`, or `git checkout .` — those are
  denied outright in `.claude/settings.json`. See
  [`docs/guides/ai-guardrails.md`](docs/guides/ai-guardrails.md) and the full list in the Copilot
  instructions.

### Deployment essentials

- **Two compose files.** `docker-compose.yml` is development (bind mounts, dev
  servers, throwaway credentials — never deploy it); `docker-compose.prod.yml` is
  production (built images, secrets from a root `.env`, only Caddy publishes a
  port). `frontend/Dockerfile.dev` is the dev server; `frontend/Dockerfile` builds
  and serves with nginx.
- **The deployment is single-origin over HTTPS, and both halves are requirements.**
  `src/api/client.ts` ships an empty base URL so the app calls `/api` on its own
  origin — the Vite proxy does that in dev and **nginx does it in production**;
  without it the design inverts into cross-origin calls that fail on mixed
  content, CORS and Local Network Access. And `navigator.gpu` only exists in a
  secure context, so an HTTP deploy makes every model page report `unsupported` on
  hardware that works.
- **`X-Forwarded-Proto` is a chain** — Caddy → nginx → `SECURE_PROXY_SSL_HEADER`.
  Break it and `/admin/` rejects every POST on CSRF, or `SECURE_SSL_REDIRECT`
  loops. The prod compose sets `SECURE_SSL_REDIRECT=False` because Caddy already
  redirects at the edge.
- **Migrations are opt-in** (`RUN_MIGRATIONS=1`, on the `migrate` service only) —
  replicas racing `migrate` is a real failure mode. **WhiteNoise** serves Django's
  static files from inside the backend container, collected at build time, or
  `/admin/` renders unstyled at `DEBUG=False`.
- **nginx serves `.wasm` as `application/wasm`** (ONNX Runtime's streaming
  compilation refuses anything else) and keeps `index.html` `no-cache` while
  `/assets/` is `immutable`. Model weights never touch the server — any CSP must
  not block the HF CDN.
- **`npm run check:bundle`** budgets the entry chunk and keeps `echarts` behind its
  lazy wrapper; it runs inside the frontend image build, so a leak fails the image.
- CI (`.github/workflows/ci.yml`) runs lint, tests, the build, the bundle budget,
  the mocked Playwright suite and both image builds. It does **not** run the
  `@slow` specs — those still need `just fe-e2e-slow` by hand.
  See [`docs/guides/deployment.md`](docs/guides/deployment.md).

### Backend essentials

- Class-based views; serializers in `serializers.py`, business logic in `services.py` (never in views).
- Models use UUID primary keys. Use `get_user_model()` — never import `User` directly
  (`AUTH_USER_MODEL = "accounts.CustomUser"`, email is the username field).
- Split settings: `core/settings/{base,dev,prod,test}.py`. Config via `django-environ`.
- Always `makemigrations` after model changes. Avoid N+1 with `select_related`/`prefetch_related`.
- **Registry app (`apps/registry/`)** is the model catalog: `ModelCard` + `InferenceRun` under
  `/api/registry/` (DRF ViewSets + router). It stores metadata only — **never run inference on the
  backend.** Public read, auth-gated writes; users see only their own runs.

### Frontend essentials

- React 19 + TypeScript ~6.0 + Vite 8 (dev server on `:5180`). Functional components only.
- All API calls go through `src/api/client.ts` (Axios + JWT with silent 401 refresh). Its base URL is **empty by default** — requests hit `/api` on the page's own origin and the Vite dev server proxies them to `VITE_API_PROXY_TARGET`. This keeps LAN/HTTPS access working: a page served from `https://192.168.x.x:5180` calling `http://localhost:8006` directly is blocked by mixed content, CORS, and Local Network Access.
- Server state lives in TanStack Query; global UI flags in Zustand + Immer (`src/store/`, one file per concern) — never put server data in Zustand.
- The sidebar is **taxonomy-driven**: categories/tasks live in `components/layout/taskTaxonomy.ts` (data), rendered by `components/layout/Sidebar.tsx`. To add a task, add a data entry — map it to a real route via `REAL_ROUTES` (e.g. the Theory tools Linear Model Training → `/training` and Tensor Arithmetic → `/tensor`), else it falls through to the generic `routes/tasks.$slug.tsx` placeholder. Per-category expand state is in `store/ui.ts`.
- Forms use React Hook Form + Zod schemas (`src/schemas/`, one file per domain).
- Styling is Tailwind v4 (CSS-first, no config file) + shadcn/ui in the **`base-nova`** style, built on **`@base-ui/react`** primitives (NOT Radix). Add components with `npx shadcn@latest add <component>`.
- Charts: ECharts, always via the lazy `src/components/charts/EChart.tsx` wrapper (`echarts` is heavy — keep it code-split). Render Markdown/LLM output with `src/components/Markdown.tsx` (`react-markdown` + `remark-gfm`).
- **Every task page is the same pipeline: Select → Load → Run → Output** — pick a model, load its weights, run it on an input, show the result. This is the standard in [`docs/standards/model-page-pattern.md`](docs/standards/model-page-pattern.md); read it before adding a task route. Two state machines, kept orthogonal: **load** (`idle → loading → ready | error`, with `progress` as a self-loop, `retry(overrides?)` out of `error` and `cancel()` back to `idle`) and **run** (id-correlated requests, `running` derived from an in-flight *count*, never a boolean). The plumbing lives once in `model/useModelWorker.ts`; task hooks (`useTts`, `useAsr`, `usePipeline`, `useAudioClassifier`, `useEnhance`, `useVad`) are thin wrappers returning the same contract — `status`/`idle`/`loading`/`ready`/`progress`/`loadProgress`/`loadedInMs`/`backend`/`load`/`retry`/`cancel`/`run`/`running`/`result`/`error`. The shell is `components/model/` (`ModelPage` + `ModelPicker`/`ModelStatus`/`InputPanel`/`OutputPanel`, plus `DeviceStatus` for pages that probe a GPU instead of downloading weights). Never re-derive the pending map, the teardown, or a bespoke page shape. Nothing downloads until the user asks: `idle` is the default and the size estimate + large-model warning are shown first — quoted **once**, by `ModelPicker`; `ModelStatus` must not repeat the number. Errors render in the slot that produced them.
- **Two buttons on a task page spend anything, and they are LOAD and GENERATE.** SELECT and INPUT are *choices* — free, reversible, and they commit to nothing. **Only LOAD loads:** not arriving at the page, not switching models, not returning to a page you used before, and **not a cache hit** (cached weights make the click cheap, not unnecessary — they still cost memory, a GPU device and a warm-up). **Only GENERATE runs:** picking a sample, dropping a file, finishing a recording, placing a point on a picture, or editing a prompt/label/template beside the input all land in INPUT and stop there. A row of five samples that each ran the model meant browsing cost five inferences; that is the failure mode. The corollary: **the input sources are not gated on `ready`** — only the GENERATE trigger is. A control that merely re-reads a result in hand (`/vad`'s threshold, detection's floor, segmentation's opacity) still re-derives on the main thread without a press, because it spends nothing.
- **A model page survives a refresh by restoring the selection, not the session and not the load.** A Worker cannot outlive a page load. `store/models.ts` persists the selected model and *only* that — it declares `partialize` so a stale `autoResume` key from an older build cannot revive the auto-resume behaviour. `model/cache.ts` still probes Transformers.js's `transformers-cache` bucket (answering "not cached" on any failure), but only to change the LOAD button's words to "Load model (cached)" — an informed click, never an absent one. The LOAD slot's bar reports the **aggregate** from `model/progress.ts` — monotonic percent by bytes, indeterminate until a size is known, warm-up as its own phase, no ETA — never a raw per-file `progress_callback` event, which restarts at zero for each of a model's 4–8 files.
- **The input is held state, not an event.** `hooks/useImagePick.ts` and `hooks/useAudioPick.ts` (with `components/vision/ImageSourcePanel.tsx` / `components/audio/AudioSourcePanel.tsx`) own the decode and hand the run a copy — `useAudioPick`'s `take()` returns `clip.audio.slice()` because every audio worker detaches the buffer it is given, and `toPayload` copies by default for the same reason. Neither pick hook takes an "on picked" callback; `useImagePick` used to, thirteen routes fired an inference from it, and the parameter is **gone** rather than unused so it cannot return one route at a time. Pressing GENERATE twice, or changing a parameter and pressing again, must cost one decode and two inferences. OUTPUT renders the frame/clip captured **inside** the run, never the input currently held, or a new pick restyles the previous result.
- **Every task page is the same four-stage pipeline** — SELECT → LOAD → RUN → OUTPUT — specified in
  [`docs/standards/model-page-pattern.md`](docs/standards/model-page-pattern.md). The shared worker
  plumbing lives in **`src/model/useModelWorker.ts`**: worker lifecycle keyed on a `key` string, the
  id-correlated pending table, and the two state machines (A: `idle → loading → ready | error` with
  `retry()`; B: per-request, `running` is an **inflight count**, never a boolean — a boolean lies the
  moment two requests overlap). Task hooks (`useAsr`/`useTts`/`usePipeline`) are thin typed wrappers;
  don't reimplement the plumbing. Worker messages share one envelope (`ModelRequest`/`ModelResponse`
  in `src/model/types.ts`) with only the payload per task.
- **The page is horizontal, not one column** (page-pattern §4/§4a). SELECT + LOAD collapse into a
  ~20rem **setup rail**; RUN + OUTPUT sit side by side as the **workbench**, so a result never lands
  below the fold and there is no scroll between the input being edited and the output being judged.
  One column below `md`, a horizontal setup strip over side-by-side work columns at `md…xl`, all
  three at `xl`. Placement is by CSS-grid **area** — **DOM order stays 1→2→3→4 at every breakpoint**,
  so never reorder the source to move a band. Both work columns need `min-h-0 min-w-0` or the inner
  scroll won't engage and a wide result pushes the page sideways; the height clamp starts at `md` so
  a phone is never trapped in a scroll container. The transport row is `sticky bottom-0`, never
  `mt-auto`-pinned — pinning opens a chasm on any task whose input is short.
- **Testing a task page** has a fixed shape (see §8 of the page-pattern standard, which lists the `data-testid` contract — `slot-1`…`slot-4`, `output-panel`, `output-empty`, `model-ready`, `error-note` — shared by Vitest and Playwright). Every task page test asserts: nothing downloads on mount (hook called with **no `autoLoad` argument** — its default is `idle` — **and** `load` not called); a refresh restores the selection and leaves the page `idle` **even with the weights cached**; **choosing an input runs nothing** (pick a sample while `ready`, assert `run` was not called and `output-empty` is still there, then press the trigger and assert it was — the single most valuable assertion here, because an input that silently starts an inference looks like a working page); **the input survives its run** (press twice, assert one `fetch`/`decodeToMono`/`recordMic` against two `run` calls); `load`/`retry` fire from the LOAD slot; all four slots render with `output-empty` before any run; the GENERATE trigger is disabled until `ready` while the input sources are not; and errors land in the slot that produced them. A band is a labelled `region`, so `getByLabelText` can match both a band and a field inside it — query by role, or by the field's own control. **Layout is never a Vitest assertion** — jsdom has no geometry, so route tests check presence and order only, and the arrangement (side-by-side columns, output above the fold, no horizontal overflow, no transport chasm) is asserted in `e2e/specs/model-page.spec.ts`.
- **Visualizing models & their structure** follows [`docs/standards/model-visualization.md`](docs/standards/model-visualization.md) — a shared grammar of stage/arrow schematics, canvas weight/activation heatmaps (diverging red=+/blue=−, alpha=magnitude), param chips, theme-token colors, and lazy charts. The primitives live in `components/viz/` (`schematic.tsx`: Stage/Arrow/ParamChip · `heatmap.tsx`: HeatmapTile/DivergingLegend · `PanZoom.tsx`: a drag/wheel/fit surface for any child, with an optional `onScaleChange` for a canvas that must keep its strokes a constant size on screen); the Training route (`components/training/`) and Tensor route (`routes/tensor.tsx`) are the reference callers, and the three graph routes use `PanZoom`. Reuse those primitives; don't invent parallel ones.
- **The system panel reports load and capacity, never utilisation — and samples nothing while it is shut.** `src/telemetry/` (samplers + the 1 Hz loop), `components/telemetry/` (the cards) and `components/layout/RightPanel.tsx` (docked `aside` at `lg`, a Sheet below it, `Alt+Shift+M`). A browser exposes **no** host CPU percent, **no** GPU utilisation and **no** VRAM, so every metric is a `Metric<T>` — `{ status: "ok", value }` or `{ status: "unavailable", reason }`, never a zero standing in for "unknown" (`telemetry/types.ts`), and each card carries one line saying what its number *is*. The observer-effect rules are the acceptance criteria: one interval, only while open **and** the tab visible; samples in fixed-capacity ring buffers behind refs (never Zustand, never TanStack Query); the cache walk every 10th tick; a slow tick **skipped, not queued**; closing clears the history rather than drawing a chart across the gap. Sparklines are inline SVG per the viz standard §5 — no `echarts` in this panel. GPU bytes are a **ledger, not a probe** (`webgpu/allocations.ts`): free buffers with `releaseBuffer()` instead of `.destroy()`, and because every kernel runs in a worker realm the worker publishes its ledger to the page over a `MessagePort` handed out by `createWebGPUWorker()` — a `BroadcastChannel` is origin-wide and would fold a second tab's allocations into this page's total. `useModelWorker` reports its inflight count and download bytes to `telemetry/activity.ts` (nothing outside the hook can observe either); the `ModelRequest`/`ModelResponse` envelope does **not** grow a telemetry variant. See [`docs/explanations/telemetry-panel.md`](docs/explanations/telemetry-panel.md).
- Tests: Vitest + Testing Library + MSW (`src/test/server.ts`, `handlers.ts`). `src/test/setup.ts` also polyfills `localStorage` because Node ≥25 ships a stub that shadows the DOM env's.
- **End-to-end tests are Playwright** (`e2e/`), covering what happy-dom can't: routing/app shell, real-browser auth, and WebGPU. Default run is fully mocked (no backend); `@backend`-tagged specs need `just be-seed-e2e`, and
  `@slow`-tagged specs (real Hugging Face downloads + real ONNX sessions) need `just fe-e2e-slow` (or the per-route `fe-e2e-enhance` / `fe-e2e-vad` / `fe-e2e-vision` / `fe-e2e-link` / `fe-e2e-graphcls`; `fe-e2e-models` is the seconds-long id + dtype check in `model-ids.spec.ts`, across every modality). Import `test`/`expect` from `e2e/fixtures/base`, not `@playwright/test`. Shared page-object verbs (`load`, `waitForReady`, `backend`, `sizeNote`, `blockModelDownloads`) live on `ModelPageObject`, not on a modality subclass. Two traps: never `page.route("**/api/**")` (it also matches `/src/api/*` module URLs and stops the app booting), and keep the `test.include`/`test.exclude` block in `vite.config.ts` pinned to `src/` or Vitest swallows the E2E specs. **A `@slow` spec asserts a known label on a known input** — "a result appeared" would have passed while a quantized model called a tiger a snake. See [`docs/guides/e2e-testing.md`](docs/guides/e2e-testing.md).

### WebGPU essentials (`src/webgpu/`)

- **Raw WebGPU only — no ML framework** *in `src/webgpu/`*. Custom-kernel models are WGSL compute
  shaders in `webgpu/shaders/` (imported as strings via Vite `?raw`). Types come from `@webgpu/types`
  (in `tsconfig.app.json` `types`). This rule scopes to the hand-written runtime — running *pretrained*
  models in the UI (audio ASR/TTS/classification, vision classification, etc.) may use
  **Transformers.js / ONNX Runtime Web**,
  which run the same HF checkpoints on WebGPU or WASM. See
  [`docs/guides/adding-a-model.md`](docs/guides/adding-a-model.md) §8.
- The pipeline: `getGPUDevice()` (memoised, device-lost aware) → `createComputePipeline(wgsl)` →
  storage/uniform buffers (`buffers.ts`) → `dispatchWorkgroups` → `readBackFloat32`. `runtime.ts` is
  the reference (`runMatmul`).
- **Heavy compute runs in the Web Worker** (`worker.ts`), driven from the main thread via
  `workerClient.ts` (transfers input/output `ArrayBuffer`s, correlates by request id). Never block the UI thread.
- `detectWebGPU()` never throws — returns `unsupported` / `no-adapter` / `no-device` / `ready`.
  `ready` means a `GPUDevice` was actually acquired (it calls `requestDevice()`), not just that an
  adapter exists. UI must degrade gracefully. Cross-check every new kernel against a CPU reference.
- **A false `unsupported` is common:** `navigator.gpu` needs a **secure context** (HTTPS or `localhost`),
  so the dev server runs over HTTPS and a plain-HTTP LAN origin hides WebGPU; **Firefox on Linux/macOS**
  also needs `dom.webgpu.enabled` in `about:config`. See [`docs/explanations/webgpu-inference.md`](docs/explanations/webgpu-inference.md).
- To add a model: write the kernel + register a `ModelCard`. See [`docs/guides/adding-a-model.md`](docs/guides/adding-a-model.md).


### In-browser graph learning (`/graph`, `/link-prediction`, `/graph-classification`)

The other carve-out, and the opposite one: there is **no checkpoint at all**. A graph
neural network is one sparse gather repeated a few times, so the model is written as WGSL
and trained in the tab. `lib/cora.ts` + `lib/data/cora.bin` (the dataset),
`lib/graphLayout.ts`, `webgpu/gnn.ts` + `webgpu/gat.ts` (the model and its backward pass),
`webgpu/shaders/gnn_aggregate.wgsl` + `webgpu/gnnRuntime.ts` (the kernel), and
`webgpu/graphSession.ts` (the worker side). See [`docs/roadmaps/graph.md`](docs/roadmaps/graph.md).

- **Message passing is one scaled gather**, `out[i] = α_i Σ_{j ∈ N(i) ∪ {i}} β_j x[j]`, and
  GCN / GraphSAGE / GIN are just a choice of the two scale vectors. Three invariants, all
  of which fail *silently*: the **self-loop is not stored** and is added by the kernel
  (`A_hat = A + I`; storing it double-counts), the **graph must be symmetric** because that
  is what makes the backward pass's `Âᵀ` the same kernel with α and β swapped, and you must
  **project before you gather** (`Â(XW)`, never `(ÂX)W` — 2708×16 versus 2708×1433).
  `lib/cora.test.ts` asserts the first two over the real committed binary.
- **GAT is not a scale vector, and the roadmap was wrong to say it was.** Its coefficients
  are learned per *edge* from the features, the layer owns two vectors, and the gradient
  flows into the features twice. It goes through the `Propagator` seam instead, and its
  gather deliberately **does not** use the shader: it is O(|E|·d) ≈ 0.2 ms next to a
  2708×1433×16 projection that already runs on the GPU. The page says which half runs where.
- **A wrong aggregation still produces a falling loss and a plausible accuracy curve.** The
  guard is a **finite-difference gradient check** per architecture (`webgpu/gnn.test.ts`),
  plus a GPU-vs-CPU kernel cross-check in `e2e/specs/webgpu/graph.spec.ts`. Two things that
  check needed: it runs at a **generic point** (zero-init biases put some preactivations
  exactly on ReLU's kink, where a central difference reports half the gradient), and it
  needs a **dropout pass**, because the masked input transpose is unreachable without one.
- **Cora is bundled sparse-encoded, 161 KB.** Dense f32 features are 15.5 MB; the matrix is
  1.27% dense with every stored value 1. `scripts/prepare-cora.mjs` rebuilds it and prints
  its header. Input dropout is worth several points (0.754 → 0.776) and is affordable only
  because `prepareInput` records where the 49 216 nonzeros land in **both** layouts —
  re-masking and re-transposing 3.9 M elements per epoch would cost more than the model.
- **The layout is the expensive part, not the model** (~900 ms vs ~2 s). Computed once in
  the worker and never recomputed: changing the architecture or the depth re-trains, and
  must never re-lay-out. The features never cross `postMessage` — the worker fetches and
  decodes the dataset itself and sends back only what the canvas draws (~80 KB).
- **Oversmoothing needs a number, and the obvious one is wrong.** Mean pairwise cosine over
  all node pairs does *not* move monotonically with depth on real Cora; similarity between
  **adjacent** nodes does (0.947 → 0.978 as accuracy falls 0.78 → 0.52). Rows are normalised
  first so shrinkage is not mistaken for smoothing. **GIN's collapse is a different failure**
  — unnormalised sum overflows and ReLU zeroes everything — so `deadFraction` is reported
  separately and the page refuses to call it oversmoothing.
- **A gradient check does not pin a forward pass, and a canvas hides its geometry.** Two
  gaps worth knowing before writing the next kernel or the next visualization: GAT's
  finite-difference checks pass even if the softmax is normalised over the wrong set, so
  `gat.test.ts` asserts the property (every output is a convex combination of its closed
  neighbourhood) and calibrates it by zeroing the attention vectors, which must collapse the
  layer to an exact mean. And happy-dom gives a canvas **no 2D context and no
  `ResizeObserver`**, so a component test reaches only the guarded early-returns — pull the
  geometry out as a pure function (`GraphCanvas.layoutToPixels`) or it is untested. Doing that
  immediately surfaced a drawing centred and then shifted again by half the padding.
- **`e2e/specs/webgpu/` was skipping everywhere, on every machine.** The fixture probed
  `navigator.gpu` from `about:blank`, whose opaque origin is not a secure context, so the
  probe always said `unsupported`. It now probes from a served origin, and the webgpu
  project passes `--enable-unsafe-swiftshader` so a runner with no `/dev/dri` still executes
  real WGSL. If you add a kernel, that is what will check it.
- **Link prediction (`/link-prediction`) is the second page on this path, and its one bug fails
  *upward*.** Same Cora, same kernels, a decoder instead of a classifier
  (`score(u,v) = z_u · z_v` through a sigmoid) — but the supervision is on **edges**, so the
  held-out citations must leave the **graph**, not just the loss. An encoder still allowed to
  aggregate over an edge it is later scored on has already averaged the endpoints together and
  answers from memory: test AUC goes to ~0.99 and the page looks *better*. `lib/edgeSplit.ts` owns
  that, and three consequences are silent on their own — **both directed copies** go (the backward
  pass's `Âᵀ` needs symmetry), the **degrees are recomputed** (`archScales` builds `D^-1/2` from
  them, so a stale degree leaks the edge's existence into the normalisation), and an edge whose
  removal would **isolate** an endpoint is kept instead. Negatives are rejected against the **full**
  graph: a pair that is really a held-out citation is mislabelled, not negative.
- **The decoder's gradient is the half-right kind.** A pair scatters into **both** endpoints' rows;
  updating only one still produces a falling loss and a rising AUC. `linkPredictor.test.ts` pins it
  with finite differences over encoder *and* decoder, per architecture.
- **The metric is AUC (with AP beside it), and the E2E assertion is a band, not a floor** —
  above 0.85, below 0.985, because leakage is the failure and a floor would pass more comfortably
  *with* the bug (`just fe-e2e-link`).
- **Do not inherit a sibling page's hyperparameters.** `/graph` regularises hard for 140 labelled
  nodes; this page has ~4500 supervised edges and the same settings cost it 0.19 of AUC —
  measured 0.737 against 0.925 on the same split. Reuse that looks like a decision is often an
  inheritance; run it.
- **Graph classification (`/graph-classification`) completes the category, and its lesson is a
  *number*, not a kernel.** 1113 PROTEINS graphs, one label each. PROTEINS is 663/450, so a
  classifier that ignores the molecule scores **0.598** — and the class prior is the easiest
  thing in the dataset to learn, so that is precisely what a broken readout or a mis-built
  union produces. The **majority baseline is rendered beside every accuracy**, travels *inside*
  the metrics so the two can never come from different splits, and the E2E assertion is
  "above the baseline", not above chance (`just fe-e2e-graphcls`). A metric owes its null
  model on screen wherever one exists.
- **The whole dataset is one graph.** Batching graph classification is a block-diagonal
  **disjoint union** plus a node→graph vector — 43 471 nodes, smaller than the Cora matmul
  `/graph` already runs — so there is no mini-batching and no second code path. The bug a
  union can have is an edge leaking into the next graph's node range: two proteins share a
  neighbourhood, the model trains happily, nothing downstream notices. `lib/proteins.test.ts`
  asserts it directly, and `buildUnion` **checks** symmetry and self-loop-freedom per graph
  rather than assuming them.
- **The readout needs no parameters**, because mean pooling and a linear layer commute
  (`mean(W·h) = W·mean(h)`): read out the *logits* and §3.4's entire new arithmetic is one
  reduction and its adjoint. Sum and mean differ by the `1/n_g` that makes one size-invariant
  — invisible in a loss curve — so the finite-difference check runs over four architectures ×
  both modes, plus a model-free adjoint identity.
- **It is the only route in the repo that downloads a *dataset*** (2.06 MB from the Hub,
  cached in IndexedDB). A `curl -I` with no `Origin` header makes huggingface.co look like it
  refuses cross-origin reads; send the header and it echoes the caller's origin.

### In-browser pretrained models (`src/audio/`, `src/vision/`)

The carve-out from the raw-WebGPU rule: pretrained HF checkpoints (audio ASR/TTS/classification) run
through **Transformers.js** (`@huggingface/transformers`, plus `kokoro-js` for TTS), and two tasks —
speech enhancement and voice activity detection — run on **`onnxruntime-web` directly**. Keep both out of `src/webgpu/` — the
runtimes never mix. See [`docs/guides/adding-a-model.md`](docs/guides/adding-a-model.md) §8
(Transformers.js) and §9 (a bare ONNX graph) for the full recipes.

- **One worker per modality, not per task.** Discriminative tasks share the generic
  `pipeline.worker.ts` (the task string travels in the `load` message). ASR keeps its own worker (it
  drives the real-time capture loop). `tts.worker.ts` owns the whole **text→audio** modality — Kokoro,
  MMS/SpeechT5 *and* MusicGen — because they all fit one `TtsSynthesizer` interface; a fourth worker
  would have bought nothing. Each engine (`asrEngine`/`pipelineEngine`/`ttsEngine`/`enhanceEngine`/`vadEngine`) is a pure,
  unit-testable message handler; the `*.worker.ts` file is a thin wrapper around it.
- **Every engine owes three behaviours:** *one model live at a time* (null the reference **first**,
  then dispose via `disposeQuietly` — a failed teardown must not leave a stale model live);
  *warm-up on load* (one throwaway inference before `ready`, posting `{ status: "warmup" }`, and never
  failing the load if it throws); and *never block the main thread*.
- **ASR precision is a deliberate special case.** `loadOpts()` = fp16 on WebGPU / q8 on WASM, but ASR
  uses **`asrLoadOpts()`**, which keeps the **decoder at fp32 on WASM**. The quantized
  Whisper/Moonshine decoders cannot open a session on the ONNX Runtime bundled with
  `@huggingface/transformers` 4.2.0 (`qdq_actions.cc:137 … Missing required scale`), so a uniform q8
  breaks the universal fallback entirely. Don't "simplify" it away; re-test when ORT updates.
- **Check a model id against the Hub before shipping it.** Two entries once pointed at
  `onnx-community/*` repos that don't exist (401). `just fe-e2e-models` verifies all of them in seconds.
- **ASR timestamps are take-relative.** The live loop re-transcribes only the tail 30 s, so the
  model's own timestamps restart at 0 on a longer take; `useLiveAsr`'s `shiftChunks()` offsets them
  by the window start before the route renders `m:ss`. Don't render `chunks` straight from the worker.
- **The backend probe and size guardrail are shared, and live in `src/model/`** (`backend.ts`,
  `size.ts`) — they started under `audio/`, and moved rather than being copied when vision arrived.
  Never write a second probe.
- **Size-before-load guardrail:** every catalogue entry carries `params` (millions); `model/size.ts` +
  `components/model/ModelPicker.tsx` quote the download for both backends and warn past
  `LARGE_MODEL_BYTES`. Supply measured `bytes` when the params estimate would mislead — ASR's fp32
  decoder makes the WASM download ~3x the estimate.
- **Heavy/experimental models are gated, not auto-loaded.** `/text-to-audio` (MusicGen, 571 MB q8 /
  ~1 GB fp16) states its size and speed cost and downloads nothing until the user opts in — an E2E
  spec asserts zero Hub requests before the click. Use the same pattern for anything this heavy.
  MusicGen also needs `MusicgenForConditionalGeneration` directly: the `text-to-audio` **pipeline**
  throws "Missing the following inputs: input_ids" on 4.2.0.
- **Audio-to-Audio (`src/audio/enhance/`) has no Transformers.js path at all.** DeepFilterNet3
  publishes the neural graph only — normalised ERB/spectral features in, an ERB mask plus complex
  deep-filter coefficients out — so the STFT, ERB filterbank, feature normalisation, deep filtering
  and overlap-add are ours. Its four standing rules:
  - **48 kHz, not 16 kHz** — the only audio route that isn't. Pass `SAMPLE_RATE` to `decodeToMono` /
    `recordMic`; a 16 kHz assumption leaking in from ASR discards the band the model repairs.
  - **Import ORT as `onnxruntime-web/webgpu`**, the same subpath Transformers.js uses, so Vite emits
    one shared WASM asset (the bare entry adds a second 26 MB build). It is in `optimizeDeps.include`
    too: discovered inside a Worker mid-session, Vite re-optimises and reloads the page mid-load.
  - **Validate `deepfilter-auxiliary.bin` on load and refuse to run on a mismatch** (`parseAux`) —
    124 KB of untyped float32 whose forward matrix is `[481,32]` and inverse `[32,481]`, *different*
    orders, where a transposed read still looks like a valid matrix.
  - **Wrong DSP fails silently**, so it is pinned against the official implementation (libDF) through a
    captured fixture rather than against our own expectations. `SPEC_SCALE = 2*hop/fft²` is
    load-bearing: the unit-norm feature divides by `sqrt(state)` and is *not* level-invariant, so
    dropping it makes the network mask clean speech away as noise.
- **Voice Activity Detection (`src/audio/vad/`) is the other bare-ONNX route, and the only recurrent
  one.** Silero v5 scores one 32 ms frame at a time and hands back a state tensor for the next call.
  Three rules:
  - **The window is 576 samples, not 512** — 64 samples of preceding context are prepended to each
    frame (upstream's `OnnxWrapper.__call__` does the same). Input dims are dynamic, so a bare 512
    runs fine and returns scores that never fire: speech reads 0.05 instead of 0.9.
  - **State and context reset per clip**, never per session, or the previous take leaks into the next.
  - **It is pinned to WASM on purpose, not as a fallback.** 0.30 ms per frame on CPU (~100x real time)
    leaves nothing for a GPU to win, and its LSTM/`If` ops aren't covered by ORT's WebGPU provider.
    The threshold→segment step (`segments.ts`) is pure and runs on the main thread, so dragging the
    threshold re-derives segments without re-running the model.
- **Unit tests mock the network and ORT, so they cannot catch a broken model.** Both DeepFilterNet
  bugs above shipped past a green suite, and a 512-sample VAD window would too. The `@slow` E2E specs
  (`e2e/specs/audio-models.spec.ts`) are the guard; `just fe-e2e-enhance` additionally measures a real
  SDR improvement for the enhancement route, and the VAD spec asserts a real speech fraction on a
  known clip.

### In-browser vision (`src/vision/`)

The second modality on the Transformers.js path, and the same shape as `src/audio/`:
one generic worker for every discriminative task (`vision.worker.ts` — the task travels in the
`load` message), a pure `engine.ts` that owes the same three behaviours, and thin task hooks over
`useVisionPipeline`. Shipped: `/image-classification`, `/depth`, `/object-detection`,
`/segmentation`, `/zero-shot-image-classification`, `/zero-shot-object-detection`,
`/image-features`, `/mask-generation`, `/image-to-text`, `/pose`, `/video-classification`,
`/background-removal`, `/super-resolution`, `/image-to-3d`.
The rest of the category stays on a server, with a reason per task — see
[`docs/roadmaps/vision.md`](docs/roadmaps/vision.md) §3.12.

- **The last three cover *part* of a slug, and each page says which part.**
  `/super-resolution` is the `image-to-image` slug's single-pass half (editing is
  diffusion); `/image-to-3d` is `image-to-3d`'s depth-to-cloud half (reconstruction is
  SD-derived). A route that quietly answers a smaller question than its name promises is
  the failure mode; the header sentence is the fix, and a test asserts it.
- **`/background-removal` added a taxonomy row the Hub does not have** (Transformers.js
  invented the `background-removal` pipeline), taking Computer Vision from 19 rows to 20.
  It is also the only route whose blocking question was a **licence**: `briaai/RMBG-1.4`
  is Creative Commons **non-commercial** in an MIT repo, so Apache-2.0 `Xenova/modnet` is
  the default and RMBG is offered with the restriction rendered beside the choice
  (`components/vision/LicenceNote.tsx`). Read the model card before writing the catalogue
  entry — it is the one thing that can invalidate a finished route.
- **Never threshold a matte.** `vision/matte.ts` blends (`src*a + bg*(1-a)`) and the PNG
  keeps its alpha; a hard threshold turns a matting model into a segmenter with extra
  steps and cannot be undone downstream. Two traps behind it: the segmentation pipeline
  takes an **argmax** branch whenever the processor exposes a `post_process_*_segmentation`
  method (both entries publish a plain `ImageFeatureExtractor`, so they escape it —
  check a third one's `preprocessor_config.json`), and **MODNet is a *portrait* matting
  model** that returns a near-empty matte rather than an error on anything else, which is
  why `PORTRAIT_SAMPLES` exists and is listed first.
- **Tiling is `/super-resolution`'s whole correctness surface**, and it fails silently.
  `vision/tile.ts` is pure so it can be pinned: normalising by *accumulated weight* makes
  the identity round-trip exact, the `+0.5` in the feather stops a zero-weight seam
  painting a black line, and only the top-left `scale x tile` region of a patch is read
  because `Swin2SRImageProcessor` pads up to a multiple of 8 and a padded patch would drift
  every later tile. A mis-assembled upscale is perfectly sharp — `just fe-e2e-superres`
  scores it by PSNR against a ground truth the spec builds itself (crop a sample, halve it
  in the page, upload the buffer). That spec turned the WASM precision pin from a
  precaution into a **measurement**: at q8 Swin2SR scores 27.39 dB against bicubic's 27.55
  — *worse than not running it* — while fp32 scores 27.80, so `dtypes: { wasm: "fp32" }`
  stays. It also caught `MS_PER_TILE` being out by a factor of ten. **Say whether a dtype
  pin is a measurement or a precaution**; only one of them is evidence.
- **`/image-to-3d` is the one page where both runtimes appear, and they still do not mix.**
  Inference is `src/vision/`, the unprojection is pure arithmetic (`vision/pointCloud.ts`),
  the render pass is hand-written WGSL in `src/webgpu/` (`pointRenderer.ts` +
  `shaders/points.wgsl` — the repo's first *render* pipeline). They meet in the route as a
  `Float32Array`; neither module imports the other. Three things it settled: **inverse
  depth means a big value is *near***, so distance is its reciprocal and getting it
  backwards turns the scene inside out while still looking like a point cloud; **bounds
  must be read back out of the float32 buffer**, not from the doubles that wrote them;
  and WebGPU's `point-list` is always one pixel, so points are **instanced quads** with
  depth testing. The vertex buffer is written once per inference, the camera uniform once
  per frame. The focal length is an **assumption** — relative depth carries no intrinsics —
  and the page says so beside the slider.

- **Four vision tasks own an engine instead of riding the generic worker**, and the criterion is
  always the same one: it is not a plain `pipeline()` call. `vision/zeroshot/` (split towers, to
  cache label embeddings), `vision/sam/` (two graphs — encode once, decode many),
  `vision/caption/` (the `image-to-text` pipeline cannot load Florence-2 at all: its model type is
  registered for image-text-to-text, not vision2seq, and the pipeline has nowhere to put a task
  token), and `vision/pose/` (two models live at once). Everything else is a thin hook over
  `useVisionPipeline`.
- **`/pose` is the single deliberate exception to "one model live at a time."** Top-down pose is a
  detector plus a pose model and neither half is useful alone, so a catalogue entry names both and
  quotes the **combined** download. Two consequences that bit: `model/progress.ts` had to be keyed
  on **repo + file** (both checkpoints publish an `onnx/model_fp16.onnx`, and the second was
  overwriting the first's entry, so the bar hit 100% halfway through), and both models are disposed
  with `Promise.allSettled` so a detector whose teardown throws does not skip the larger one.
- **`VisionModel.backends` is now enforced, not merely declared.** `model/useBackendProbe.ts`
  answers "what would this load on" before anything downloads and `ModelPicker` disables a model the
  machine cannot run, with the reason on the row. A `null` probe gates nothing — treating the
  undecided state as WASM greys out every WebGPU model for a frame on each page load.
- **`VisionModel.graphs` names the ONNX files an entry actually downloads** (default `["model"]`).
  CLIP as a feature extractor loads `vision_model.onnx`; SAM ships two graphs; Florence-2 ships
  four. `just fe-e2e-models` checks *those* files for the dtype each backend asks for — hard-coded
  to `model.onnx` it would look at the wrong file and pass.
- **Two coordinate round-trips are the whole correctness surface of their pages, and both fail
  silently.** SAM: a click is in CSS pixels and the canvas is sized to the source, so `OverlayCanvas`'s
  `onPick` owns the conversion — a mis-mapped point still returns a plausible mask. Pose:
  `post_process_pose_estimation` scales the heatmap peak by the box's *size* and never adds its
  *origin*, so `vision/pose/pose.ts` does — otherwise the skeleton floats beside the person. Both
  `@slow` specs assert geometry (a mask **coverage band**, the nose **above** the ankles), because
  no count-based assertion can catch either.
- **`/video-classification` is a frame-level baseline and says so as a correctness requirement**,
  not as decoration: no real video transformer has an ONNX export, the page states the limitation
  next to the result, and an E2E spec asserts the copy.

- **`/zero-shot-image-classification` is the one vision route that does not use a pipeline.**
  `src/vision/zeroshot/` drives the CLIP/SigLIP text and vision towers separately so the label
  embeddings are encoded **once and reused** across frames — the pipeline re-encodes them every call,
  which on a live feed is ~40% of the work redone to produce identical numbers. That is the documented
  criterion for owning an engine (not a plain `pipeline()` call), same as `audio/enhance/` and
  `audio/vad/`. The cache holds several prompt sets (`TEXT_CACHE_LIMIT`), because the page compares two
  templates and a single entry would thrash; it is cleared on a checkpoint change.
  - **Splitting the towers means owning the model's last three steps** — normalise, scale,
    softmax/sigmoid — in `zeroshot/scoring.ts`. `scale` is `exp(logit_scale)`, `bias` is `logit_bias`,
    both **read out of the published weights** (CLIP 100.000006; SigLIP 117.330795 / -12.932437;
    SigLIP2 112.668907 / -16.771725). **A wrong scale fails silently**: scores stay in [0, 1] and the
    ranking is unchanged, so only comparing numbers against the full graph catches it —
    `just fe-e2e-zeroshot`. That spec also *recovers* the scale from the reference model's own
    probabilities (a softmax is shift-invariant, so `ln p_i − ln p_j = scale·(cos_i − cos_j)`),
    which is what pins the constant rather than merely bounding the error.
  - **The parity check must run at fp32.** `model.onnx` and `text_model.onnx`/`vision_model.onnx`
    are *separately quantized* exports, so at q8 the two paths' embeddings differ and a scale-100
    softmax magnifies that into a ~0.07 probability gap — measured. At fp32 they agree to six
    decimals. The route itself still runs q8 on WASM, so its numbers do not match a q8 pipeline
    run; both are valid quantizations and the ranking is unaffected.
  - **The pipeline applies its own `hypothesis_template` (`"This is a photo of {}"`) by default.**
    Any path that templates its own prompts must pass `hypothesis_template: "{}"`, or it silently
    compares two templates that are neither of the ones on screen. Store labels as bare nouns (`cat`,
    not `a cat`) so a template composes correctly.
- **The shared page pieces already exist — reuse them, never re-copy.** `hooks/useImagePick.ts` (file
  / drop / sample decoding, and the one-object-URL-at-a-time rule), `hooks/useCameraFrames.ts` (open
  → grab → `downscale` → exactly one frame in flight), `components/vision/ImageSourcePanel.tsx` (the
  RUN slot's input surface) and `components/vision/OverlayCanvas.tsx` (a canvas at source resolution,
  painted by a callback). `ImageSourcePanel` renders the **input** only — an overlay is a result, and
  results belong in OUTPUT, or the two slots collapse into one.
- **A threshold, an opacity slider or a class toggle re-derives; it never re-runs.** Detection asks
  the model once at a low floor and filters what came back; segmentation keeps the masks rather than
  a finished canvas. Same pure-derivation trick `/vad` uses for its threshold.
- **`percentage: false` is pinned in `useObjectDetector`**, and `scaleDetections` maps boxes from the
  downscaled inference frame back onto the source. Getting either wrong looks like a mediocre
  detector rather than a bug.

- **A `RawImage` does not survive `postMessage`.** It is a class instance, so the clone arrives with
  no methods and the pipeline rejects it. Send `toPayload(image)` (pixels + `{width,height,channels}`)
  and rebuild with `fromPayload` in the worker. `toPayload` **copies by default** — the page is
  usually still displaying what it just sent, and transferring the buffer blanks the preview. Pass
  `{ copy: false }` only for a spent webcam frame.
- **Nor does a `Tensor`, and that one throws.** Its `data`/`dims` are prototype *getters* over an
  internal ORT tensor, so structured clone refuses it: `#<_Tensor> could not be cloned`. Every result
  goes through `toCloneable` (`vision/serialize.ts`) before the engine posts it. Depth estimation is
  the task that hits it, and **only a real model load surfaces it** — the unit suite mocks the worker
  away and a mocked E2E run never loads weights.
- **Never resize or normalise for the model.** `AutoProcessor` reads the model's own
  `preprocessor_config.json`; that file *is* the input contract. `downscale()` caps the *source*
  resolution (resolution is the throttle — 1280x720 costs ~4x 640x480) and is not preprocessing.
- **A quantized export can be wrong rather than merely worse.** `onnx-community/mobilenetv4_conv_small`
  at q8 labels a tiger "sidewinder, horned rattlesnake" (44%); at fp32 it says "tiger" (62%).
  Depthwise-separable convs are the classic int8 casualty. A catalogue entry pins precision per
  backend with `dtypes` (e.g. `{ wasm: "fp32" }`) and then owes **measured** `bytes`. Only a real
  inference catches this: assert a known label on a known image in the `@slow` spec
  (`just fe-e2e-vision`), never "five rows appeared".
- **Check the files, not just the repo.** `just fe-e2e-models` verifies every catalogue id resolves
  *and* that each vision entry publishes the dtype its backend asks for — a repo with only an fp32
  `model.onnx` resolves fine on the API and 404s at load.
- **Never queue frames.** `useLiveFrames` grabs the next frame only once the previous result is back;
  a rAF loop that posts every frame drifts seconds behind. `useCamera` owns camera teardown — a
  leaked `MediaStream` leaves the webcam light on.
- **Normalise a single-channel map before painting it** (`drawHeatmap`): relative depth is on an
  arbitrary scale, and without it the canvas is uniformly black or white. Say so in the UI too — the
  values are not metres — and read the ramp's direction from the catalogue entry, because Depth
  Anything emits inverse depth (big = near) while Depth Pro emits metres (big = far).

### In-browser vision-language models (`src/multimodal/`, `/image-text-to-text`)

The third modality on the Transformers.js path, and the first **streaming** one. Same
shape as `src/audio/` and `src/vision/` — a pure `engine.ts` owing the same three
behaviours, a thin `vlm.worker.ts` around it, a `client.ts`, and `useVlm` over
`useModelWorker`. See [`docs/roadmaps/multimodal.md`](docs/roadmaps/multimodal.md).

- **There is no `image-text-to-text` pipeline in 4.2.0.** `SUPPORTED_TASKS` has 25 entries
  and that is not one, so the worker drives `AutoModelForImageTextToText` + `AutoProcessor`
  directly. This is the **third** page planned around a pipeline that could not carry it —
  MusicGen needed `MusicgenForConditionalGeneration`, Florence-2 needed
  `Florence2ForConditionalGeneration` because `image-to-text` resolves via
  `AutoModelForVision2Seq`, whose registry has no `florence2`. **Check `SUPPORTED_TASKS`
  before planning a page around a pipeline**, not after.
- **`q4f16` via `vlmLoadOpts()`, never a literal in a worker** (`asrLoadOpts` is the
  precedent). `loadOpts()`'s fp16 is 514 MB for SmolVLM-256M against 189 MB at q4f16.
  `Dtype` gained `q4f16`/`q4` and `BYTES_PER_PARAM` gained an entry for each — a widened
  `Dtype` without them renders **"NaN MB"** on a real page with nothing failing on the way
  there.
- **A q4f16 size estimate is wrong, and worse the smaller the model is.** SmolVLM-256M's
  `embed_tokens_q4f16.onnx` is 56.8 MB — *the same size as its fp16 build*, because the
  embedding table is not 4-bit quantized at all, and that is 30% of the download. Every VLM
  entry carries **measured `bytes`**, re-checked against the Hub by `just fe-e2e-models`.
  Two related traps: Qwen3-VL keeps its weights in external `.onnx_data` files (summing
  only the `.onnx` stubs measures a 1373 MB model at 1.2 **MB**), and the roadmap's
  Qwen2-VL-2B is 2668 MB at q4f16, not the ~1.1 GB it quoted.
- **`ModelResponse<TResult, TPartial = never>` gained a `partial` variant** — progress
  *inside* one run, correlated to the request id. Machine A stays `ready`, `running` stays
  an inflight count, and a partial whose request already settled is **dropped** so a late
  chunk cannot repaint a finished answer. The defaulted generic is what made it free: the
  arm is uninhabited for every non-streaming worker, so no existing engine, switch or test
  changed. It is in the shared envelope rather than a private protocol because NLP
  text-generation will need exactly the same thing.
- **`apply_chat_template` is not decoration, and getting it wrong has no error attached** —
  the output is a fluent, confident sentence that does not answer the question. Three
  silent traps: `{ type: "image" }` is a **slot** filled positionally from the image list;
  `add_generation_prompt` is what makes the model *answer* rather than continue the
  question; and `generate` returns **prompt + answer**, so the prompt's tokens are sliced
  off before decoding or the user gets their own question back.
- **512 is SmolVLM's own tile size, not a round number.** Its `preprocessor_config.json`
  sets `do_image_splitting: true` with `max_image_size.longest_edge: 512`, so a 2048px
  input is cut into up to a 4x4 grid **plus a global view** — seventeen encodes for one
  question. Downscaling the source to 512 produces one tile: ~64 image tokens instead of
  over a thousand. The DocVQA page must set its **own** number (a document needs pixels)
  rather than inherit this one — the `/link-prediction` lesson again.
- **The encode gets its own state in OUTPUT**, because the pause before the first token is
  seconds and an unlabelled pause is indistinguishable from a hang. The question is held
  INPUT: typing it, tapping a preset and picking an image all run nothing, and only
  GENERATE spends. The answer is labelled with the question it was **actually** asked,
  captured inside the run so editing the box afterwards cannot relabel a result on screen.
- **One VLM live at a time, no exception** — these are the largest downloads in the app and
  a leaked session ends the tab. Null the reference *first*, then dispose.
- **"Has a GPU" is not the gate — `shader-f16` is.** An adapter without that feature loads
  `q4f16` weights happily, reports `ready`, and then fails on the **first operator** of
  every run (`Program Gather requires f16 but the device does not support it`) — the worst
  outcome available, because the user pays for the download first. `supportsShaderF16()`
  is the real probe and `useBackendProbe({ requireShaderF16: true })` folds it in, so the
  picker disables the row *before* anything is fetched; the page then names the missing
  feature, because "resolved to wasm" is misleading on a machine that plainly has a GPU.
  Found by `just fe-e2e-vlm` on SwiftShader, which is what a runner with no `/dev/dri` gets.
- **Qwen3-VL-2B is deliberately absent, and two of this repo's own documents disagreed
  about it.** At 1373 MB it is past `adding-a-task-page.md` §0's ~1 GB ceiling and past
  `size.test.ts`'s budget, while the roadmap wanted it shipped. The ceiling won; it gets
  its own plan rather than arriving as a side effect. SmolVLM-500M (358 MB) is the second
  rung.
- **`components/Markdown.tsx` silently drops every prop but `{children, className}`**, so a
  `data-testid` handed to it never reaches the DOM — `/image-to-text` passes one that has
  never resolved. Put the testid on a wrapper.
- **`just fe-e2e-vlm` is the only test that can catch a broken chat template**, and it needs
  a real GPU (the models are WebGPU-only by catalogue declaration). It asserts a **known
  answer on a known image**; "some text appeared" would pass straight through the failure
  this page actually has.

### In-browser document QA (`src/multimodal/docvqa/`, `/document-question-answering`)

The category's second route, and the one that is **not** like the others: it rides a real
`pipeline()` call, needs no GPU, and does not stream. See
[`docs/roadmaps/multimodal.md`](docs/roadmaps/multimodal.md) §3.3.

- **Check `SUPPORTED_TASKS` *and* the registry — here both pass.** 4.2.0 carries
  `document-question-answering`, its registry maps exactly `vision-encoder-decoder →
  VisionEncoderDecoderModel` (Donut's model type), and `Xenova/donut-base-finetuned-docvqa`
  is the pipeline's own default model. This is the rare case where
  [`adding-a-model.md`](docs/guides/adding-a-model.md) §8 applies unchanged, so the route
  owns no engine beyond the generic one.
- **It is one checkpoint, and that is the pipeline's doing, not a shortage of models.**
  `DocumentQuestionAnsweringPipeline` hardcodes Donut's prompt
  (`<s_docvqa><s_question>…</s_question><s_answer>`), so another architecture would be
  prompted with tokens it has never seen — fluent and unrelated, the Florence-2 failure.
- **This route does not downscale, and it is the only one that must not.**
  `preprocessor_config.json` is `do_resize` + `do_thumbnail` + `do_pad` at a fixed
  2560x1920, and `thumbnail()` **never upscales** — it shrinks to fit, then pads. So the
  encoder always sees a 2560x1920 tensor: **inference cost is constant**, and a smaller
  source is *padded*, not enlarged. A resolution slider would therefore trade legibility
  away for no speed at all — the plan proposed one and it was wrong. `MAX_SOURCE_SIDE =
  2560` is a **memory bound at the processor's own dimension**, not preprocessing. The page
  says why it differs from every sibling route, because an unexplained inconsistency reads
  as an oversight.
- **`answer` can be `null`, and nothing errors.** The pipeline extracts with
  `decoded.match(/<s_answer>(.*?)<\/s_answer>/)` and returns `[{ answer: null }]` when that
  misses — an ordinary outcome, arriving on exactly the documents the model found hardest.
  The engine passes `null` through rather than flattening it to `""`, and OUTPUT renders it
  explicitly instead of showing a blank panel.
- **The WASM decoder must stay fp32, and the ORT bug is not ASR-specific.** A uniform q8
  cannot open a session in the browser at all (`qdq_actions.cc:137 … Missing required
  scale: …embed_tokens.weight_merged_0_scale`) — **the same bug `asrLoadOpts` was written
  for**, on a model that is not ASR. Read that note as *any encoder-decoder whose decoder is
  quantized*, on the WASM provider bundled with 4.2.0. Expressed per entry as
  `dtypes: { wasm: { encoder_model: "q8", decoder_model_merged: "fp32" } }`; if a third
  family hits it, generalise `asrLoadOpts` rather than copying it again. Cost: **596.7 MB
  on WASM instead of 218.7**, the same ~3x ASR pays, and worth it for the same reason —
  the alternative is *no* CPU path.
- **Only a real browser load catches that.** Unit tests mock the runtime, the mocked E2E run
  never loads weights, `fe-e2e-models` confirms the files exist (they do), and the identical
  call loads cleanly under `onnxruntime-node`. `just fe-e2e-docvqa` found it.
- **Both backends, ungated, deliberately.** An encoder plus a short extractive decode is a
  real CPU path, unlike the chat decoders in `src/multimodal/` — merely an expensive one.
- **WebGPU keeps fp16, unpinned.** `q4f16` would cut it to 241.0 MB, but that would be a
  **precaution, not a measurement**, and document QA is where quantization error lands
  straight on small print.
- **The repo publishes three *alternative* decoders**, so a naive sum quotes a 4 GB model.
  Sum the graphs the entry declares (`encoder_model` + `decoder_model_merged`), and carry
  measured `bytes`.
- **The page owes two sentences**: the privacy argument (the one page where a user feels
  why the architecture was chosen) and that the answer is **extracted, not reasoned** —
  Donut copies a span and cannot add up a column. Both asserted by an E2E spec, the
  `/video-classification` precedent.
- **`just fe-e2e-docvqa` needs no GPU** and pins a known answer on a known document —
  `invoice.png` is the Transformers.js docs' own DocVQA example, invoice number `us-001`.

**Two Base UI gotchas (carried over from the Radix → Base UI migration):**

1. **No `asChild` / no `<Slot>` — use `render`.** Base UI primitives compose via a `render` prop, e.g. `<Button render={<Link to="/x" />} />` (never `<Button asChild><Link/></Button>`). Likewise `FormControl` has no `Slot`: it merges ARIA/id props onto its child via Base UI's `useRender`, so it must wrap **exactly one** React element (`<FormControl><Input {...field} /></FormControl>`).
2. **No `forwardRef` — `ref` is a plain prop (React 19).** Base UI components don't use `React.forwardRef`; they accept `ref` as a normal prop. The `ui/` wrappers must spread `{...props}` straight onto the primitive and must not re-introduce `forwardRef`. This is what lets RHF's `{...field}` (which carries a `ref`) bind to `<Input>`.

---

## Permissions

`.claude/settings.json` keeps a broad `Bash(*)` **allow** so the local dev loop (`just dev`,
`just be-test`, `just fe-e2e`, `uv run …`) is friction-free, then fences the dangerous edges —
deny beats ask beats allow:

- **allow** — everything, including `git add`/`commit`/`switch`/`checkout -b`/`stash`.
- **ask** — `git push`, `git rebase`, `git merge`, `gh pr create|merge`,
  `gh issue create|edit|close`, `docker compose down`, `rm -rf`, plus the two `just` recipes that
  wrap `docker compose down -v` (`just down-v`, `just db-reset`) — a wrapper is a different
  command string, so it needs its own rule. `git merge` prompting is deliberate and is the
  point at which `develop` lands on `main`: it is a release, so it is confirmed each time.
- **deny** — `git push --force`/`-f`, `git reset --hard`, `git clean`, `git branch -D`,
  `git checkout .`.

Pattern denies stop accidents, not a determined command line (`git push origin +main` is a
force-push with no `--force` in it), so the real protection is the git setup: work on
`develop`, branch protection on `main`, and the reflog. The "Absolute Don'ts" above still
apply — the allowlist removes prompts, not judgement. Full rationale in
[`docs/guides/ai-guardrails.md`](docs/guides/ai-guardrails.md).
