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
| **Audio category roadmap** (5 of 6; Text to Audio cut for size) | [`docs/roadmaps/audio.md`](docs/roadmaps/audio.md) |
| **Computer Vision roadmap** (13 of 20, plus the shared `src/vision/` module) | [`docs/roadmaps/vision.md`](docs/roadmaps/vision.md) |
| **Graph ML roadmap** (**complete, 4 of 4**; no checkpoint, pure WGSL) | [`docs/roadmaps/graph.md`](docs/roadmaps/graph.md) |
| **Multimodal roadmap** (**complete as scoped, 3 of 3**; VLMs, `q4f16`, streaming, video frames) | [`docs/roadmaps/multimodal.md`](docs/roadmaps/multimodal.md) |
| **NLP roadmap** (complete as scoped: **11 of the category's 12 taxonomy rows** shipped, across 10 of the file's 11 sections — §3.7 covers two rows with one engine — and Table QA does not port; encoders are free, decoders are a budget) | [`docs/roadmaps/nlp.md`](docs/roadmaps/nlp.md) |
| **Tabular roadmap** (**complete as scoped, 3 of 3**; the fit is the feasibility bar, not the download) | [`docs/roadmaps/tabular.md`](docs/roadmaps/tabular.md) |
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
just fe-e2e-videovlm # @slow: a real SmolVLM2-Video load — the only *multi-image* template guard (needs a GPU)
just fe-e2e-text    # @slow: text classification, pinned by a known label on a known sentence
just fe-e2e-qa      # @slow: extractive QA, pinned by a **character range** — not a string
just fe-e2e-zeroshot-text # @slow: zero-shot text — a known ranking, and a template proven to reach the model
just fe-e2e-fillmask # @slow: fill-mask — the same question through two tokenizers; the RoBERTa half is the test
just fe-e2e-embed   # @slow: embeddings — a *spread*, not a threshold (a collapse passes a threshold)
just fe-e2e-translate # @slow: translation — content words, then the **reverse pair** on the same page
just fe-e2e-summarize # @slow: summarization — two assertions and the Phase 0 latency, logged not asserted
just fe-e2e-textgen # @slow: text generation — greedy twice must be **byte-identical**
just fe-e2e-rank    # @slow: text ranking — the four stages must *disagree*, in a named direction
just fe-e2e-tabular # the model ladder fitted in a real browser, pinned above the majority baseline
just fe-e2e-forecast # the backtest spread against the single-split number (no model, no worker)
just fe-e2e-models  # check every model id (audio + vision + multimodal + text) resolves on the HF Hub (seconds)
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
  as soon as its work is complete** — every phase done, tests green, the commits on `develop`
  (`gh issue close <n> --comment "..."`). That runs **unattended**: closing a finished issue is
  bookkeeping, it is reversible from the GitHub UI, and leaving completed work open is the
  failure mode this rule exists to prevent. Closing no longer waits for the `main` merge, so
  **"closed" means done and integrated on `develop`, not shipped** — `main` is what has shipped,
  and the merge is a separate release step that still prompts. The closed issue is the record.
  Reference the issue number in the commit message (`Closes #12`).
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
- **Merging into `main` is a release, not the thing that finishes an issue.** An issue is
  finished — and closed — when its work is complete on `develop`. The merge is what makes that
  work *shipped*: `git switch main`, merge `develop` (`git merge --no-ff develop`, so the
  issue's commits stay legible as one landing), push. Merge only whole issues — `develop`
  holding half an issue is why the merge waits, not a reason to cherry-pick. Both the merge and
  the push prompt, because a release is outward-facing. Never commit directly on `main`.
- **Commits are cheap; pushes are not.** Committing and switching run unattended — they are
  reversible, and `git reflog` recovers almost anything local. **Ask before anything
  outward-facing or unrecoverable:** `git push`, `git rebase`/`git merge` (the merge into `main`
  included), `gh pr create`/`gh pr merge`, `gh issue create`/`gh issue edit`,
  `docker compose down -v`, deleting migrations, or modifying shared `.env` files.
  **`gh issue close` is the exception and runs unattended** — it publishes no content, it is
  undone with one click, and an issue whose work has landed should not stay open waiting for a
  prompt. **Never**
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
- **A dynamic `import()` of a module something else imports statically does nothing.** Rolldown says so — `[INEFFECTIVE_DYNAMIC_IMPORT] … is dynamically imported by X but also statically imported by Y, dynamic import will not move module into another chunk` — and it is a **build warning, not an error**, so it survives a green `npm run build` indefinitely. `useRanking` awaited `import("@/model/similarity")` while four other modules imported it statically: the module stayed in the entry chunk and the await bought nothing but a microtask on the dense-search path. Code-splitting works only if **every** importer is lazy, which is why `EChart.tsx` is a wrapper rather than a convention — one static import of `echarts` anywhere undoes it, and `npm run check:bundle` is what fails then. Grep for the module's other importers before reaching for `await import`.
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
- **Testing a *task hook* is a different job from testing its route, and one assertion carries most of it: which model id reached the pipeline.** A task hook is a thin wrapper, so the bugs available to it are all of the form "the control changed a label and not the checkpoint" — `useTranslate` is the pure case, since a Marian pair *is* a direction, so a hook that accepted `{ from, to }` and dropped them would keep translating correctly in the direction it was built for and every output assertion would still pass. Assert `useTextPipeline` was called with `(meta.task, meta.id, autoLoad, meta.dtypes)`, across **every** catalogue entry rather than one. Two mechanics that matter: **hoist the `load` mock so it is stable across renders** — returning a fresh `vi.fn()` from the mock factory makes every re-render look like a new hook, and `setResult` re-renders, so "did it load" becomes unanswerable and a call-count assertion fails for the wrong reason; and **`expect.anything()` rejects `undefined`**, so an entry with no `dtypes` needs the real value (`meta.dtypes`) rather than a wildcard — which is worth asserting explicitly, since "no precision pin" is a catalogue answer and a pin arriving later should be a deliberate edit with a measurement behind it.
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
  drives the real-time capture loop). `tts.worker.ts` owns the whole **text→speech** modality — Kokoro
  and MMS/SpeechT5 behind one `TtsSynthesizer` interface; it carried MusicGen too until
  `/text-to-audio` was cut, and a fourth worker would still buy nothing. Each engine (`asrEngine`/`pipelineEngine`/`ttsEngine`/`enhanceEngine`/`vadEngine`) is a pure,
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
- **The roadmaps are filtered by a feasibility bar, and it has two halves.** A task becomes a
  page only if it **runs client-side** *and* its **cheapest usable checkpoint is under ~500 MB**
  (measured off the Hub, never estimated). The size test is about the **floor, not the ceiling**:
  a cheap default plus a gated heavy option is fine; a page whose every entry is heavy is the
  failure, and is what removed `/text-to-audio`, `/image-to-text` and
  `/document-question-answering`. Sweeping the NLP roadmap at that bar removed six models and
  cost no page — every section kept a 23–284 MB default.
- **"In the browser" means the user's hardware, not necessarily the GPU.** Prefer WebGPU
  (`loadOpts()` defaults to it), but a CPU-only path is the right answer when the CPU is
  faster: `/vad` is pinned to WASM because Silero's LSTM/`If` ops have no WebGPU coverage and
  it already runs ~100x real time, and the Tabular roadmap's decision trees are TypeScript in a
  Worker because recursive splits have no matmul to accelerate. A task that needs a *server* is
  the thing that does not qualify.
- **Measure a download; never estimate it.** The NLP sweep found five of six quoted sizes wrong,
  one by 4x *and* recommended as its page's default (`deberta-v3-base-zeroshot-v2.0`: quoted
  "~180 MB", actually **738.6 MB** — the repo publishes one fp32 `model.onnx` and no q8 at all).
  **An official in-repo export is not automatically a quantized one**, and **never infer a
  seq2seq size from the parameter count**: sum `encoder_model` + `decoder_model_merged` only,
  never the alternative `decoder_model` / `decoder_with_past_model` the same repo publishes.
- **Heavy models are gated, not auto-loaded** — state the size and speed cost, download nothing
  until an explicit opt-in, and assert zero Hub requests before the click in an E2E spec. But
  **a gate is not a substitute for a size that fits**: `/text-to-audio` did all of that in front
  of MusicGen and was cut anyway, because a 599 MB minimum was the *whole* page. Gate a heavy
  entry that sits beside light ones; do not gate a page into existence.
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
`/image-features`, `/mask-generation`, `/pose`, `/video-classification`,
`/background-removal`, `/super-resolution`, `/image-to-3d`.
Six more stay on a server, with a reason per task ([`docs/roadmaps/vision.md`](docs/roadmaps/vision.md)
§3.12); `/image-to-text` shipped and was **cut for size** (§3.9).

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

- **Three vision tasks own an engine instead of riding the generic worker**, and the criterion is
  always the same one: it is not a plain `pipeline()` call. `vision/zeroshot/` (split towers, to
  cache label embeddings), `vision/sam/` (two graphs — encode once, decode many) and
  `vision/pose/` (two models live at once). Everything else is a thin hook over
  `useVisionPipeline`. `vision/caption/` was a fourth — the `image-to-text` pipeline cannot load
  Florence-2 at all — and went with its route.
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
  CLIP as a feature extractor loads `vision_model.onnx`; SAM ships two graphs; the VLM entries ship
  three. `just fe-e2e-models` checks *those* files for the dtype each backend asks for — hard-coded
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
  values are not metres — and read the ramp's direction from the catalogue entry rather than
  hard-coding it: Depth Anything emits inverse depth (big = near), while a metric model emits
  metres (big = far). `DepthModel.metric` is that seam, and it is currently unset by every
  entry because Depth Pro was cut at 1009 MB — the seam stays so a metric model can plug in.

### In-browser vision-language models (`src/multimodal/` — `/image-text-to-text`, `/visual-question-answering`, `/video-text-to-text`)

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
  `data-testid` handed to it never reaches the DOM. Put the testid on a wrapper, as this
  route does — the now-deleted `/image-to-text` passed one that never resolved.
- **`just fe-e2e-vlm` is the only test that can catch a broken chat template**, and it needs
  a real GPU (the models are WebGPU-only by catalogue declaration). It asserts a **known
  answer on a known image**; "some text appeared" would pass straight through the failure
  this page actually has.
- **Three routes, one engine — and that is the hazard as much as the point.**
  `/image-text-to-text` (a question about a picture), `/visual-question-answering`
  (the same, shaped for a short answer) and `/video-text-to-text` (a frame sampler
  in front of it) share one worker, one `engine.ts`, one `useVlm` and one
  `VlmRun` envelope. Separate routes because the Hub has separate tags and a user
  looking for VQA does not click "Image Text to Text"; the rule that keeps them
  from drifting into three catalogues and three hooks is that **a change one page
  needs belongs in the shared hook, and the other two get it**. A reviewer should
  be able to diff two of the route files and see only the question-shaping
  difference.
- **`/visual-question-answering` downloads nothing new, and its entire mechanism is
  `multimodal/prompt.ts`.** `composePrompt(question, { terse })` returns the exact
  string that will be sent *and* the cap with it — 128 tokens as typed, or the
  question plus `Answer in one word.` at 16. Three things it settles: the terse cap
  is **16, not 1**, because the instruction is the mechanism and the cap is only a
  backstop (truncating mid-word would make the demonstration indistinguishable from
  the scissors); a question that **already asks for brevity is not instructed
  again**, since ordering a small decoder twice makes it answer the instruction;
  and the **composed prompt is on screen before the click and labels the answer
  after it** — a page that rewrites the prompt silently is the
  `hypothesis_template` problem again. The toggle looks like a filter, so it is
  this page's §1.6 hazard: flipping it runs nothing.
- **`/video-text-to-text` is a frame sampler plus an image model, and says so beside
  the result** — a correctness requirement, the same one `/video-classification`
  carries, asserted by an E2E spec. `SmolVLM2-256M-Video-Instruct`, 189.2 MB
  measured, `model_type: smolvlm` (which 4.2.0 defines as a subclass of
  `idefics3` with the processor re-exported unchanged), in its own
  `VIDEO_VLM_MODELS` array — the two pages share the worker and the type, not the
  list.
- **N images, not one, and the widening was strictly additive.** `VlmRun.image`
  became `VlmRun.images`, an **ordered list**, and the chat template grew from one
  `{ type: "image" }` slot to N filled **positionally** — so the count is derived
  from the list's own length at the call site rather than passed beside it. A
  single-image run is a one-element list; `/image-text-to-text`'s own route tests
  are what proved the shipped page unchanged.
- **Frames multiply the tile problem; they do not add to it.** A 640x360 frame is
  five tiles, so eight of them is forty encodes for one question. `MAX_FRAME_SIDE`
  is 512 — the model's own `video_sampling.video_size.longest_edge`, quoted rather
  than inherited from `MAX_INFERENCE_SIDE`. The frame count is the cost dial
  (64 image tokens each, attended over for every generated word), capped at **8**;
  the model's config allows 64, which is a number for a server.
- **Sampling is by count, not by rate, and it is shown.** `multimodal/frames.ts`
  samples the **centre of each of N equal slices** (0 is usually a black frame,
  `duration` is past the last decodable one, and both look like the model ignoring
  the video). That is deliberately *not* `vision/video.ts`'s `frameTimes`, which
  samples at a fixed fps because `/video-classification`'s frames are independent
  passes; here they share one prompt, so eight means eight on a six-second clip and
  a six-minute one. The decode is still `sampleVideo` — it grew a `times` option
  (a **function of the duration**, which only the decoder has read by then) rather
  than a second copy. The filmstrip in OUTPUT is the frames the model was actually
  given: sampling invisibly makes every wrong answer unattributable.
- **The reverse toggle is the one control on these pages that legitimately spends.**
  Feed the frames backwards; if the answer does not change, the model is describing
  a picture rather than reading a sequence — the usual result at this size, and the
  finding rather than the failure. It cannot re-derive (the model must see the other
  order), so flipping runs nothing and the next GENERATE is a real second inference,
  which the page says before the click. The `@slow` spec asserts the **re-run**, not
  a difference in the answer — asserting a difference would pin a property the model
  does not have. Decoding is cached on **(clip, frame count)** and deliberately not
  on order: `useVideoPick` owns that, and the one-object-URL rule with it.
- **`just fe-e2e-videovlm` is the only guard on the multi-image template**, as
  `fe-e2e-vlm` is for the single-image one: N frames out of step with N slots
  produces a fluent answer about the wrong pictures, with no error anywhere. Both
  need a real GPU with `shader-f16`.

### In-browser NLP (`src/text/` — `/text-classification`, `/token-classification`, `/zero-shot-classification`, `/fill-mask`, `/question-answering`, `/text-features`, `/sentence-similarity`, `/translation`, `/summarization`, `/text-generation`, `/text-ranking`)

The fourth modality on the Transformers.js path, and **the cheapest module in the
app, for a reason worth knowing before planning a page**: there is no text
equivalent of `audio/io.ts` or `vision/image.ts`. The input is already a string,
so there is no decode step, no preprocessing and no transport problem — a
`RawImage` does not survive `postMessage` and a `Tensor` throws outright, while a
string crosses as itself. That absence is why `/text-classification` is the
category's first page rather than a more impressive one. Same shape as the other
three: one generic worker per modality (`pipeline.worker.ts`, task in the `load`
message), a pure `engine.ts` owing the same three behaviours, a `client.ts`, and
thin task hooks over `useTextPipeline`. See [`docs/roadmaps/nlp.md`](docs/roadmaps/nlp.md).

- **`/question-answering` is the category's first route that does not ride the
  generic worker, and the reason is a finding worth carrying forward.** The plan
  was written around `question-answering` being in `SUPPORTED_TASKS` and
  returning `{ answer, score, start, end }` with **character** offsets. It is in
  `SUPPORTED_TASKS` — and it returns `{ answer, score }`. `start` and `end` are
  declared *optional* in its types and **never populated**, past a literal
  `// TODO add start and end?` in the pipeline's own source;
  `token-classification` has the same unwritten TODO in the same place, and
  Transformers.js 4.2.0 has no `return_offsets_mapping` anywhere. So a caller
  reading `result.start` type-checks cleanly and gets `undefined` at runtime.
  **Check the fields a pipeline actually populates, not only that the task
  exists** — an optional field in a `.d.ts` is a claim about the type, not about
  the runtime. This is the fourth page planned around a pipeline that could not
  carry it, after MusicGen, Florence-2 and `/image-text-to-text`.
- **Owning the span means owning the alignment** (`text/offsets.ts`), and it is
  the page's whole correctness surface. `wordPieceOffsets` walks the tokenizer's
  pieces along the passage and returns one character range each — and **returns
  `null` rather than guessing** on any mismatch (`[UNK]`, a lowercasing
  tokenizer, an accent-stripping normaliser). The honest fallback is "no
  highlight, answer quoted": a near-miss mark lands beside the word it means and
  reads as a styling bug. It is safe for this category's *cased* checkpoints
  (`do_lower_case: false`, `strip_accents: null`) — a property of the checkpoint,
  so read its `tokenizer_config.json` before shipping an entry that highlights.
- **`context.indexOf(answer)` is not a shortcut, it is wrong on the page's own
  sample.** Asked what WebGPU supports that WebGL does not, the model answers
  `general-purpose compute shaders`; the tokenizer decodes those same ids as
  `general - purpose compute shaders`, which does not occur in the passage, so a
  substring search returns **-1**. Slicing [213, 244) returns the passage's
  characters, hyphen intact. A search also takes the *first* occurrence, which on
  a passage naming someone twice highlights the wrong one. `QaAnswer` keeps
  `text` (sliced) and `decoded` (the tokenizer's) as separate fields so the
  difference is asserted rather than assumed.
- **`qa/select.ts` is a deliberate transcription of the pipeline's span choice**,
  not an improvement: same masking, same two softmaxes, same `p(start)·p(end)`
  sweep over every `i ≤ j`, **no maximum answer length** (HF's Python caps at 15
  tokens; Transformers.js does not, and a cap changes the answer on exactly the
  unsure questions this page is about), and CLS left in the softmax denominator
  before its score is zeroed. Measured against the pipeline on nine
  question/passage pairs: same answer, same score to six decimals, all nine — so
  the offsets were added without the answers moving.
- **The model cannot abstain, and the page says so as a requirement, not a
  footnote.** SQuAD 1.1 heads always answer; the squad2 checkpoints that can
  decline have **no ONNX export**, so it is unavailable rather than unshipped.
  The note lives in OUTPUT's *description* rather than beside the result, so it
  is on screen before the first answer — a caveat that arrives only once you
  already believe the answer has arrived comes too late — and it is keyed off
  `QaModel.canAbstain` so an abstaining export retires it without a rewrite. Same
  class as `/video-classification`'s frame-level disclaimer, pinned by a test for
  the same reason. **The disclaimer ships with its demonstration**: a sample asks
  "Who won the 1998 World Cup?" of the Eiffel Tower passage and the model answers
  "Gustave Eiffel" **at 0.94** — chosen over an off-topic pair scoring 0.03,
  because the lesson is that it is *confident*, so the score is not a usable "do
  I know this" signal either.
- **`just fe-e2e-qa` asserts a character range, not a string.** "A span appeared"
  passes while the alignment is off by a token, and "the span reads Gustave
  Eiffel" passes while it marks the second mention of a name.
- **Every entry carries measured `bytes` for both backends** — stricter than
  vision's "measure where an estimate would mislead", and a finding rather than a
  preference. The NLP roadmap's size tables were all `q8` figures while
  `loadOpts()` asks for **fp16 on WebGPU**, the backend any machine with an
  adapter gets: roughly **double**, all the way down the category. It moves
  several entries across `LARGE_MODEL_BYTES` and moves Summarization's floor from
  283.9 MB to 563.6 MB, over the feasibility bar. `just fe-e2e-models` re-checks
  the quoted numbers against the Hub rather than trusting them.
- **`q4` is not a lever for an encoder.** On every encoder measured,
  `model_q4.onnx` is *larger* than `model_quantized.onnx` (distilbert-sst-2:
  118.9 MiB q4 against 64.5 MiB q8) and `q4f16` usually is too. 4-bit is a decoder
  format; only `/text-generation` has anything to gain from it.
- **There is no `textLoadOpts()`, deliberately.** The `asrLoadOpts`/`vlmLoadOpts`
  precedent is for a precision decision that holds across a *family*, and the
  measurements do not support one: q8-on-WebGPU is right for a seq2seq summarizer
  and wrong by default for a 22 MB embedder. Pins go per entry in `dtypes`, each
  saying in its comment whether it is a **measurement or a precaution**.
- **A base model is not a classifier, and the failure is silent.**
  `onnx-community/ModernBERT-base-ONNX` was cut from `/text-classification`'s
  catalogue against the roadmap's own table: a base encoder has no trained head, so
  it emits `LABEL_0`/`LABEL_1` from randomly initialised weights — confident,
  fluent and meaningless, with nothing failing on the way there. Read what a
  checkpoint was fine-tuned *for* before the catalogue entry, not after.
- **`ScoreList` refuses to render a single row**, and that is its whole design: a
  classifier's argmax is the least informative thing it produces, because
  "POSITIVE" looks identical at 0.99 and at 0.51. Callers pass the full label set,
  and a near-tie is stated in words.
- **`SpanOverlay` + `highlight()` slice the original string by character offset,
  and never rebuild the text from tokens.** Concatenated subwords lose the
  whitespace between them, so the highlight lands a character or two off — a wrong
  answer that reads as a styling problem. The assertion that pins it is that the
  concatenation of every slice equals the input **exactly**. Overlapping spans are
  **reported, not interleaved** (two spans claiming the same characters cannot both
  be drawn, and picking one quietly shows a confident highlight over a range no
  model proposed), and every span carries its **type as visible text, not colour
  alone** — the four validated `--entity-*` hues sit in the colour-vision band that
  is legal only with a secondary encoding, and no 5-hue subset passes at all, which
  is why `MISC` and `DATE` share a slot (different models, never on screen
  together).
- **`aggregation_strategy: "simple"` is pinned in the engine, not passed by the
  hook.** Without it `token-classification` returns one result per *subword token*,
  so "Wellington" comes back as `Well`/`##ing`/`##ton` and the page paints three
  highlights across one word — a rendering bug rather than an error, and one
  forgetful call site away at every future caller. `pinnedArgs()` is the single
  call site.
- **Transformers.js 4.2.0 returns no character offsets, and the types say otherwise.**
  `token-classification` hands back `entity_group`/`score`/`word` and nothing else — it
  ships with the work unwritten (`// TODO add start and end?`) and declares `start`/`end`
  **optional**, so `result.start` type-checks and is `undefined` at runtime. Every span is
  then dropped as invalid and the page renders the user's text with nothing marked, which
  is indistinguishable from a model that found nothing. It passed the unit suite (the mock
  supplied offsets the real pipeline never produces) and the mocked E2E run (which loads no
  bytes); **only `just fe-e2e-text` caught it.** `locateEntities()` recovers them by walking
  the source forward — forward rather than `indexOf` from zero, or a name said twice marks
  the first occurrence twice; with whitespace made flexible on the retry, because WordPiece
  decodes `Jones-Smith` as `Jones - Smith`; and merging *contiguous* same-label spans,
  because `aggregation_strategy: "simple"` returns "Priya Raman" as `P` + `##riya Raman`.
  Spans separated by whitespace are **not** merged — that would join "Berlin Munich" into
  one LOC. An entity that cannot be placed is reported on screen, never dropped quietly.
- **`/token-classification` is where "it runs in your browser" stops being a
  performance claim**: redacting a document you may not upload is a real reason to
  want the model on this side of the wire. Redaction is a pure derivation over
  spans in hand, so toggling it — or changing which types it removes — runs
  nothing.
- **Nothing in this category is debounced.** The roadmap wants live classification
  on a 200–300 ms pause; the page-pattern rule wins and is absolute. Typing is
  INPUT, and only GENERATE spends — a debounced auto-run is the
  five-samples-five-inferences failure with a timer in front of it.
- **The head-to-head on `/text-classification` is a second LOAD, not a toggle.**
  A second model is a second download and a second model in memory, so the page
  quotes the cost before the click. Its samples are chosen so the three models
  **disagree**; a sample set every model gets right demonstrates nothing about any
  of them.
- **`just fe-e2e-text` asserts a known label on a known sentence**, never "a ranked
  list appeared" — which is exactly what a model with a broken tokenizer also
  produces. It pins the head-to-head *structurally* (SST-2 has two classes,
  FinBERT three), because asserting that the two rankings differ would pin a
  property neither model promises.
- **`/zero-shot-classification` is the first page whose *label set* is the user's,
  and its cost model has to be said out loud.** An NLI model runs **once per
  label** — a `for` loop over the hypotheses with `await this.model(inputs)`
  inside it, no batching anywhere — so ten labels is ten inferences on one press.
  The pass count sits beside GENERATE and is **derived from the label list as it
  is edited**, so editing labels still spends nothing. `text/zeroShot.ts` owns
  that derivation, plus label parsing (bare nouns, deduped case-insensitively —
  a repeated label is a second forward pass returning the same logits).
- **The hypothesis template is INPUT, so it is on screen and editable**, and a
  template **without `{}` is refused**. This is the `hypothesis_template` finding
  from `/zero-shot-image-classification` transplanted: the text pipeline applies
  `"This example is {}."` unless told otherwise, so the route sends the template
  explicitly on every run, shows the composed hypothesis for the first label, and
  carries the template into OUTPUT with the result. Without the placeholder every
  label composes to the *same* hypothesis, so every label gets the same logits and
  the ranking is arbitrary — with nothing throwing and an ordinary-looking bar
  chart on screen.
- **`multi_label` changes the arithmetic, not the model, and cannot re-derive** —
  single-label is one softmax across the labels' entailment logits, multi-label is
  entailment-against-contradiction per label. So flipping it runs nothing and the
  next GENERATE is a real second inference, exactly as `/video-text-to-text`'s
  reverse toggle. **One label is always scored independently**
  (`softmaxEach = multi_label || labels.length === 1`) whatever the toggle says,
  and the page says so rather than rendering a lone 1.00 as certainty.
- **An NLI head that does not declare `entailment` is scored on the wrong logit,
  silently.** The pipeline looks the index up by name in `config.label2id` and
  falls back to `2` with a console warning — and the right index is 1 for
  DeBERTa-xsmall, 0 for MobileBERT and DistilBERT, 2 for BART. The output of that
  mistake is a full, confident, wrongly-ordered list, so `just fe-e2e-models`
  checks the mapping on the Hub. Check it before writing a zero-shot entry.
- **`isHeavyDownload` lives in `model/size.ts`, not beside the page that needed it
  first.** BART-large-MNLI is 816 MB on WebGPU (the roadmap's 411 MB is its q8
  size), so it gets `/depth`'s second opt-in — and the gate *moved* rather than
  being copied, the same move `backend.ts` and `size.ts` made out of `audio/`. It
  is still a **size predicate, never a model id**, which is what let it survive
  Depth Pro being cut. A page whose floor is a 26 MB model may ship an 816 MB
  entry; §0's bar is about the floor.
- **`just fe-e2e-zeroshot-text` is the only guard on the template.** It asserts a
  known ranking on a known sentence, then re-runs the same premise under a bare
  `{}` and asserts the **scores move** — two templates producing identical numbers
  is precisely what a page that lets the pipeline apply its own default looks like.
  It asserts movement rather than a flipped ranking, which would pin a property the
  model does not promise.
- **`/fill-mask` is the one page where a *base* model is the qualification**, not
  the disqualification the bullet above makes it: masked language modelling is
  the objective these encoders were pretrained on, so the head is the real one.
  Four entries across **three tokenizer families**, deliberately, so the
  mask-token hazard is one click away rather than theoretical.
- **Never write `[MASK]` in code.** `text/mask.ts` takes the token as an argument
  everywhere, `FillMaskModel.maskToken` carries it as catalogue data (so the page
  can show it and insert it *before* the 219 MB download), and the engine
  reconciles whatever the page sent against the **loaded tokenizer's own**
  `mask_token` — so a catalogue entry that drifts produces a note on screen
  rather than a failed run. `just fe-e2e-models` reads each repo's
  `tokenizer_config.json` and fails on a mismatch.
- **The plan's premise about that trap was wrong, and the correction is the
  useful part.** A hard-coded `[MASK]` on RoBERTa does **not** return fluent wrong
  predictions: `FillMaskPipeline` looks `mask_token_id` up in the ids and raises
  `Mask token (<mask>) not found in text.` The bug is loud. Measure the failure
  mode before designing around it — the three defences stay because the failure
  is still one the user did nothing to cause, not because it is silent.
- **The silent failure is a *second* mask.** The pipeline `findIndex`es the ids,
  fills the first and drops the rest with no error: "The `[MASK]` of France is
  `[MASK]`." comes back as "the border of france is." — one filling, a sentence
  quietly missing a word. The route refuses anything but **exactly one**, with
  the reason on the trigger rather than a dead button.
- **A model change rewrites the mask already in the box, and says so.** `[MASK]`
  sitting in a box now pointed at RoBERTa is the page's own hazard with the user
  holding it. Rewriting beats refusing (the sentence is the part worth keeping),
  and the alternative to rewriting silently is not refusing — it is saying
  nothing. `MASK_TOKENS` is **derived** from the catalogue, so a fifth family
  arrives as an entry rather than an edit to `mask.ts`.
- **Splice the user's string; never render the pipeline's `sequence`.** That
  field is a `tokenizer.decode(…)`, so an uncased model hands back "the capital of
  france is paris." and the page would silently rewrite what was typed. Same rule
  as `highlight()`, one layer up. Byte-level BPE also leaves the word-initial
  space on (` Paris`), and trimming can collide two token ids onto one label —
  which `ScoreList` keys its rows on, so the duplicate is dropped.
- **Bias probing is framed as evidence about the corpus, not the world**, and the
  framing travels *inside* the result. Three paired prompts differing by a single
  word, one batched GENERATE, rendered side by side under the prompts that
  produced them. A pair is the mechanism: one prompt shows a plausible sentence,
  two identical prompts show what changed when one word did.
- **DistilBERT does not know the capital of France**, and that is the page's own
  lesson rather than a quantization artefact — at fp32 it is *worse* (marseille,
  nantes, toulouse; no paris in the top three) while BERT says paris at 0.33 and
  ModernBERT at 0.88. So `just fe-e2e-fillmask` asserts *paris* on BERT and
  RoBERTa and never on DistilBERT, and **the RoBERTa half is the test** — the
  same question through a different tokenizer, which a hard-coded literal cannot
  pass.
- **Embeddings are two routes over one engine** (`/text-features`,
  `/sentence-similarity`) — one catalogue, one hook, two taxonomy rows, for the
  same reason `/visual-question-answering` is not folded into
  `/image-text-to-text`. The cheapest floor in the app: 22 MiB.
- **The pooling is catalogue data, and the plan was wrong to pin it.** #43 said
  to pin `{ pooling: "mean", normalize: true }` in the engine; only the second
  half survives. A sentence embedding *is* a pooling of the token rows, and
  which one is part of how the checkpoint was trained — mean for
  all-MiniLM/all-mpnet/Nomic, **CLS** for BGE and gte-modernbert. Mean-pool a
  CLS-trained model and you get a vector of the right width that ranks
  plausibly and is wrong, with nothing failing. So `EmbedModel.pooling` carries
  it, on screen in SELECT before a byte moves, and `EmbedModel.upstream` names
  the repo whose `1_Pooling/config.json` is the authority — **the `Xenova/*`
  ONNX mirrors do not publish that file**, so it cannot be read at load time.
  `just fe-e2e-models` reads it back and fails on a mismatch. `normalize: true`
  *is* pinned in the engine: every consumer here compares by cosine.
- **The failure is a collapse, so the assertion is a spread.** Drop the pooling
  or the normalisation and every similarity lands in a narrow band near 0.9 —
  each pair looks like every other and the page appears to work. "The
  paraphrase scores above 0.5" passes comfortably on exactly those vectors, so
  `just fe-e2e-embed` asserts the **gap** between a paraphrase and an unrelated
  pair (~0.62 against ~0.02 on all-MiniLM). A missing `pooling` also fails
  loudly one layer down now: `toVector` throws on the unpooled `[1, T, D]`
  hidden state rather than taking row 0, which is a real vector that would rank.
- **Matryoshka truncation is a measurement, not a demonstration** — four of the
  five entries were never MRL-trained, so the pages report what the cut cost
  instead of implying it was free. `truncate()` returns `kept`, the fraction of
  the vector's length the prefix held, which is precisely the factor a
  *missing* renormalisation would scale every similarity by; `‖v‖ = 1.000`
  under the strip at every width is what shows the renormalisation happened.
  `nomic-embed-text-v1.5` is in the catalogue so the genuine MRL case is one
  click away, and it is the entry that needs a **task prefix** —
  `EmbedModel.prefixes` is named by *use* (`symmetric`/`query`/`document`) so a
  page cannot reach for the wrong one, with the composed string on screen
  before the click. Truncation re-derives and spends nothing.
- **Cache embeddings by the exact string, never by a hash of it.** A hash is the
  obvious reach and is strictly worse: a collision serves another sentence's
  vector with nothing failing, which on a similarity page is a confident wrong
  number — and the strings are the user's own input, in a Map, in one tab. Keyed
  on the **composed** string, because the same sentence as a query and as a
  document is two different vectors. A model change clears it, asserted
  directly.
- **`feature-extraction` is the first text task whose result is a `Tensor`**, so
  `vision/serialize.ts` and `vision/similarity.ts` **moved** to `model/` rather
  than being copied — the `backend.ts`/`size.ts` move out of `audio/` again. A
  `Tensor` is a Transformers.js fact and a unit vector has no modality. The text
  engine flattens every result through `model/serialize.ts` exactly as vision's
  does: a `Tensor` does not merely arrive stripped of its methods, it refuses to
  be cloned at all.

- **Translation is one pair at a time, and a pair is a model.** There is no language
  argument in the task: a Marian checkpoint carries its own direction and `tr(text)` takes
  nothing else. So the direction control is a *model selector* in SELECT, switching it is
  another ~209 MB download, and `useTranslate` takes no language at all — a hook that
  accepted `{ from, to }` and dropped them would look like it worked, because the model
  keeps translating the way it was built. `just fe-e2e-translate` asserts the **reverse
  pair on the same page**, which is the only thing that catches a control that changed the
  label without changing the checkpoint.
- **A quantized seq2seq decoder cannot open a WASM session, on any family.** Marian and
  BART are the third and fourth to hit `qdq_actions.cc:137 … Missing required scale` after
  Whisper and Donut, so the spec is now **`SEQ2SEQ_WASM_DTYPES`** in `model/backend.ts` —
  `asrLoadOpts` is expressed in terms of it and every seq2seq entry references it, rather
  than the literal being copied a fifth time. The encoder quantizes fine; only the decoder
  pays full precision. It costs Marian 2.7x (101 MB → 271 MB) and it costs DistilBART its
  **entire CPU path**, because there the fallback is 742.8 MB. Only a real in-browser load
  catches it: the unit suite mocks the runtime, the mocked E2E run loads no bytes,
  `fe-e2e-models` confirms the files exist, and the identical call loads under
  `onnxruntime-node`.
- **`/translation`'s `LARGE_MODEL_BYTES` inconsistency resolved itself.** The plan expected
  en↔de (199.6 MiB fp16) to slip under the warning while en→es (213.1 MiB) crossed it. With
  the WASM pin the CPU download is 271–289 MB for every pair and `sizeEstimate` keys `large`
  off the **bigger** of the two, so every pair warns consistently. The inconsistency was an
  artefact of a WASM path that does not exist — worth remembering before writing prose to
  explain a threshold.
- **NLLB is 1.76 GB, not the 894.6 MB the roadmap quotes** — that is its q8 figure, and a
  seq2seq's q8 is unreachable on WASM and not what `loadOpts()` asks for on WebGPU. So
  `/translation` states the trade-off **per pair** (one specialist ≈ an eighth of one NLLB)
  rather than summing its catalogue, which comes to 1.6 GB and reads as an argument against
  the design.
- **`/summarization` survived its Phase 0 gate by exactly one configuration**, and the pin
  is what keeps the page alive rather than a preference: `distilbart-cnn-6-6` at
  `q8`/WebGPU is 283.9 MB and opens; fp16 is 563.6 MB, q8-on-WASM does not open, and the
  fp32-decoder fallback is 742.8 MB. Un-pinning `dtypes.webgpu` takes the page over the
  size bar silently, so a test asserts it. The **latency** half of Phase 0 is still open —
  the measurement box had only SwiftShader, whose 83 s per summary says nothing about
  hardware — so `just fe-e2e-summarize` logs the figure where there is a GPU.
- **A page needs a floor, so `/summarization` opens with `Xenova/t5-small`** (154 MB
  WebGPU / 202 MB WASM, both verified), which the plan did not consider. DistilBART is
  GPU-only by declaration. T5-small is a *much* weaker summarizer and on this page that is
  a feature: the subject is the lead-3 baseline, and a model that visibly loses to three
  sentences of the article makes the lesson concrete.
- **The lead-3 baseline is the OUTPUT slot's empty state.** It needs no model, so it is on
  screen before anything downloads, already saying what the model will be measured
  against — which keeps `output-empty` present for the four-slot contract *and* gives
  Phase 1 its "rendered from idle". `text/lead3.ts` returns offsets and **slices**, never a
  rebuild: joining split pieces loses whichever whitespace the split consumed, so the
  baseline would quietly differ from the article it claims to quote. Its splitter gets
  "Dr. Smith" wrong on purpose, and both the page and a test say so. The summary and the
  baseline are captured from the **same** article inside the run, so editing the box
  afterwards moves neither.
- **The faithfulness check borrows a model rather than adding one.** Scoring a summary
  sentence against the article *is* a one-label NLI call (`softmaxEach` is true when
  `labels.length === 1`), so it reuses `/zero-shot-classification`'s cheapest entry
  (27 MB on CPU) with its own opt-in and its own LOAD. **One pass per sentence**, and per
  sentence rather than once for the whole summary because an aggregate hides the single
  fabricated clause. It must pass `hypothesis_template: "{}"` — the third page to need
  that — and it states its own caveat: MNLI premises are single sentences, so an article is
  out of distribution and truncated at 512 tokens.

- **`/text-generation` is the decoding strategies made interactive, not a chatbot** — and
  it is the payoff for #30 putting `partial` in the **shared** `ModelResponse` envelope.
  Streaming needed nothing the envelope did not have: `partial` carries the text so far
  against the request id, Machine A stays `ready`, `running` stays an inflight count, and a
  partial whose request already settled is dropped — all three already in
  `useModelWorker`, and already asserted in `useModelWorker.test.ts`, which is where the
  rule lives. `TextGenPartial` is deliberately **not** a stage union like the VLM's: a
  text-only decoder has no encode phase, so a stage would say nothing the arriving text
  does not. It gets its own worker (`textgen.worker.ts`) for the ASR reason — the task that
  owns a loop owns its worker.
- **The `qdq_actions.cc:137` bug is a property of the *export*, not of the architecture**,
  and GPT-2 is what proved it. Its q8 `decoder_model_merged` graph fails with
  `transformer.wte.weight_merged_0_scale` — the same error as Whisper, Donut, Marian and
  BART — and GPT-2 is **decoder-only**, so "any encoder-decoder whose decoder is quantized"
  was never the rule. Nor is it "tied embeddings": SmolLM2-360M has
  `tie_word_embeddings: true` and its `model_quantized.onnx` loads and generates on WASM
  fine (34 s, ~180 ms a token, measured). The older `Xenova/*` q8 builds fuse the embedding
  matmul into `MatMulNBits` with a merged scale the bundled provider cannot find; newer
  exports do not. **So: try q8, and if the session fails naming a `*_merged_0_scale`, pin
  the graph holding the embeddings to fp32 or find a newer export.** The fp32 fallback only
  exists for a *multi-graph* model — a decoder-only one has nothing to pin, which is why
  GPT-2 has no CPU path at all (its only unquantized build is 500 MB).
- **`model_file_name` does pass through `pipeline()`** — the file was found; the session is
  what failed. That half of #47's Phase 0 is settled and worth keeping: a repo whose only
  quantized build lives under a legacy `decoder_model_merged_*` name *is* reachable, via
  `MODEL_SESSION_CONFIG[DecoderOnly]`'s `{ model: options.model_file_name ?? 'model' }`.
- **SmolLM2-360M-Instruct is the default, against §3.8's own preference.** GPT-2 loops more
  readily and stays as the demonstration, but it measures 251 MB with no CPU path while
  SmolLM2 measures 273 MB genuinely quantized and runs on both. The roadmap's
  "128.3 MB at q4f16" was wrong three ways: `model_q4f16.onnx` is within 19 bytes of the
  fp16 build (`Conv1D` weights are skipped by the exporter), there is no
  `model_quantized.onnx` so a default `q8` 404s, and the 128.3 MB file is the legacy graph
  above.
- **`pickBackendForF16` closes a gap that had been luck, not design.** The picker asks
  "would this load on a GPU?" through `useBackendProbe({ requireShaderF16: true })`; the
  worker asked it with a plain `pickBackend()`. The two could not disagree while every
  `q4f16` entry declared `backends: ["webgpu"]` — the row was disabled, so the worker was
  never asked on a bad machine. The first entry with an f16 GPU path **and** a working CPU
  fallback breaks that: the row is legitimately enabled, and a plain `pickBackend()` then
  sends the user to an adapter that loads the weights, reports `ready` and fails on the
  first operator of every run. Use it wherever the resolved precision may be f16.
- **Every decoding control is INPUT and spends.** Temperature, `top_p`, `top_k`,
  greedy-vs-sampling and the repetition penalty cannot re-derive from a finished
  generation, so changing one runs nothing and the page says the next GENERATE is real.
  Under greedy the sampling knobs are **disabled and not sent at all** rather than sent and
  ignored: Transformers.js warns on a sampling parameter in a greedy run, and a warning the
  user cannot see is worse than an option that is visibly absent.
- **`just fe-e2e-textgen` asserts greedy run twice is byte-identical.** It is the only
  assertion that proves the decoding parameters reach the model — a page that dropped them
  would still generate, still read correctly, and still pass a known-continuation check —
  and it is the property the page depends on, since a repetition loop is attributable to
  greedy decoding only if greedy decoding is reproducible. The comparison test deliberately
  does **not** assert the two halves differ: low-temperature sampling can reproduce the
  greedy path, and asserting otherwise would pin a property the model does not promise.
- **`/playground` lost the Text Generation row and gained a Theory row of its own.** It is
  a WebGPU demo surface, not a task page, so it should not claim a task row — but
  re-pointing the row alone would have left it reachable only by URL, which is not what
  "stays reachable on its own terms" means. It is now **GPU Playground** under **Theory**,
  this repo's own non-Hub category, alongside the other hand-written WGSL surfaces;
  `categoryForPath("/playground")` is `"Theory"` and a test pins both halves.

- **`/text-ranking` runs all four retrieval stages client-side**, and two of them need no
  model: BM25 (`text/bm25.ts`) and reciprocal rank fusion (`text/rrf.ts`) are pure
  TypeScript, so `k1`, `b` and RRF's `k` re-score from a held index on the main thread and
  BM25 ranks before anything is downloaded. Only **embed corpus** (N passes, once) and
  **search** (one for the query plus one **per rerank candidate**) spend, and the page
  states each count before the click. A cross-encoder scores a query and a document
  *together*, so unlike an embedding it cannot be precomputed — which is the whole reason
  reranking is applied to a shortlist of 20 rather than to a corpus.
- **No new engine arm was needed for the cross-encoder.** A cross-encoder *is* a sequence
  classifier: the task is `text-classification` and the input is a `{ text, text_pair }`
  object, which `TextInput` already described. The plan budgeted a branch; there was
  nothing to add. Check the shape you already have before widening a union.
- **Two models live at once — `/pose`'s declared exception, with a new obligation.**
  `/pose` loads its pair inside one worker, so the repo-keyed progress table spans it.
  `/text-ranking` loads its pair in a **worker each**, so there are two tables and keying
  does not help: showing either bar fills to 100% and restarts, the same "100% halfway
  through" failure from the other direction. **`combineProgress`** sums the bytes against
  the catalogue entry's *measured combined size* — a constant, so the percent is monotonic
  by construction rather than by a stored clamp. Use it for any load spread over two
  workers. **The same split is why there is no `Promise.allSettled` teardown here**, and
  the plan asking for one was reading across from `/pose`: one worker owning two models
  can get a combined teardown wrong, so it needs the combinator. A worker each cannot —
  the engine rule ("one model live at a time") holds per worker unmodified, each engine
  disposes its single model with `disposeQuietly`, and `useModelWorker` terminates each
  worker independently, so a dispose that throws in one is structurally unable to reach
  the other. Read a borrowed bullet against the shape you actually built: `allSettled`
  stays right for `/pose` and is dead code here.
- **RRF does not have the property its name suggests.** "A document both lists rank second
  beats one first in one list and last in the other" is the intuitive reading and is false:
  `1/(k+r)` is convex, so by Jensen the extreme pair wins (at `k=60`, 0.032796 against
  0.032787). Tiny margin, fixed sign, and visible on a page showing four rank lists. RRF
  rewards being loved by one retriever over being liked by both.
- **At q8, a sentence's embedding depends on the batch it was embedded in** — and it
  matters enough to reorder near-ties. Measured on all-MiniLM with the same weights and the
  same text: the page's pattern (corpus as one batch, query alone) scores a decoy 0.4556
  and the right answer 0.4459; batching query and corpus together gives 0.4394 and 0.4538.
  **So choose an E2E sample against the page's own call pattern**, not an idealised offline
  one — the first version of `just fe-e2e-rank` was written the second way and failed on
  the page that shipped it. The counter-example stays in the catalogue, labelled, because a
  page that quietly dropped it would be claiming more than it measured.
- **`mixedbread-ai/mxbai-rerank-xsmall-v1` publishes no fp16 build**, so `loadOpts("webgpu")`
  asks for a file that 404s at load. Pinned to q8 on both backends — a **missing file, not
  a precision judgement**, a distinction that matters because a precision pin invites "try
  removing it".
- **The sample corpus is built so the stages disagree**, which is the only way a spec about
  four stages says anything about three of them: `just fe-e2e-rank` asserts that BM25 does
  **not** find the planted answer while the dense and reranked stages do. A corpus every
  stage ranks identically is satisfied by a page quietly rendering one list four times,
  which is this page's most plausible bug.

### In-browser tabular models (`src/tabular/` — `/tabular-classification`, `/tabular-regression`)

The fifth modality, and the one that inverts the question. Every other category asks
whether the *weights* fit in a tab; here there are no weights at all — the model is
**fitted in the tab on the user's own CSV**, so the question is whether the *training*
fits. Nothing is downloaded, nothing is uploaded, and a reload loses the file by design.
See [`docs/roadmaps/tabular.md`](docs/roadmaps/tabular.md).

- **The dataset is the memory budget, so there is no row type anywhere in the module.** One
  `Float32Array` per column, a `string[]` level table for a categorical one, and an explicit
  `Uint8Array` missing mask — never a sentinel, because `NaN` and `0` are both real values in
  real data and any sentinel silently becomes one of them on some file. The parser is ours
  rather than `papaparse`'s for exactly this reason: a general-purpose parser returns an array
  of row objects, which is the memory problem the module exists to avoid.
- **`Number("")` is `0`**, and that is the single most expensive coercion in JavaScript for a
  file parser: one empty cell in a numeric column becomes a real zero and shifts every mean,
  every split threshold and every coefficient that column touches, with nothing failing
  anywhere. A **ragged row is rejected with its line number**, never padded — a best-effort
  recovery fits a model on shifted columns, which trains happily and is wrong everywhere.
  Blank lines are skipped on a multi-column file and **kept as a missing value on a
  single-column one**, because there the two are the same characters and only the width
  decides.
- **Phase 0 was a *timing* measurement, not a size one, and it set the caps** (`limits.ts`
  carries the table). 20 columns, 2 classes, CPU-reference matmul: at 10 000 rows the whole
  ladder is ≤ 3 s; at 50 000 the worst case is 12 s; at 200 000 boosting at depth 6 is
  **51 s**. So `MAX_ROWS = 50_000`, boosting's depth caps at **6** and the forest's at 12, and
  **nothing was cut from the ladder**. Rows past the cap are **sampled evenly**, never
  truncated to a prefix: the first 50 000 rows of a file sorted by date is a different dataset,
  and saying "50 000 rows" about it would be true and misleading.
- **FIT is LOAD relabelled by the route, and the fit itself is a `run`.** Machine A is
  untouched and no page invents a `fitting` status. `load` hands typed arrays already in the
  tab to a worker in the same tab and spends nothing; the *fit* costs, so it is a request —
  which is what puts its metrics on `partial`, with `running` an inflight count and Machine A
  staying `ready`. The FIT button posts `load` and its `run` together and `fitEngine.ts`
  **serialises them with a chain** rather than trusting message order; `stop` jumps that queue,
  or it would be delivered after the fit it was meant to interrupt.
- **A fit's progress is determinate, and `model/progress.ts` is neither touched nor reused.**
  Its indeterminate mode is the obvious reach for a page with no bytes and is the wrong one:
  epochs, trees and rows are hyperparameters the user set a moment ago. The counter is
  route-owned and the engine **throttles it to ~20 posts a second** — `useModelWorker` sets
  React state from every `partial`, so a booster posting per tree would re-render a hundred
  times in eight seconds. (`useLinearTraining` rAF-batches for the same reason; batching one
  step earlier also skips the `postMessage`.)
- **Two buttons spend, one per band: FIT in slot 2, PREDICT in the RUN transport.** Choosing a
  sample, dropping a file, picking the target, toggling a feature and moving a hyperparameter
  are all *choices*. This is the page where §1.2 is easiest to break — "pick a target column
  and it fits" feels responsive and is the five-samples-five-inferences failure with a dropdown
  in front of it. **The threshold slider is the legitimate re-derivation**: it recomputes
  precision, recall and the whole confusion matrix from the held-out probabilities the fit
  already returned, on the main thread, with no worker message.
- **Every encoding decision is fitted on the training rows only** — the imputation mean, the
  standardiser's variance, the level tables — and `design.ts` takes the training indices
  explicitly so the rule is impossible to forget. Fitting them over the whole frame raises the
  held-out score and looks like a better page: `/link-prediction`'s leakage lesson in its
  cheapest form, and it fails *upward*.
- **Trees are TypeScript in a Worker with no GPU at all, and the page says why.** Recursive
  splitting is branch-heavy and does not vectorise; the linear and MLP rungs go through the
  same `MatmulFn` seam `webgpu/linearModel.ts` uses — which is **reused, not reimplemented**
  (softmax, cross-entropy and the SGD loop included). The engine reports the compute it
  *actually* used, so a silent CPU fallback cannot make the page's one sentence untrue.
- **The MLP ships because it loses.** Asserting "deep learning does not win on tabular data",
  or omitting the model and explaining why, proves nothing — so it is a genuine fit on the same
  split, and `just fe-e2e-tabular` asserts it comes in under the boosting model, because that
  claim is on screen. Measured on the bundled synthetic sample: baseline 0.556, logistic 0.611,
  forest **0.731**, boosting 0.700, MLP 0.631.
- **A sample every family gets right demonstrates nothing about any of them**, and is satisfied
  by a page that fits one model and draws it four times. Penguins (CC0) and wine (CC BY 4.0)
  are real and near-separable — the linear *floor* wins on both — so the third sample is
  **generated** (`scripts/make-tabular-samples.mjs`), built around an interaction no straight
  line can express, and labelled synthetic everywhere it appears. Every sample renders its
  licence.
- **Permutation importance has two silent failures.** Forgetting to **restore** a shuffled
  column poisons every column measured afterwards and still returns a complete, ordered,
  plausible ranking; and sharing one random stream across columns makes each shuffle depend on
  how many came before it, so the ranking moves when an unrelated feature is toggled off. It is
  grouped by the **source** column, not the design column — shuffling one indicator of a
  one-hot city produces an impossible row rather than a measurement.
- **Nothing is persisted and nothing is uploaded, and the claim is asserted rather than
  stated.** `lib/mnistCache.ts` and `lib/proteinsCache.ts` cache their dataset to IndexedDB and
  are right to — they cache a public benchmark; this would be caching someone's payroll. A
  reload loses the file, by design. **No route posts an `InferenceRun`** (`createInferenceRun`
  still has no caller anywhere in `src/`), and tests spy on `fetch`, `indexedDB.open` and
  `Storage.setItem` during a real fit.
- **The parse gets its own one-shot worker**, not a message on the fit worker, and the reason
  is the shared envelope: `ModelResponse`'s `ready` carries `{ model, backend }` and nothing
  else, so a parse riding Machine A could not hand the column list back — and the page needs
  the columns *before* it can offer a target.
- **The bundled CSVs are loaded by dynamic `import()`**, not static `?raw`. Three of them are
  ~97 KB of text no minifier can compress, and a static import puts every byte in the entry
  chunk — `npm run check:bundle` fails on exactly that. It works only because nothing imports
  them statically; one static importer anywhere and the dynamic import buys a microtask and
  nothing else.

- **`/tabular-regression` is the same worker, engine, hook and ladder — only the diagnostics
  differ**, and a reviewer should be able to diff the two route files and see exactly that. It
  is a second route rather than a branch for the `/visual-question-answering` reason: same
  engine, different question, separate Hub tags, and no shared diagnostics — residuals replace
  the confusion matrix and an interval replaces a threshold.
- **Ridge is a Cholesky *solve*, not an inverse, and that is the page's teaching point.**
  `XᵀX` and `Xᵀy` are `O(n·d²)` and `O(n·d)` in the row count and go to the GPU; the `d×d`
  factorisation is microseconds and stays on the CPU — dispatching a kernel to invert a matrix
  smaller than one workgroup is the trees lesson pointing the other way. And `λ > 0` is exactly
  what makes `XᵀX + λI` positive definite, which is what makes the factorisation legitimate:
  that is what the penalty buys, and it is why this is ridge rather than plain least squares.
  At `λ = 0` on a design with two identical columns the factorisation **cannot complete**, and
  the page says so — an inverse would have returned one of infinitely many coefficient vectors
  and looked entirely fine.
- **`transform.ts` owns the units so the route cannot mislabel them.** A fit on `log1p(y)`
  produces a smaller RMSE for the same reason a logarithm is smaller; reading it beside a raw
  fit's RMSE and concluding the transform helped is the mistake the toggle exists to
  demonstrate. So the comparable numbers are **always** the back-transformed ones in the
  target's own units, the log-space ones are a *separate* labelled field, and the page renders
  what it is handed. `forwardTarget` also refuses a target at or below −1 rather than emitting
  `NaN`s a gradient loop then propagates through every weight while the fit still "completes".
- **The log toggle spends, and the page says so before the click.** A fit on `log1p(y)` cannot
  be re-derived from a fit on `y`, so flipping it runs nothing and the next FIT is a real second
  fit — same class as `/video-text-to-text`'s reverse toggle and `/zero-shot-classification`'s
  `multi_label`.
- **A band of the wrong width looks entirely correct**, so the quantile route reports its
  **measured coverage** of the held-out rows and the E2E spec pins that to a range rather than
  asserting a band was drawn. The quantile lines are fitted independently and can cross; they
  are sorted per row, which changes no line's level and only relabels which is which where they
  had already crossed.
- **The residual plot is a main panel because the aggregate number hides the failure.** A model
  with a respectable R² and a funnel-shaped residual plot is systematically worse at one end,
  and no scalar on the page says so.
- **`REGRESSION_FAMILIES` is a separate list, not an `objective` flag on the classification
  four.** A depth that suits a Gini split is not automatically right for variance reduction, and
  the defaults here were measured on this page's own samples — `/link-prediction` paid 0.19 of
  AUC for the lesson that reuse which looks like a decision is often an inheritance.

### In-browser forecasting (`src/forecast/` — `/time-series-forecasting`)

The one route in the app with **no model, no download and no worker**, and each of those is
a decision rather than an omission. See [`docs/roadmaps/tabular.md`](docs/roadmaps/tabular.md) §3.3.

- **Both foundation forecasters are unavailable, and it is a missing *export*.** Neither
  `google/timesfm-2.0-500m-pytorch` nor `ibm-granite/granite-timeseries-patchtst` publishes ONNX
  weights (re-checked 2026-09-25 — the check is one API call each and a plan that says "checked"
  about someone else's repo owes a date). No amount of quantization reaches that, so it is
  `adding-a-task-page.md` §0 **question 1** the row fails, not question 2. It does not stop the
  page, because the baselines *are* the page.
- **There is no worker, and the page asserts that it still has none.** Naive is `last`, seasonal
  naive is `last season`, drift is a straight line through the first and last points, and a
  rolling backtest is a loop over slices — the whole page is O(points × windows) on a few
  thousand points. Wrapping that in a worker to look consistent would add a protocol, a mock and
  an asynchronous boundary with nothing to put across it. "No worker" is invisible when a
  refactor undoes it, so the E2E spec **counts `Worker` constructions** and the unit test stubs
  the constructor.
- **Three bands, and the fourth is explained where it would have been.** FIT is absent — nothing
  to download, nothing to fit — and `DeviceStatus` would be wrong too, since the page raises no
  GPU question. An unexplained gap reads as an oversight, and here the absence is the category's
  point. `slot-4` genuinely is not rendered and a test asserts it; model-page-pattern §7 carries
  the row, because four bands is the default precisely so that three is a decision someone wrote
  down.
- **The backtest must be rolling-origin, never k-fold, and the leakage fails *upward*.** Shuffled
  cross-validation on a time series trains on the future and scores the past — the
  `/link-prediction` lesson in its oldest form — and the metric *improves*. `noLeakage()` is a
  function rather than a comment, and the most valuable test in the module calls it on every
  window of every mode.
- **One split's number is shown next to the distribution of numbers**, and both come out of **one**
  backtest call. Two calls could differ in the season, the horizon or the method, and the gap the
  page draws would be attributed to the splitting. A page that showed only the spread would be
  correct and would not make the point; one that showed only the single number would be every
  other forecasting tutorial.
- **The single split *is* one of the windows** — the last rolling origin is the notebook split —
  so "the spread straddles it" is trivially true and useless as an assertion. `just
  fe-e2e-forecast` asserts the spread is **wide** (`max > 1.5 × min`), which is exactly what a
  backtest reusing one split's numbers for every window would fail. Measured on the airline
  sample: single 47.8 against a window range of 12.6–53.1.
- **MASE is the column to read, and its denominator is where it goes wrong.** It divides by the
  **in-sample** naive error, so 1.0 means "no better than repeating the last value". Scaled by the
  test window instead, or with a season of 1 where the metric is meant to be seasonal, it is still
  a small number near 1 and nothing looks broken. MAPE returns **null**, not `Infinity` and not an
  average over the non-zero rows — that average describes a different test set from every other
  number in the object.
- **The season length is a control, never detected.** Guessing the period and being wrong produces
  a confident, plausible forecast off by a phase, with no symptom. Same reason **gaps and irregular
  spacing are reported rather than interpolated**: a filled gap gives a seasonal forecast off by a
  phase and the error is then blamed on the method.
- **Drift is the mean per-step change, `(last − first)/(n − 1)`, not the regression slope.** The two
  agree exactly on a straight line — which is the series everyone checks a drift forecast against —
  so the wrong one survives review. `regressionSlope` is exported *only* so a test can assert drift
  is not it.
- **The implied frequency is the *modal* step.** A mean is dragged by one long gap; so is a median
  on a short series, where three monthly points with one month missing give diffs
  `[1 month, 2 months]` whose upper median is the gap itself — and the page then reports the
  regular interval as the anomaly.
- **The honesty note is a correctness requirement**, the third of its kind after
  `/video-classification`'s frame-level disclaimer and `/question-answering`'s cannot-abstain note:
  the page says it has no learned model, in OUTPUT's **empty state** so it is read before a forecast
  exists, and a test pins the copy.

### Three routes were built and then cut for size — read this before adding one

`/text-to-audio` (MusicGen: **599 MB** WASM / **1127 MB** WebGPU),
`/image-to-text` (Florence-2 544 MB, vit-gpt2 482 MB) and
`/document-question-answering` (Donut: 411 MB WebGPU / 597 MB WASM) all shipped,
ran correctly, and were removed. Each failed the same test:
[`adding-a-task-page.md`](docs/guides/adding-a-task-page.md) §0 question 2 —
**every checkpoint the task has is a several-hundred-megabyte download, with no
lighter entry to fall back on.** Two heavy *entries* went with them: Depth Pro
(1009 MB) off `/depth` and SigLIP 2 (751 MB) off `/zero-shot-image-classification`.

- **A gate is not a substitute for a size that fits.** All three stated the cost,
  downloaded nothing until an explicit click, and had an E2E spec asserting zero
  Hub requests before it. They were cut anyway. Gating makes a heavy entry honest
  *beside light ones*; it cannot rescue a page that is nothing but a heavy entry.
- **§0 is cheap and skipping it is not.** The measurement that condemned each page
  was available before a line was written. Ask the three questions first.
- **The taxonomy row stays; the route goes.** All three fall through to
  `/tasks/$slug`, asserted by a test in `taskTaxonomy.test.ts` so none can be
  re-mapped without a smaller model to point at. The sidebar mirrors the Hub, not
  our build state, and a documented "too heavy, here are the numbers" is a
  finished piece of work.
- **Rewrite a mechanism rather than deleting it with its subject.** `isHeavy` was
  a Depth Pro id check and is now a size threshold (`HEAVY_MODEL_BYTES`), so the
  `/depth` gate outlived the entry it was written for; `DepthModel.metric` stays
  unset for the same reason. An id check dies with its model and the next heavy
  entry arrives ungated.

**Findings these routes paid for, kept because they outlive them:**

- **Any encoder-decoder whose decoder is quantized cannot open a WASM session** on
  the ORT bundled with 4.2.0 (`qdq_actions.cc:137 … Missing required scale`). ASR
  hit it first; **Donut proved it is not ASR-specific**. Expressed per entry as
  `dtypes: { wasm: { encoder_model: "q8", decoder_model_merged: "fp32" } }`, at
  ~3x the download — the alternative is *no* CPU path. The note lives in
  `model/backend.ts` beside `asrLoadOpts`; generalise that function if a third
  family hits it. **Only a real browser load catches it**: unit tests mock the
  runtime, the mocked E2E run never loads weights, `fe-e2e-models` confirms the
  files exist, and the identical call loads cleanly under `onnxruntime-node`.
- **Check `SUPPORTED_TASKS` before planning a page around a pipeline**, not after.
  Three pages needed a model class instead: MusicGen
  (`MusicgenForConditionalGeneration` — the `text-to-audio` pipeline throws
  "Missing the following inputs: input_ids"), Florence-2
  (`Florence2ForConditionalGeneration` — `image-to-text` resolves via
  `AutoModelForVision2Seq`, whose registry has no `florence2`) and
  `/image-text-to-text` (no such task). Donut was the one that passed the check.
- **A pipeline can hardcode one model's prompt.** `DocumentQuestionAnsweringPipeline`
  bakes in `<s_docvqa><s_question>…</s_question><s_answer>`, so a second
  architecture would be prompted with tokens it has never seen — fluent and
  unrelated, the Florence-2 failure. That is why DocVQA could never have had a
  lighter second entry.
- **Never resize for the model, in both directions.** Donut's processor is
  `do_resize` + `do_thumbnail` + `do_pad` at a fixed 2560x1920 and `thumbnail()`
  **never upscales** — so inference cost was constant and a smaller source was
  *padded*, not enlarged. The plan's resolution slider would have traded
  legibility for no speed at all. `AutoProcessor` reads the model's own config
  and that file *is* the input contract.
- **`components/Markdown.tsx` silently drops every prop but `{children, className}`**,
  so a `data-testid` handed to it never reaches the DOM. Put the testid on a
  wrapper — `/image-text-to-text` does.
- **With both OCR-capable routes gone, OCR has no page.** `/image-text-to-text` is
  the nearest thing, and it is a VLM answering a question rather than a
  transcription. Likewise **there is no metric-depth path** now Depth Pro is cut:
  ZoeDepth has no export.

**Two Base UI gotchas (carried over from the Radix → Base UI migration):**

1. **No `asChild` / no `<Slot>` — use `render`.** Base UI primitives compose via a `render` prop, e.g. `<Button render={<Link to="/x" />} />` (never `<Button asChild><Link/></Button>`). Likewise `FormControl` has no `Slot`: it merges ARIA/id props onto its child via Base UI's `useRender`, so it must wrap **exactly one** React element (`<FormControl><Input {...field} /></FormControl>`).
2. **No `forwardRef` — `ref` is a plain prop (React 19).** Base UI components don't use `React.forwardRef`; they accept `ref` as a normal prop. The `ui/` wrappers must spread `{...props}` straight onto the primitive and must not re-introduce `forwardRef`. This is what lets RHF's `{...field}` (which carries a `ref`) bind to `<Input>`.

---

## Permissions

`.claude/settings.json` keeps a broad `Bash(*)` **allow** so the local dev loop (`just dev`,
`just be-test`, `just fe-e2e`, `uv run …`) is friction-free, then fences the dangerous edges —
deny beats ask beats allow:

- **allow** — everything, including `git add`/`commit`/`switch`/`checkout -b`/`stash`, and
  **`gh issue close`**: it publishes no content, one click undoes it, and a finished issue
  should not stay open waiting for a prompt.
- **ask** — `git push`, `git rebase`, `git merge`, `gh pr create|merge`,
  `gh issue create|edit`, `docker compose down`, `rm -rf`, plus the two `just` recipes that
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
