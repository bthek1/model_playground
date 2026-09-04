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
Web** for running *pretrained* models (e.g. the audio tasks) in the UI. It is a decoupled
monorepo:

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
| **Driving a model page (SELECT→LOAD→RUN→OUTPUT)** | [`docs/standards/model-page-pattern.md`](docs/standards/model-page-pattern.md) |
| Auth flow (JWT) | [`docs/explanations/auth-flow.md`](docs/explanations/auth-flow.md) |
| API endpoints & request/response shapes | [`docs/standards/api-contracts.md`](docs/standards/api-contracts.md) |
| Local dev setup | [`docs/guides/local-setup.md`](docs/guides/local-setup.md) |
| **End-to-end tests (Playwright)** | [`docs/guides/e2e-testing.md`](docs/guides/e2e-testing.md) |
| **A model with no Transformers.js task (bare ONNX)** | [`docs/guides/adding-a-model.md`](docs/guides/adding-a-model.md) §9 |
| Celery / async tasks | [`docs/guides/celery_setup.md`](docs/guides/celery_setup.md) |
| Feature plans (phased) | [`docs/plans/in-progress/`](docs/plans/in-progress/) (active), [`docs/plans/completed/`](docs/plans/completed/) (done) |

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
just fe-e2e-models  # check every audio model id resolves on the HF Hub (seconds)
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
- **Plans are phased.** Any feature touching more than one file gets a plan first — phased, with a
  Testing section. New plans go in **`docs/plans/in-progress/`**. Update the plan's `Status`
  (`Draft → In Progress → Complete`) as work progresses; when it reaches `Complete`, **move the file
  to `docs/plans/completed/`** (`git mv`). Completed plans are kept as a record, not deleted.
- **Never commit `.env` files.** `.env.example` is the source of truth for required vars.
- **Backend ↔ frontend communicate only via the API contract** — never mix their concerns.
- **Ask before destructive or remote actions.** Do not `git commit`, `git push`, `git reset --hard`,
  `docker compose down -v`, delete migrations, or modify shared `.env` files without explicit
  confirmation. See the full list in the Copilot instructions.

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
- Charts: ECharts via the lazy `src/components/charts/EChart.tsx` wrapper, or Recharts inline. Render Markdown/LLM output with `src/components/Markdown.tsx` (`react-markdown` + `remark-gfm`).
- **Every task page is the same pipeline: Select → Load → Run → Output** — pick a model, load its weights, run it on an input, show the result. This is the standard in [`docs/standards/model-page-pattern.md`](docs/standards/model-page-pattern.md); read it before adding a task route. Two state machines, kept orthogonal: **load** (`idle → loading → ready | error`, with `progress` as a self-loop and `retry()` out of `error`) and **run** (id-correlated requests, `running` derived from an in-flight *count*, never a boolean). The plumbing lives once in `model/useModelWorker.ts`; task hooks (`useTts`, `useAsr`, `usePipeline`, `useAudioClassifier`, `useEnhance`) are thin wrappers returning the same contract — `status`/`idle`/`loading`/`ready`/`progress`/`backend`/`load`/`retry`/`run`/`running`/`result`/`error`. The shell is `components/model/` (`ModelPage` + `ModelPicker`/`ModelStatus`/`InputPanel`/`OutputPanel`, plus `DeviceStatus` for pages that probe a GPU instead of downloading weights). Never re-derive the pending map, the teardown, or a bespoke page shape. Nothing downloads until the user asks: `idle` is the default and the size estimate + large-model warning are shown first. Errors render in the slot that produced them.
- **Every task page is the same four-stage pipeline** — SELECT → LOAD → RUN → OUTPUT — specified in
  [`docs/standards/model-page-pattern.md`](docs/standards/model-page-pattern.md). The shared worker
  plumbing lives in **`src/model/useModelWorker.ts`**: worker lifecycle keyed on a `key` string, the
  id-correlated pending table, and the two state machines (A: `idle → loading → ready | error` with
  `retry()`; B: per-request, `running` is an **inflight count**, never a boolean — a boolean lies the
  moment two requests overlap). Task hooks (`useAsr`/`useTts`/`usePipeline`) are thin typed wrappers;
  don't reimplement the plumbing. Worker messages share one envelope (`ModelRequest`/`ModelResponse`
  in `src/model/types.ts`) with only the payload per task.
- **Testing a task page** has a fixed shape (see §8 of the page-pattern standard, which lists the `data-testid` contract — `slot-1`…`slot-4`, `output-panel`, `output-empty`, `model-ready`, `error-note` — shared by Vitest and Playwright). Every task page test asserts: nothing downloads on mount (hook called with `autoLoad: false` **and** `load` not called), `load`/`retry` fire from the LOAD slot, all four slots render with `output-empty` before any run, run controls disabled until `ready`, and errors landing in the slot that produced them. A band is a labelled `region`, so `getByLabelText` can match both a band and a field inside it — query by role, or by the field's own control.
- **Visualizing models & their structure** follows [`docs/standards/model-visualization.md`](docs/standards/model-visualization.md) — a shared grammar of stage/arrow schematics, canvas weight/activation heatmaps (diverging red=+/blue=−, alpha=magnitude), param chips, theme-token colors, and lazy charts. The primitives live in `components/viz/` (`schematic.tsx`: Stage/Arrow/ParamChip · `heatmap.tsx`: HeatmapTile/DivergingLegend); the Training route (`components/training/`) and Tensor route (`routes/tensor.tsx`) are the reference callers. Reuse those primitives; don't invent parallel ones.
- Tests: Vitest + Testing Library + MSW (`src/test/server.ts`, `handlers.ts`). `src/test/setup.ts` also polyfills `localStorage` because Node ≥25 ships a stub that shadows the DOM env's.
- **End-to-end tests are Playwright** (`e2e/`), covering what happy-dom can't: routing/app shell, real-browser auth, and WebGPU. Default run is fully mocked (no backend); `@backend`-tagged specs need `just be-seed-e2e`, and
  `@slow`-tagged specs (real Hugging Face downloads + real ONNX sessions) need `just fe-e2e-slow`. Import `test`/`expect` from `e2e/fixtures/base`, not `@playwright/test`. Two traps: never `page.route("**/api/**")` (it also matches `/src/api/*` module URLs and stops the app booting), and keep the `test.include`/`test.exclude` block in `vite.config.ts` pinned to `src/` or Vitest swallows the E2E specs. See [`docs/guides/e2e-testing.md`](docs/guides/e2e-testing.md).

### WebGPU essentials (`src/webgpu/`)

- **Raw WebGPU only — no ML framework** *in `src/webgpu/`*. Custom-kernel models are WGSL compute
  shaders in `webgpu/shaders/` (imported as strings via Vite `?raw`). Types come from `@webgpu/types`
  (in `tsconfig.app.json` `types`). This rule scopes to the hand-written runtime — running *pretrained*
  models in the UI (audio ASR/TTS/classification, etc.) may use **Transformers.js / ONNX Runtime Web**,
  which run the same HF checkpoints on WebGPU or WASM. See
  [`docs/plans/completed/audio-models-in-browser.md`](docs/plans/completed/audio-models-in-browser.md).
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


### In-browser pretrained models (`src/audio/`)

The carve-out from the raw-WebGPU rule: pretrained HF checkpoints (audio ASR/TTS/classification) run
through **Transformers.js** (`@huggingface/transformers`, plus `kokoro-js` for TTS), and one task —
speech enhancement — runs on **`onnxruntime-web` directly**. Keep both out of `src/webgpu/` — the
runtimes never mix. See [`docs/guides/adding-a-model.md`](docs/guides/adding-a-model.md) §8
(Transformers.js) and §9 (a bare ONNX graph) for the full recipes.

- **One worker per modality, not per task.** Discriminative tasks share the generic
  `pipeline.worker.ts` (the task string travels in the `load` message). ASR keeps its own worker (it
  drives the real-time capture loop). `tts.worker.ts` owns the whole **text→audio** modality — Kokoro,
  MMS/SpeechT5 *and* MusicGen — because they all fit one `TtsSynthesizer` interface; a fourth worker
  would have bought nothing. Each engine (`asrEngine`/`pipelineEngine`/`ttsEngine`) is a pure,
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
- **Size-before-load guardrail:** every catalogue entry carries `params` (millions); `audio/size.ts` +
  `components/audio/ModelPicker.tsx` quote the download for both backends and warn past
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
- **Unit tests mock the network and ORT, so they cannot catch a broken model.** Both bugs above
  shipped past a green suite. The `@slow` E2E specs (`e2e/specs/audio-models.spec.ts`) are the guard;
  `just fe-e2e-enhance` additionally measures a real SDR improvement for the enhancement route.

**Two Base UI gotchas (carried over from the Radix → Base UI migration):**

1. **No `asChild` / no `<Slot>` — use `render`.** Base UI primitives compose via a `render` prop, e.g. `<Button render={<Link to="/x" />} />` (never `<Button asChild><Link/></Button>`). Likewise `FormControl` has no `Slot`: it merges ARIA/id props onto its child via Base UI's `useRender`, so it must wrap **exactly one** React element (`<FormControl><Input {...field} /></FormControl>`).
2. **No `forwardRef` — `ref` is a plain prop (React 19).** Base UI components don't use `React.forwardRef`; they accept `ref` as a normal prop. The `ui/` wrappers must spread `{...props}` straight onto the primitive and must not re-introduce `forwardRef`. This is what lets RHF's `{...field}` (which carries a `ref`) bind to `<Input>`.

---

## Permissions

`.claude/settings.json` allows all `Bash(*)` commands in this project to keep the local dev loop
friction-free. The "Absolute Don'ts" above still apply — the allowlist removes prompts, not judgement.
