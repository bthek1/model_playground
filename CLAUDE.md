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
| **Adding a task page (end-to-end procedure)** | [`docs/guides/adding-a-task-page.md`](docs/guides/adding-a-task-page.md) |
| Feature plans (phased) | **GitHub issues**, label [`plan`](https://github.com/bthek1/model_playground/issues?q=is%3Aissue+label%3Aplan) — open = active, closed = done |
| **Audio category roadmap** (complete, 6 of 6) | [`docs/roadmaps/audio.md`](docs/roadmaps/audio.md) |
| **Computer Vision roadmap** (1 of 19, plus the shared `src/vision/` module) | [`docs/roadmaps/vision.md`](docs/roadmaps/vision.md) |
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
just fe-e2e-vision  # @slow vision specs: a real MobileNetV4 load + classification (seconds)
just fe-e2e-models  # check every model id (audio + vision) resolves on the HF Hub (seconds)
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
  when the work lands** (`gh issue close <n> --comment "..."`) — the closed issue is the record.
  Reference the issue number in the commit message (`Closes #12`).
- **Never commit `.env` files.** `.env.example` is the source of truth for required vars.
- **Backend ↔ frontend communicate only via the API contract** — never mix their concerns.
- **Ask before destructive or remote actions.** Do not `git commit`, `git push`, `git reset --hard`,
  `docker compose down -v`, delete migrations, modify shared `.env` files, or create/edit/close
  GitHub issues (`gh issue …`) without explicit confirmation. See the full list in the Copilot
  instructions.

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
- **A model page survives a refresh by restoring decisions, not sessions.** A Worker cannot outlive a page load. `store/models.ts` persists the selected model and the intent to load it; `model/useModelSelection.ts` resumes on mount **only when that intent meets a cache hit** (`model/cache.ts` probes Transformers.js's `transformers-cache` bucket and answers "not cached" on any failure — the safe direction is one extra click, never bandwidth spent unasked). An uncached model still asks, and a resume is labelled "Restoring from cache…". The LOAD slot's bar reports the **aggregate** from `model/progress.ts` — monotonic percent by bytes, indeterminate until a size is known, warm-up as its own phase, no ETA — never a raw per-file `progress_callback` event, which restarts at zero for each of a model's 4–8 files.
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
- **Testing a task page** has a fixed shape (see §8 of the page-pattern standard, which lists the `data-testid` contract — `slot-1`…`slot-4`, `output-panel`, `output-empty`, `model-ready`, `error-note` — shared by Vitest and Playwright). Every task page test asserts: nothing downloads on mount (hook called with `autoLoad: false` **and** `load` not called), `load`/`retry` fire from the LOAD slot, all four slots render with `output-empty` before any run, run controls disabled until `ready`, and errors landing in the slot that produced them. A band is a labelled `region`, so `getByLabelText` can match both a band and a field inside it — query by role, or by the field's own control. **Layout is never a Vitest assertion** — jsdom has no geometry, so route tests check presence and order only, and the arrangement (side-by-side columns, output above the fold, no horizontal overflow, no transport chasm) is asserted in `e2e/specs/model-page.spec.ts`.
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
`useVisionPipeline`. Shipped: `/image-classification`. Everything else in the category is research
with a phased plan — see [`docs/roadmaps/vision.md`](docs/roadmaps/vision.md).

- **A `RawImage` does not survive `postMessage`.** It is a class instance, so the clone arrives with
  no methods and the pipeline rejects it. Send `toPayload(image)` (pixels + `{width,height,channels}`)
  and rebuild with `fromPayload` in the worker. `toPayload` **copies by default** — the page is
  usually still displaying what it just sent, and transferring the buffer blanks the preview. Pass
  `{ copy: false }` only for a spent webcam frame.
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
  arbitrary scale, and without it the canvas is uniformly black or white.

**Two Base UI gotchas (carried over from the Radix → Base UI migration):**

1. **No `asChild` / no `<Slot>` — use `render`.** Base UI primitives compose via a `render` prop, e.g. `<Button render={<Link to="/x" />} />` (never `<Button asChild><Link/></Button>`). Likewise `FormControl` has no `Slot`: it merges ARIA/id props onto its child via Base UI's `useRender`, so it must wrap **exactly one** React element (`<FormControl><Input {...field} /></FormControl>`).
2. **No `forwardRef` — `ref` is a plain prop (React 19).** Base UI components don't use `React.forwardRef`; they accept `ref` as a normal prop. The `ui/` wrappers must spread `{...props}` straight onto the primitive and must not re-introduce `forwardRef`. This is what lets RHF's `{...field}` (which carries a `ref`) bind to `<Input>`.

---

## Permissions

`.claude/settings.json` allows all `Bash(*)` commands in this project to keep the local dev loop
friction-free. The "Absolute Don'ts" above still apply — the allowlist removes prompts, not judgement.
