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
| **Visualizing a model & its structure (UI standard)** | [`docs/standards/model-visualization.md`](docs/standards/model-visualization.md) |
| Auth flow (JWT) | [`docs/explanations/auth-flow.md`](docs/explanations/auth-flow.md) |
| API endpoints & request/response shapes | [`docs/standards/api-contracts.md`](docs/standards/api-contracts.md) |
| Local dev setup | [`docs/guides/local-setup.md`](docs/guides/local-setup.md) |
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
just fe-test        # vitest
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

- React 19 + TypeScript ~6.0 + Vite 8 (dev server on `:5174`). Functional components only.
- All API calls go through `src/api/client.ts` (Axios + JWT with silent 401 refresh).
- Server state lives in TanStack Query; global UI flags in Zustand + Immer (`src/store/`, one file per concern) — never put server data in Zustand.
- The sidebar is **taxonomy-driven**: categories/tasks live in `components/layout/taskTaxonomy.ts` (data), rendered by `components/layout/Sidebar.tsx`. To add a task, add a data entry — map it to a real route via `REAL_ROUTES` (e.g. the Theory tools Linear Model Training → `/training` and Tensor Arithmetic → `/tensor`), else it falls through to the generic `routes/tasks.$slug.tsx` placeholder. Per-category expand state is in `store/ui.ts`.
- Forms use React Hook Form + Zod schemas (`src/schemas/`, one file per domain).
- Styling is Tailwind v4 (CSS-first, no config file) + shadcn/ui in the **`base-nova`** style, built on **`@base-ui/react`** primitives (NOT Radix). Add components with `npx shadcn@latest add <component>`.
- Charts: ECharts via the lazy `src/components/charts/EChart.tsx` wrapper, or Recharts inline. Render Markdown/LLM output with `src/components/Markdown.tsx` (`react-markdown` + `remark-gfm`).
- **Visualizing models & their structure** follows [`docs/standards/model-visualization.md`](docs/standards/model-visualization.md) — a shared grammar of stage/arrow schematics, canvas weight/activation heatmaps (diverging red=+/blue=−, alpha=magnitude), param chips, theme-token colors, and lazy charts. The primitives live in `components/viz/` (`schematic.tsx`: Stage/Arrow/ParamChip · `heatmap.tsx`: HeatmapTile/DivergingLegend); the Training route (`components/training/`) and Tensor route (`routes/tensor.tsx`) are the reference callers. Reuse those primitives; don't invent parallel ones.
- Tests: Vitest + Testing Library + MSW (`src/test/server.ts`, `handlers.ts`). `src/test/setup.ts` also polyfills `localStorage` because Node ≥25 ships a stub that shadows the DOM env's.

### WebGPU essentials (`src/webgpu/`)

- **Raw WebGPU only — no ML framework** *in `src/webgpu/`*. Custom-kernel models are WGSL compute
  shaders in `webgpu/shaders/` (imported as strings via Vite `?raw`). Types come from `@webgpu/types`
  (in `tsconfig.app.json` `types`). This rule scopes to the hand-written runtime — running *pretrained*
  models in the UI (audio ASR/TTS/classification, etc.) may use **Transformers.js / ONNX Runtime Web**,
  which run the same HF checkpoints on WebGPU or WASM. See
  [`docs/plans/in-progress/audio-models-in-browser.md`](docs/plans/in-progress/audio-models-in-browser.md).
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

**Two Base UI gotchas (carried over from the Radix → Base UI migration):**

1. **No `asChild` / no `<Slot>` — use `render`.** Base UI primitives compose via a `render` prop, e.g. `<Button render={<Link to="/x" />} />` (never `<Button asChild><Link/></Button>`). Likewise `FormControl` has no `Slot`: it merges ARIA/id props onto its child via Base UI's `useRender`, so it must wrap **exactly one** React element (`<FormControl><Input {...field} /></FormControl>`).
2. **No `forwardRef` — `ref` is a plain prop (React 19).** Base UI components don't use `React.forwardRef`; they accept `ref` as a normal prop. The `ui/` wrappers must spread `{...props}` straight onto the primitive and must not re-introduce `forwardRef`. This is what lets RHF's `{...field}` (which carries a `ref`) bind to `<Input>`.

---

## Permissions

`.claude/settings.json` allows all `Bash(*)` commands in this project to keep the local dev loop
friction-free. The "Absolute Don'ts" above still apply — the allowlist removes prompts, not judgement.
