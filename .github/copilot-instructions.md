# Copilot Instructions

> This file describes the project conventions for GitHub Copilot. [`CLAUDE.md`](../CLAUDE.md)
> is the Claude Code counterpart and covers the same conventions — keep the two in sync.

## Project Overview
**Model Playground** — a web app for running ML models (LLMs, computer vision, custom networks)
**directly in the browser on the user's GPU/CPU**. Two client-side inference paths coexist: a
**raw-WebGPU runtime** (`src/webgpu/`, hand-written WGSL compute shaders) for custom kernels, and
**Transformers.js / ONNX Runtime Web** for running pretrained models (e.g. audio tasks) in the UI.
It is a monorepo containing a decoupled web application:
- `backend/` — Django REST Framework API (Python) with PostgreSQL and Celery. Acts as a **model
  registry** (catalog metadata + inference-run records); it does **not** run inference.
- `frontend/` — React SPA built with Vite (TypeScript) with TanStack Query + TanStack Router,
  Tailwind CSS, shadcn/ui, React Hook Form + Zod, Zustand, Vitest. Owns the **WebGPU runtime**
  in `src/webgpu/`; inference runs client-side in a Web Worker.

The backend exposes only API endpoints. The frontend consumes them via HTTP. Model weights are
fetched by the browser from a model host/CDN (`ModelCard.weights_url`), never proxied through
Django. They are developed and deployed independently.

This began as a generic Django+React template, so keep shared infrastructure generic and reusable —
prefer documented conventions over one-off solutions. WebGPU inference is the domain focus; see
`docs/explanations/webgpu-inference.md`.

---

## Backend (`backend/`)

**Stack:** Python 3.13, Django 5.1+, Django REST Framework, PostgreSQL, psycopg3, JWT auth (simplejwt), Celery + Redis (async tasks), django-environ, uv (package manager), ruff (lint/format), mypy (type checking), pytest + pytest-django

**Conventions:**
- All endpoints are prefixed with `/api/`
- Use class-based views (`APIView`, `generics.*`, or `ViewSet`) over function-based views
- Serializers live in `serializers.py`, business logic in `services.py`, not in views
- Use `get_object_or_404` and DRF's exception handling — never raw try/except for HTTP errors
- All responses use DRF's `Response` object — never `JsonResponse`
- Models use UUIDs as primary keys (`models.UUIDField(default=uuid.uuid4, editable=False)`)
- Use `select_related` / `prefetch_related` to avoid N+1 queries
- Database migrations live in `apps/<appname>/migrations/` — always run `makemigrations` after model changes
- Environment config via `django-environ` — never hardcode secrets or DB credentials
- `AUTH_USER_MODEL = "accounts.CustomUser"` — always use `get_user_model()`, never import `User` directly

**Auth model:** `CustomUser` extends `AbstractUser` with email as `USERNAME_FIELD` (no `username` field).
```python
# Correct — get the custom user model
from django.contrib.auth import get_user_model
User = get_user_model()
```

**Database:**
- PostgreSQL via `psycopg[binary]` (psycopg3)
- Connection configured entirely through `DATABASE_URL` env var
- Use `django.db.models.indexes` for frequently queried fields
- Prefer `bulk_create` / `bulk_update` for batch operations

**Auth:** JWT via `rest_framework_simplejwt`. Protected routes use `IsAuthenticated` permission class.
Token endpoints: `POST /api/token/` and `POST /api/token/refresh/`.

**Settings:** Split into `core/settings/base.py`, `dev.py`, `prod.py`, `test.py`.
```python
# base.py pattern
import environ
env = environ.Env()
environ.Env.read_env(BASE_DIR / ".env")

DATABASES = {'default': env.db('DATABASE_URL')}
AUTH_USER_MODEL = "accounts.CustomUser"
```

**Testing:**
- Run with `just be-test` or `cd backend && uv run pytest`
- Test settings: `DJANGO_SETTINGS_MODULE = "core.settings.test"` (SQLite, fast password hasher)
- Use `factory-boy` + `faker` for fixtures, `freezegun` for time mocking
- Fixtures go in `conftest.py` (app-level or root `backend/conftest.py`)
- Test markers: `slow`, `integration`, `development`
- Coverage: `just be-test-cov`

**Code quality:**
- Lint: `just be-lint` (`ruff check`)
- Format: `just be-fmt` (`ruff format`)
- Type check: `uv run mypy .`

**API docs:** `drf-spectacular` is installed. Schema at `/api/schema/`, Swagger UI at `/api/schema/swagger-ui/`.

**Async tasks — Celery:**
- Broker is Redis; results are stored in Postgres via `django-celery-results` (`CELERY_RESULT_BACKEND = "django-db"`)
- Periodic schedules are managed in Django admin via `django-celery-beat` (`DatabaseScheduler`)
- The Celery app lives in `core/celery.py`; tasks live in `apps/<appname>/tasks.py` and are auto-discovered
- Define tasks with `@shared_task` so they don't import the app instance directly
- Run locally with `just celery-up` (worker + beat via Docker) or `just celery-worker` (worker outside Docker); `just flower` starts the Flower monitoring UI on port 5555
- Full setup and the DRF dispatch/poll/revoke pattern: `docs/guides/celery_setup.md`

**Model registry (`apps/registry/`):**
- The catalog of browser-runnable models: `ModelCard` (metadata, `weights_url`, free-form `config`)
  and `InferenceRun` (client-reported run metrics). Exposed under `/api/registry/` via DRF
  `ViewSet`s + a `DefaultRouter` (`ModelCardViewSet`, `InferenceRunViewSet`).
- Permissions: public read of public models; auth-gated create/update; users see only their own runs.
- **The backend never runs inference** — it stores metadata only. `services.record_inference_run()`
  persists what the client reports. Endpoints: `docs/standards/api-contracts.md`.

---

## Frontend (`frontend/`)

**Stack:** React 19, TypeScript ~6.0, Vite 8 (dev server on `:5180`), TanStack Router, TanStack Query v5, Axios, Tailwind CSS v4, shadcn/ui (`base-nova` style on `@base-ui/react`), React Hook Form, Zod, Zustand (+ Immer), Vitest + MSW, date-fns, ECharts, react-markdown. ESLint 10.

**Conventions:**
- Functional components only — no class components
- All API calls go through `src/api/client.ts` (Axios instance with JWT interceptor). Its base URL is **empty by default** — requests hit `/api` on the page's own origin and the Vite dev server proxies them to `VITE_API_PROXY_TARGET`, so LAN/HTTPS access isn't blocked by mixed content, CORS, or Local Network Access
- **Server state** managed exclusively by TanStack Query (`useQuery`, `useMutation`, `useInfiniteQuery`)
- **Global/UI state** managed by Zustand stores in `src/store/` — never store server data in Zustand
- **Routing** managed by TanStack Router — file-based routes under `src/routes/`
- **Local component state** managed by `useState` / `useReducer`
- Co-locate component tests in the same folder as the component (`ComponentName.test.tsx`)
- No business logic in components — extract to custom hooks in `src/hooks/`
- Use TypeScript strictly — no `any`, define response types from API contracts in `src/types/`
- Query keys are defined as constants in `src/api/queryKeys.ts`
- Use `cn()` from `src/lib/utils.ts` for all conditional `className` merging (wraps `clsx` + `tailwind-merge`)
- All path imports use the `@/` alias (resolves to `src/`) — never use relative `../../` imports across feature boundaries
- shadcn/ui components live in `src/components/ui/` — copy-paste via `npx shadcn@latest add <component>`, never modify generated files directly

**Styling — Tailwind CSS + shadcn/ui:**
```ts
// src/lib/utils.ts — always use cn() for conditional classes
import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
```
- Tailwind CSS v4: CSS-first config via `@import "tailwindcss"` in `src/index.css` — no `tailwind.config.js`
- shadcn/ui uses CSS variables for theming — do not override them with arbitrary Tailwind values
- Install new shadcn/ui components with `npx shadcn@latest add <component>` (the `base-nova` style in `components.json` pulls Base UI versions)

**shadcn + Base UI (`@base-ui/react`, not Radix):**
- Components are built on Base UI primitives — there is **no Radix `<Slot>` and no `asChild`**. To render a primitive as another element, pass a `render` prop: `<Button render={<Link to="/x" />} />`, not `<Button asChild><Link/></Button>`.
- **forwardRef gotcha:** Base UI components are React 19 components that take `ref` as a normal prop — they do **not** use `React.forwardRef`. Our `ui/` wrappers must spread `{...props}` (which carries `ref`) straight onto the primitive and must not re-introduce `forwardRef`. This is what lets RHF's `{...field}` (which includes a `ref`) bind to `<Input>` correctly.
- **input/FormControl gotcha:** `FormControl` has no `Slot` to clone its child. It uses Base UI's `useRender` to merge ARIA/id props onto its child, so `FormControl` must wrap **exactly one** React element (e.g. `<FormControl><Input {...field} /></FormControl>`) — not a string, fragment, or multiple children.

**Forms — React Hook Form + Zod:**
```ts
// Define schema in src/schemas/<domain>.ts
import { z } from 'zod'
export const loginSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
})
export type LoginSchema = z.infer<typeof loginSchema>

// Use in component
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
const form = useForm<LoginSchema>({ resolver: zodResolver(loginSchema) })
```
- Zod schemas live in `src/schemas/` (one file per domain)
- Always use shadcn/ui `Form`, `FormField`, `FormItem`, `FormMessage` primitives — they wrap RHF context

**Global state — Zustand:**
```ts
// src/store/ui.ts
import { create } from 'zustand'
import { immer } from 'zustand/middleware/immer'

interface UIState {
  sidebarOpen: boolean
  setSidebarOpen: (open: boolean) => void
}

export const useUIStore = create<UIState>()(immer((set) => ({
  sidebarOpen: true,
  setSidebarOpen: (open) => set((s) => { s.sidebarOpen = open }),
})))
```
- One file per concern: `src/store/ui.ts`, `src/store/auth.ts`, etc.
- Use `immer` middleware for state mutations
- Never put server-fetched data in Zustand — that belongs in TanStack Query

**TanStack Query patterns:**
```ts
// Always define query keys centrally
export const queryKeys = {
  users: {
    all: ['users'] as const,
    detail: (id: string) => ['users', id] as const,
  },
}

// Mutations always invalidate relevant queries on success
const mutation = useMutation({
  mutationFn: createUser,
  onSuccess: () => {
    queryClient.invalidateQueries({ queryKey: queryKeys.users.all })
  },
})
```

**TanStack Router patterns:**
```ts
// Routes are type-safe — use useParams(), useSearch() from TanStack Router
// Loaders fetch data before render using the QueryClient
export const Route = createFileRoute('/users/$userId')({
  loader: ({ params }) =>
    queryClient.ensureQueryData(userDetailQuery(params.userId)),
  component: UserDetail,
})
```

**Testing — Vitest + React Testing Library + MSW:**
- Run with `just fe-test` or `cd frontend && npm test`
- Test environment: `happy-dom` (configured in `vite.config.ts`) — the only DOM environment; `jsdom` was installed alongside it for a while, unused, and has been removed
- Setup file: `src/test/setup.ts` — imports `@testing-library/jest-dom`, starts the MSW server, and installs an in-memory `localStorage`/`sessionStorage` polyfill (Node ≥25 ships a stub `localStorage` that shadows the DOM env's)
- HTTP-level mocking via MSW: default handlers in `src/test/handlers.ts`, shared server in `src/test/server.ts`. Override per-test with `server.use(...)`. Unhandled requests are bypassed, so module-level Axios mocks still work.
- Co-locate tests with the component/hook they test: `Button.test.tsx` next to `Button.tsx`
- Mock Axios at the module level, or intercept at the network level with MSW — never make real HTTP calls in tests
- Zod schemas are tested as pure unit tests (no DOM)

**End-to-end testing — Playwright:**
- Run with `just fe-e2e` (or `cd frontend && npm run test:e2e`); `just fe-e2e-install` downloads the browsers once
- Specs live in `frontend/e2e/specs/**/*.spec.ts` — E2E is `*.spec.ts` under `e2e/`, Vitest is `*.test.tsx` under `src/`
- **Write a Vitest test by default.** Use Playwright only for what happy-dom can't do: routing/app shell, real-browser auth (JWT persistence, the silent 401 refresh), and WebGPU
- Import `test`/`expect` from `e2e/fixtures/base`, never from `@playwright/test` directly
- The default run is fully mocked — no Django, no Postgres. Specs tagged `@backend` are excluded unless you run `just fe-e2e-full` (which calls `just be-seed-e2e` first)
- Shared mock payloads live in `src/test/fixtures/models.ts`, imported by both the MSW handlers and the Playwright mock so the two layers can't drift
- **Never `page.route("**/api/**")`** — that glob also matches the dev server's own module URLs (`/src/api/client.ts`), replacing the app's API client with JSON so it never boots. Match on `url.pathname.startsWith("/api/")` instead
- The dev server is HTTPS with a self-signed cert (WebGPU needs a secure context), so `ignoreHTTPSErrors: true` is set on both `use` and `webServer`
- Keep the `test.include`/`test.exclude` block in `vite.config.ts` pinned to `src/` — Vitest's default glob would otherwise swallow the E2E specs
- WebGPU specs are their own project and skip when no GPU device is available; the graceful-degradation specs run everywhere
- No `waitForTimeout`, no order dependence; a new spec must pass three consecutive runs
- Full detail: `docs/guides/e2e-testing.md`

**Utilities:**
- Date formatting: `date-fns` — always import via `src/lib/date.ts` wrappers, never call `date-fns` directly in components
- Charts: **ECharts**, always via the lazy-loaded `src/components/charts/EChart.tsx` wrapper
  (`echarts` is heavy — keep it code-split with `lazy(() => import(...))`). One charting library,
  not two: Recharts was a listed alternative that nothing ever imported, and was removed
- Markdown / LLM output: render with the `src/components/Markdown.tsx` component (`react-markdown` + `remark-gfm`)
- **Visualizing a model or its internal structure** follows the UI standard in
  `docs/standards/model-visualization.md` — a shared grammar of left-to-right stage/arrow
  schematics (`Stage`/`Arrow`/`ParamChip` in `components/viz/schematic.tsx`), canvas
  weight/activation heatmaps (diverging red=+ / blue=−, alpha=magnitude; `HeatmapTile` +
  `DivergingLegend` in `components/viz/heatmap.tsx`), theme-token (`--chart-1…5`) colors resolved
  via `getCSSVar()`, and lazy charts. The Training route (`components/training/`) and Tensor route
  (`routes/tensor.tsx`) are the reference callers. Reuse those primitives; render the real
  `ModelCard`/weight data, not stock diagrams; pass both themes and every WebGPU status.

**Model pages (`src/model/`) — one pipeline, four slots:**
- Every task page is SELECT → LOAD → RUN → OUTPUT, specified in
  `docs/standards/model-page-pattern.md`. Shared worker plumbing is **`src/model/useModelWorker.ts`**
  (creation/teardown keyed on a `key` string, id-correlated pending table, the response switch).
- Two orthogonal state machines. **A**: `idle → loading → ready | error`, with `retry(overrides?)`
  from `error`, `cancel()` from `loading` back to `idle`, and a model change returning to `idle`;
  `progress` never moves `status`. **B**: per request,
  id-correlated — `running` is an **inflight count, not a boolean** (a boolean reports idle as soon as
  the first of two overlapping requests returns). An error with an `id` is a run failure and leaves
  `status === "ready"`; an id-less error is a load failure.
- Task hooks (`useAsr`, `useTts`, `usePipeline`, `useEnhance`, `useVad`) are **thin typed wrappers** over it — don't
  reimplement worker plumbing in a new hook.
- Worker messages share one envelope: `ModelRequest<TLoad, TRun>` / `ModelResponse<TResult>` in
  `src/model/types.ts`. Only the payload is task-specific; never rename the envelope fields.

**WebGPU inference (`src/webgpu/`) — raw WebGPU, no ML framework:**
- Models are **WGSL compute shaders** in `src/webgpu/shaders/`, imported as strings via Vite's
  `?raw` suffix (`import shader from "./shaders/x.wgsl?raw"`). Do **not** add Transformers.js,
  ONNX Runtime, or WebLLM *to this runtime* — its kernels are hand-written for control and a minimal
  bundle. That constraint is scoped to `src/webgpu/`: running **pretrained** models in the UI (audio
  ASR/TTS/classification, etc.) may use Transformers.js / ONNX Runtime Web on WebGPU or WASM. See
  `docs/guides/adding-a-model.md` §8.
- GPU types come from `@webgpu/types` (registered in `tsconfig.app.json` `types`).
- Pipeline (see `runtime.ts::runMatmul` for the reference): `getGPUDevice()` (memoised, re-acquires
  after device-lost) → `createComputePipeline(device, wgsl)` → storage/uniform buffers (`buffers.ts`)
  → bind group → `dispatchWorkgroups(ceil(dim/workgroupSize))` → `readBackFloat32`.
- **Run heavy compute in the Web Worker** (`worker.ts`), driven from the main thread by
  `workerClient.ts`, which transfers input/output `ArrayBuffer`s (zero-copy) and correlates responses
  by request id. Never block the UI thread with a dispatch.
- `capabilities.detectWebGPU()` never throws — it returns one of `unsupported` / `no-adapter` /
  `no-device` / `ready`. `ready` means a `GPUDevice` was actually acquired (it calls `requestDevice()`),
  not just that an adapter was listed. UI must degrade gracefully on every non-`ready` status.
- **`unsupported` usually isn't a real capability gap.** `navigator.gpu` is only exposed in a
  **secure context** (HTTPS or `localhost`) — so the dev server runs over HTTPS (`@vitejs/plugin-basic-ssl`),
  and a plain-HTTP LAN origin hides WebGPU. **Firefox on Linux/macOS** also needs `dom.webgpu.enabled`
  in `about:config`. See `docs/explanations/webgpu-inference.md` → "Browser support & requirements".
- Cross-check every new kernel against a CPU reference during development (see `useGpuBenchmark`).
- Adding a model = write the WGSL kernel + register a `ModelCard`. See `docs/guides/adding-a-model.md`.

**Model task pages — Select → Load → Run → Output:**

Every task route is the same four-stage pipeline: *pick a model, load its weights, run it on an input,
show the output*. The modality changes; the pipeline does not. Full contract in
`docs/standards/model-page-pattern.md` — read it before adding a task route.

```
  ┌────────┐    ┌──────┐    ┌─────┐    ┌────────┐
  │ SELECT │───▶│ LOAD │───▶│ RUN │───▶│ OUTPUT │
  └────────┘    └──────┘    └─────┘    └────────┘
   ModelPicker   ModelStatus  InputPanel  OutputPanel
```

- **Two state machines, kept orthogonal — never collapse them into one enum.**
  - *Load* (`status`, one per worker): `idle → loading → ready | error`. `progress` events are a
    self-loop on `loading`. `retry(overrides?)` goes `error → loading` (the overrides are merged into
    the `load` message — that is how "Retry on CPU" pins `{ backend: "wasm" }`); `cancel()` goes
    `loading → idle` with no error shown; changing the selected model tears the worker down and
    returns to `idle`.
  - *Run* (per request): id-correlated promises in a pending map. `running` is derived from an
    **in-flight count**, not a boolean — a boolean is wrong the moment two requests overlap.
  - The `id` on an error message is the discriminator: `id != null` is a run failure (`status` stays
    `ready`); `id == null` is a load failure (`status → error`).
- **Every task hook returns the same shape:** `status` / `idle` / `loading` / `ready` / `progress` /
  `loadProgress` / `loadedInMs` / `backend` / `load` / `retry` / `cancel` / `run` / `running` /
  `result` / `error`, plus whatever is genuinely task-specific. Wrap `model/useModelWorker.ts` — do not re-derive the pending map, the teardown,
  or the response switch per task.
- **The shell is `components/model/`:** `ModelPage` (four named slots — a route cannot reorder them
  or drop OUTPUT) with `ModelPicker` · `ModelStatus` · `InputPanel` · `OutputPanel`, plus
  `DeviceStatus` for pages whose LOAD is a GPU probe rather than a download (`/tensor`). Band labels
  stay generic (Model / Load / Input / Output) — naming a band after the task duplicates the field
  label beneath it and makes the region's accessible name ambiguous.
- **The arrangement is horizontal, not a single column** (model-page-pattern.md §4/§4a): SELECT +
  LOAD collapse into a ~20rem **setup rail**, and RUN + OUTPUT sit side by side as the **workbench**,
  so the result never lands below the fold. One column below `md`, a horizontal setup strip over
  side-by-side work columns at `md…xl`, all three columns at `xl`. Placement is by CSS-grid *area*,
  so **DOM order stays 1→2→3→4 at every breakpoint** — never reorder the source to move a band.
  Both work columns need `min-h-0 min-w-0` or the inner scroll won't engage and the page goes wide.
  The download estimate is quoted once, by `ModelPicker`; `ModelStatus` does not repeat it. Layout
  itself is asserted in Playwright — jsdom has no geometry, so route tests check presence and order
  only.
- **Testing one:** the slots expose a `data-testid` contract shared by Vitest and Playwright —
  `slot-1`…`slot-4`, `model-size-note`, `model-size-warning`, `model-ready`, `load-progress`,
  `load-cancel`, `model-cached-<id>`, `model-evict`, `device-ready`, `output-panel`, `output-empty`,
  `output-running`, `error-note`. Add to that table
  (model-page-pattern.md §8), never invent an ad-hoc id. Each task page test covers: nothing
  downloads on mount, `load`/`retry` fire from the LOAD slot, four slots render with an empty
  OUTPUT, run controls gated on `ready`, and each error in its own slot. Playwright page objects
  mirror this — `ModelPageObject` is the base, `AudioPage`/`TensorPage` extend it.
- **Reference implementations:** `routes/text-to-speech.tsx` (downloads weights),
  `routes/tensor.tsx` (compile-only), `routes/tasks.$slug.tsx` (the empty case).
  `routes/training.tsx` is the one **documented exception** — a full-bleed canvas HUD that keeps its
  own layout; see model-page-pattern.md §7 before copying it.
- **`idle` is the default — nothing downloads on mount.** Weights are the user's bandwidth and the
  tab's memory. Show the size estimate and the large-model warning *first*, start the download on an
  explicit action. Compile-only tasks (WGSL pipeline compile) may `autoLoad`.
- **A refresh restores the decisions, not the session.** A Worker cannot outlive a page load, so
  `store/models.ts` persists the selected model and the intent to load it, and
  `model/useModelSelection.ts` auto-loads on mount **only when that intent meets a cache hit**
  (`model/cache.ts` probes Cache Storage). An uncached model still stays `idle` and asks; a resume is
  labelled "Restoring from cache…", never dressed up as a session that survived. Use the hook's
  `autoLoad` for the task hook, and never widen the rule — a false cache hit spends bandwidth the
  user did not agree to.
- **The progress bar reports the aggregate** (`model/progress.ts`): monotonic percent by bytes over
  all files, indeterminate until a size is known (and then `aria-valuenow` is omitted), warm-up as
  its own phase, and no ETA. Never render a raw per-file `progress_callback` payload — it restarts at
  zero for each of a model's 4–8 files.
- **Slot rules:** SELECT disabled while `loading`/`running`; LOAD is the only slot with a progress
  bar; RUN controls are `disabled={!ready || running}`; OUTPUT always renders (empty / running /
  result / error) so the page never jumps when a result lands.
- **Errors render in the slot that produced them** — load error in LOAD, decode error in RUN,
  inference error in OUTPUT. A failed inference must leave the page usable.
- Backend selection (`webgpu` → `wasm`) resolves once in the worker before `ready` and is fixed for
  that worker's life. No silent re-negotiation — it would make the timings the user reads meaningless.

**In-browser pretrained models (`src/audio/`) — Transformers.js / ONNX Runtime Web:**
- This is the carve-out from the raw-WebGPU rule above. Pretrained HF checkpoints (audio ASR / TTS /
  classification) run here via `@huggingface/transformers` (plus `kokoro-js` for TTS). Keep it out of
  `src/webgpu/` — the two runtimes never mix.
- **One worker per modality, not per task.** Discriminative tasks share the generic
  `pipeline.worker.ts` (task string travels in the `load` message). ASR keeps its own worker (it
  drives the real-time capture loop). `tts.worker.ts` owns the whole **text→audio** modality —
  Kokoro, MMS/SpeechT5 *and* MusicGen — since they all fit one `TtsSynthesizer` interface. Each engine (`asrEngine` / `pipelineEngine` / `ttsEngine`) is a pure message
  handler, unit-tested against a fake pipeline factory; the `*.worker.ts` file is a thin wrapper.
- Every engine owes three behaviours: **one model live at a time** (null the reference *first*, then
  dispose via `disposeQuietly`, so a failed teardown can't leave a stale model live); **warm-up on
  load** (one throwaway inference before posting `ready`, posting `{ status: "warmup" }`; never fail
  the load if it throws); and **never block the main thread**.
- **Precision is per backend, and ASR is a special case.** `loadOpts()` gives fp16 on WebGPU / q8 on
  WASM. ASR uses **`asrLoadOpts()`** instead, which keeps the decoder at **fp32 on WASM**: the
  quantized Whisper/Moonshine decoders cannot open a session on the ONNX Runtime bundled with
  `@huggingface/transformers` 4.2.0 (`qdq_actions.cc:137 … Missing required scale`). Do not "simplify"
  this back to a uniform q8 — it breaks the universal fallback. Re-test when ORT updates.
- **Verify a model id against the Hub before shipping it.** Two catalogue entries once pointed at
  `onnx-community/*` repos that don't exist (401 → "Unauthorized access to file"). `just fe-e2e-models`
  checks all of them in seconds.
- Every catalogue entry carries `params` (millions) driving the **size-before-load guardrail**
  (`audio/size.ts` + `components/model/ModelPicker.tsx`): the picker quotes the download for both
  backends and warns past `LARGE_MODEL_BYTES`. Add measured `bytes` when the params estimate would
  mislead (ASR's fp32 decoder makes WASM ~3x the estimate).
- **ASR timestamps are take-relative.** The live loop re-transcribes only the tail 30 s, so the
  model's timestamps restart at 0 on a longer take; `useLiveAsr`'s `shiftChunks()` offsets them by
  the window start before the route renders `m:ss`. Don't render worker `chunks` unshifted.
- **Gate anything heavy.** `/text-to-audio` (MusicGen: 571 MB q8, ~1 GB fp16) states its size and
  speed cost and downloads nothing until the user opts in; an E2E spec asserts zero Hub requests
  before the click. MusicGen also needs `MusicgenForConditionalGeneration` directly — the
  `text-to-audio` *pipeline* throws "Missing the following inputs: input_ids" on 4.2.0.
- **Audio-to-Audio (`src/audio/enhance/`) is one of two tasks with no Transformers.js path.** The
  DeepFilterNet3 export is the neural graph only (normalised ERB/spectral features in, mask + complex
  filter coefficients out), so it runs on **`onnxruntime-web` directly** and the STFT, ERB filterbank,
  feature normalisation, deep filtering and overlap-add are all ours. Four rules:
  - **48 kHz, not 16 kHz.** Native rate for DFN3, and the only audio route that isn't 16 kHz. Pass
    `SAMPLE_RATE` to `decodeToMono` / `recordMic`; a 16 kHz assumption leaking in from ASR throws
    away exactly the band the model repairs.
  - **Import ORT as `onnxruntime-web/webgpu`** — the same subpath `@huggingface/transformers` uses, so
    Vite emits one shared WASM asset. The bare entry adds a second 26 MB build. It is also listed in
    `optimizeDeps.include`, because discovering it inside a Worker mid-session forces a reload that
    resets a route mid-load.
  - **Validate `deepfilter-auxiliary.bin` on load and refuse to run on a mismatch** (`parseAux`). It is
    124 KB of untyped float32; the forward matrix is `[481,32]` and the inverse `[32,481]` — *different*
    orders, and a transposed read still looks like a valid matrix.
  - **Wrong DSP fails silently**, so it is pinned against the official implementation (libDF) via a
    captured fixture, not against our own expectations. The scaling constant in particular
    (`SPEC_SCALE = 2*hop/fft²`) is load-bearing: the unit-norm feature divides by `sqrt(state)` and is
    not level-invariant, so dropping it makes the network mask clean speech away. `just fe-e2e-enhance`
    measures a real SDR improvement end to end.
- **Voice Activity Detection (`src/audio/vad/`) is the other bare-ONNX task, and the only recurrent
  one.** Silero v5 scores one 32 ms frame at a time and returns a state tensor for the next call.
  Three rules:
  - **The window is 576 samples, not 512** — 64 samples of preceding context are prepended to each
    512-sample frame, as upstream's `OnnxWrapper.__call__` does. The graph's input dims are dynamic,
    so a bare 512 runs happily and returns scores that never cross a threshold (0.05 on speech that
    should read 0.9). This is the failure a mocked test cannot see.
  - **`state` and the context reset per clip**, not per session, or the previous take bleeds into the
    first frames of the next.
  - **Pinned to WASM deliberately.** 0.30 ms per frame on CPU is ~100x real time, a per-frame GPU
    dispatch would cost more than the work, and its LSTM/`If` ops aren't covered by ORT's WebGPU
    provider. The `@slow` spec asserts the backend so a later change can't loosen it silently.
  - Threshold → segments (`segments.ts`) is pure and runs on the main thread: dragging the threshold
    re-derives segments from the same scores, never re-running the model.
- **Unit tests mock the network and ORT, so they cannot catch a broken model.** The `@slow` E2E
  specs (`e2e/specs/audio-models.spec.ts`) are the guard.
- Adding a task: catalogue entry → worker (reuse the generic one) → hook → route → `REAL_ROUTES`.
  See `docs/guides/adding-a-model.md` §8 (Transformers.js) or §9 (a bare ONNX graph — DFN3 is the
  reference, `src/audio/vad/` the smaller one to read first).

**Env vars:** Prefix with `VITE_`. Access via `import.meta.env.VITE_*`.

**Commands:**
- Dev server: `just fe-dev`
- Build: `just fe-build`
- Lint: `just fe-lint`
- Test: `just fe-test`
- Test UI: `just fe-test-ui`
- End-to-end: `just fe-e2e` (browsers: `just fe-e2e-install`; UI: `just fe-e2e-ui`)
- Real model loads: `just fe-e2e-slow` (minutes, needs network); ids only: `just fe-e2e-models`;
  speech enhancement only: `just fe-e2e-enhance`; voice activity detection only: `just fe-e2e-vad`
- Install deps: `just fe-install`

---

## Task Runner (`justfile`)

All common tasks are defined in the root `justfile`. Use `just --list` to see all commands.

Key commands:
| Command | Description |
|---|---|
| `just up` | Start all Docker services |
| `just be-dev` | Run Django dev server locally (runs migrations first) |
| `just be-test` | Run backend test suite |
| `just be-test-cov` | Run backend tests with coverage |
| `just be-makemigrations` | Create new migrations |
| `just be-migrate` | Apply migrations |
| `just be-lint` / `just be-fmt` | Lint / format backend |
| `just fe-dev` | Run Vite dev server locally |
| `just fe-build` | Production build |
| `just fe-test` | Run frontend test suite |
| `just fe-test-ui` | Run frontend tests with Vitest UI |
| `just fe-e2e` | Run Playwright end-to-end tests (mocked API) |
| `just fe-e2e-full` | Run E2E tests against the real Django API |
| `just fe-e2e-vad` | Run the @slow voice-activity-detection specs (seconds) |
| `just fe-e2e-install` | Download the Playwright browsers (once) |
| `just fe-e2e-slow` | `@slow` specs: real model downloads + real ONNX sessions |
| `just fe-e2e-models` | Check every audio model id resolves on the HF Hub (seconds) |
| `just be-seed-e2e` | Create/reset the E2E test user (dev only) |
| `just be-startapp name` | Scaffold a new Django app |

---

## Monorepo Structure

```
/
├── backend/
│   ├── core/
│   │   ├── settings/          # base.py, dev.py, prod.py, test.py
│   │   ├── urls.py
│   │   └── wsgi.py
│   ├── apps/                  # Django apps (one per domain)
│   │   ├── accounts/          # CustomUser, JWT auth endpoints
│   │   │   ├── models.py
│   │   │   ├── serializers.py
│   │   │   ├── services.py
│   │   │   ├── views.py
│   │   │   ├── urls.py
│   │   │   └── migrations/
│   │   ├── pages/             # Health check and static page endpoints
│   │   └── registry/          # Model catalog (ModelCard) + run metadata (InferenceRun)
│   ├── conftest.py            # Root pytest fixtures
│   ├── manage.py
│   ├── pyproject.toml         # Dependencies (uv), pytest, ruff config
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── api/               # Axios client, endpoint functions, queryKeys
│   │   ├── components/
│   │   │   ├── ui/            # shadcn/ui copy-paste components
│   │   │   └── charts/        # EChart wrapper (lazy-loaded)
│   │   ├── webgpu/            # Raw-WebGPU runtime (device, buffers, pipeline, worker, shaders/)
│   │   ├── hooks/             # Custom hooks (business logic)
│   │   ├── lib/               # Shared utilities: cn(), date wrappers
│   │   ├── routes/            # TanStack Router file-based routes
│   │   ├── schemas/           # Zod validation schemas (one file per domain)
│   │   ├── store/             # Zustand stores (one file per concern)
│   │   ├── test/              # Vitest setup, MSW handlers, shared mock fixtures
│   │   ├── types/             # Shared TypeScript types from API contracts
│   │   └── main.tsx
│   ├── e2e/                   # Playwright end-to-end tests
│   │   ├── fixtures/          # base test object, mock API, WebGPU probe
│   │   ├── pages/             # page objects
│   │   └── specs/             # *.spec.ts (webgpu/ is its own project)
│   ├── vite.config.ts
│   ├── playwright.config.ts
│   ├── package.json
│   └── .env.example
├── docs/
│   ├── standards/             # Coding standards, style guides, conventions, API contracts
│   ├── guides/                # How-to guides, onboarding, local setup, deployment
│   ├── plans/                 # Feature plans, ADRs, roadmaps (phased, with testing)
│   └── explanations/          # Concept explanations, design rationale, background context
├── justfile                   # Task runner (use `just --list`)
├── docker-compose.yml
└── README.md
```

---

## Docker Compose (local dev)

```yaml
services:
  db:
    image: postgres:16
    environment:
      POSTGRES_DB: appdb
      POSTGRES_USER: appuser
      POSTGRES_PASSWORD: apppassword
    ports:
      - "5432:5432"

  backend:
    build: ./backend
    command: python manage.py runserver 0.0.0.0:8000
    volumes:
      - ./backend:/app
    ports:
      - "8000:8000"
    depends_on:
      - db
    env_file:
      - ./backend/.env

  frontend:
    build: ./frontend
    command: npm run dev
    volumes:
      - ./frontend:/app
    ports:
      - "5173:5173"
```

---

## Docs (`docs/`)

The `docs/` folder is the single source of truth for project knowledge. It is kept in sync with the codebase.

**Structure:**
- `docs/standards/` — Coding standards, style guides, naming conventions, API contracts, the
  model-page interaction standard (`model-page-pattern.md`) and the model-visualization UI
  standard (`model-visualization.md`)
- `docs/guides/` — Step-by-step how-to guides, onboarding, local setup, deployment
- `docs/explanations/` — Concept explanations, design rationale, background context

Feature plans, ADRs, roadmaps and spike notes are **not files in `docs/`** — they are GitHub
issues. See *Planning Rules* below.

**Rules:**
- When a feature, API endpoint, or architectural pattern is added or changed, update the relevant doc in `docs/` as part of the same change
- New backend apps or frontend modules should have a corresponding explanation or guide in `docs/`
- API contract changes (new endpoints, modified request/response shapes) must be reflected in `docs/standards/`
- Architecture or design decisions must be recorded as an ADR in a GitHub issue labelled `plan`
  (open while active, closed when finalised)
- Docs are written for the next developer — assume no prior context

---

## Planning Rules (GitHub issues)

Every non-trivial feature or change must have a plan **before** implementation begins, and the
plan lives in a **GitHub issue** — never in a markdown file in the repo. There is no `docs/plans/`
folder; do not recreate one.

**Lifecycle:**

| Stage | What it means | Command |
|---|---|---|
| Open, `plan` label | Draft or in progress | `gh issue create --label plan --title "Plan: <Feature>" --body-file <file>` |
| Phase checkboxes ticked | Progress is visible on the issue itself | `gh issue edit <n> --body-file <file>` |
| Closed | Complete — the closed issue is the permanent record | `gh issue close <n> --comment "<what landed>"` |

- Write the plan body to a scratch file first, then pass it with `--body-file` — it keeps long
  markdown intact. The scratch file is temporary; **never commit it**.
- Reference the issue from the work: `Closes #<n>` in the commit message or PR body.
- Roadmap/research issues (label `roadmap`) are **not plans** — they are the per-category research a
  plan gets written from, and they stay open.

**Issue title:** `Plan: <Feature Name>`

**Required plan body:**
```markdown
## Goal
One paragraph describing what this plan achieves and why.

## Background
Context and motivation. What problem does this solve?

## Phases

### Phase 1 — <Name>
- [ ] Task 1
- [ ] Task 2

### Phase 2 — <Name>
- [ ] Task 3

## Testing
- Unit tests: what to cover
- Integration tests: what to cover
- Manual verification steps

## Risks & Notes
Any known risks, open questions, or decisions deferred.
```

**Rules:**
- Plans are always phased — break work into discrete, independently deliverable phases
- Every plan must include a **Testing** section covering unit tests, integration tests, and manual steps
- Do not start implementation without a plan issue for any feature that touches more than one file
- Keep the phase checkboxes current as work progresses — the issue is the status
- **Close the issue when the work lands.** Closed issues are the record; never delete them
- Creating, editing, or closing an issue is a **remote action** — ask before running `gh issue …`
  on the user's behalf unless they asked for it

---

## General Rules
- Never mix backend and frontend concerns — they communicate only via the API contract
- Never commit `.env` files — use `.env.example` as the source of truth for required vars
- All DB access goes through Django ORM — never raw SQL unless absolutely necessary, and always parameterised
- Prefer explicit over implicit — readable code over clever code
- Write for the next developer, not just for today
- Keep `docs/` up to date — code changes and doc changes travel together

---

## Absolute Don'ts

These actions must **never** be performed without explicit user confirmation:

**Git operations — never run autonomously:**
- `git commit` — do not commit code on the user's behalf
- `git push` / `git push --force` — do not push to any remote
- `git reset --hard` — destructive, cannot be undone
- `git rebase` / `git merge` on shared branches
- `git branch -D` — do not delete branches

**GitHub (remote) — never run autonomously:**
- `gh issue create` / `gh issue edit` / `gh issue close` — plans live here, but creating or closing
  one on the user's behalf needs confirmation
- `gh pr create` / `gh pr merge`

**File system:**
- `rm -rf` on any non-temporary directory
- Deleting migration files

**Infrastructure:**
- Running `docker compose down -v` (destroys DB volumes)
- Modifying shared environment files (`.env`) in-place

**Process:**
- Bypassing pre-commit hooks (`--no-verify`)
- Dropping or truncating database tables directly
