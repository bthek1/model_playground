# Copilot Instructions

> This file describes the project conventions for GitHub Copilot. [`CLAUDE.md`](../CLAUDE.md)
> is the Claude Code counterpart and covers the same conventions — keep the two in sync.

## Project Overview
**Model Playground** — a web app for running ML models (LLMs, computer vision, custom networks)
**directly in the browser on the user's GPU/CPU**. Two client-side inference paths coexist: a
**raw-WebGPU runtime** (`src/webgpu/`, hand-written WGSL compute shaders) for custom kernels, and
**Transformers.js / ONNX Runtime Web** for running pretrained models (the audio and vision tasks) in the UI.
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
  `DivergingLegend` in `components/viz/heatmap.tsx`), a pan/zoom surface (`PanZoom` in
  `components/viz/PanZoom.tsx`, whose optional `onScaleChange` lets a canvas keep its strokes a
  constant size on screen), theme-token (`--chart-1…5`) colors resolved
  via `getCSSVar()`, and lazy charts. The Training route (`components/training/`) and Tensor route
  (`routes/tensor.tsx`) are the reference callers. Reuse those primitives; render the real
  `ModelCard`/weight data, not stock diagrams; pass both themes and every WebGPU status.
- **The system panel** (`src/telemetry/`, `components/telemetry/`, `components/layout/RightPanel.tsx`)
  reports **load and capacity, never utilisation**: a browser exposes no host CPU percent, no GPU
  utilisation and no VRAM. Every metric is `{ status: "ok", value }` or
  `{ status: "unavailable", reason }` — never a zero for "unknown" — and each card states what its
  number actually is. It samples nothing while closed or while the tab is hidden: one 1 Hz
  interval, ring buffers behind refs (not Zustand, not TanStack Query), the Cache Storage walk
  every 10th tick, a slow tick skipped rather than queued. Sparklines are inline SVG (§5), not
  ECharts. GPU bytes are a ledger, not a probe (`webgpu/allocations.ts`): free with
  `releaseBuffer()`, and the worker realm publishes its ledger over a `MessagePort` from
  `createWebGPUWorker()` (a `BroadcastChannel` would include other tabs). `useModelWorker` reports
  inflight + download bytes to `telemetry/activity.ts`; the worker envelope gains no telemetry
  variant. Full detail: `docs/explanations/telemetry-panel.md`

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
  (model-page-pattern.md §8), never invent an ad-hoc id — `audio-input`/`audio-input-empty` are the
  audio input surface's pair. Each task page test covers: nothing downloads on mount (the hook is
  called with **no `autoLoad` argument**); a refresh leaves the page `idle` **even with the weights
  cached**; **choosing an input runs nothing** (pick a sample while `ready`, assert `run` was not
  called, then press the trigger and assert it was); **the input survives its run** (press twice —
  one decode, two `run` calls); `load`/`retry` fire from the LOAD slot; four slots render with an
  empty OUTPUT; the GENERATE trigger is gated on `ready` while the input sources are not; and each
  error is in its own slot. Playwright page objects mirror this — `ModelPageObject` is the base
  (its `run(name)` presses the RUN trigger), `AudioPage`/`TensorPage` extend it.
- **Reference implementations:** `routes/text-to-speech.tsx` (downloads weights),
  `routes/tensor.tsx` (compile-only), `routes/tasks.$slug.tsx` (the empty case).
  `routes/training.tsx` is the one **documented exception** — a full-bleed canvas HUD that keeps its
  own layout; see model-page-pattern.md §7 before copying it.
- **Two buttons spend anything: LOAD and GENERATE.** SELECT and INPUT are choices — free,
  reversible, committing to nothing. **Only LOAD loads** (`idle` is the default and `autoLoad`
  defaults to `false` everywhere): not on arrival, not on a model change, not on a refresh, and
  **not on a cache hit** — cached weights make the click cheap, not unnecessary, since they still
  cost memory, a GPU device and a warm-up. Show the size estimate and the large-model warning
  *first*. Compile-only tasks (a WGSL pipeline compile) are the one case that may pass
  `autoLoad: true`.
- **Only GENERATE runs.** Picking a sample, dropping a file, finishing a recording, placing a point
  on a picture, or editing a prompt/label/template beside the input all land in INPUT and stop.
  Thirteen vision routes used to run on decode, and the four audio routes had no input stage at all
  — browsing five samples cost five inferences and there was no way to re-run the clip you had.
  The **input sources are not gated on `ready`**; only the GENERATE trigger is. A control that
  merely re-reads a result in hand still re-derives without a press, because it spends nothing.
- **The input is held state.** `hooks/useImagePick.ts` / `hooks/useAudioPick.ts` with
  `components/vision/ImageSourcePanel.tsx` / `components/audio/AudioSourcePanel.tsx` own the decode
  and hand the run a **copy** — every audio worker detaches the buffer it is given, so
  `useAudioPick.take()` is `clip.audio.slice()` and `toPayload` copies by default. Neither pick hook
  takes an "on picked" callback; the parameter is gone, not merely unused. OUTPUT renders the
  frame/clip captured **inside** the run, never the input currently held.
- **A refresh restores the selection, not the session and not the load.** A Worker cannot outlive a
  page load, so `store/models.ts` persists the selected model and *only* that — with `partialize`,
  so a stale `autoResume` key from an older build cannot revive auto-resume. `model/cache.ts` still
  probes Cache Storage, but only to say "Load model (cached)": an informed click, never an absent
  one.
- **The progress bar reports the aggregate** (`model/progress.ts`): monotonic percent by bytes over
  all files, indeterminate until a size is known (and then `aria-valuenow` is omitted), warm-up as
  its own phase, and no ETA. Never render a raw per-file `progress_callback` payload — it restarts at
  zero for each of a model's 4–8 files.
- **Slot rules:** SELECT disabled while `loading`/`running`; LOAD is the only slot with a progress
  bar; the GENERATE trigger is `disabled={!ready || running || <no input>}` while the input sources
  above it are not gated at all; OUTPUT always renders (empty / running /
  result / error) so the page never jumps when a result lands.
- **Errors render in the slot that produced them** — load error in LOAD, decode error in RUN,
  inference error in OUTPUT. A failed inference must leave the page usable.
- Backend selection (`webgpu` → `wasm`) resolves once in the worker before `ready` and is fixed for
  that worker's life. No silent re-negotiation — it would make the timings the user reads meaningless.

**In-browser pretrained models (`src/audio/`, `src/vision/`) — Transformers.js / ONNX Runtime Web:**
- This is the carve-out from the raw-WebGPU rule above. Pretrained HF checkpoints (audio ASR / TTS /
  classification) run here via `@huggingface/transformers` (plus `kokoro-js` for TTS). Keep it out of
  `src/webgpu/` — the two runtimes never mix.
- **One worker per modality, not per task.** Discriminative tasks share the generic
  `pipeline.worker.ts` (task string travels in the `load` message). ASR keeps its own worker (it
  drives the real-time capture loop). `tts.worker.ts` owns the whole **text→audio** modality —
  Kokoro and MMS/SpeechT5 behind one `TtsSynthesizer` interface (it carried MusicGen too, until
  `/text-to-audio` was cut). Each engine (`asrEngine` / `pipelineEngine` / `ttsEngine`) is a pure message
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
- **The backend probe and the size guardrail are shared and live in `src/model/`** (`backend.ts`,
  `size.ts`). They started under `audio/` and *moved* when vision arrived — never write a second copy.
- **Verify a model id against the Hub before shipping it.** Two catalogue entries once pointed at
  `onnx-community/*` repos that don't exist (401 → "Unauthorized access to file"). `just fe-e2e-models`
  (`e2e/specs/model-ids.spec.ts`, audio + vision) checks all of them in seconds — and checks that each
  vision entry publishes the dtype its backend asks for, since a repo with only an fp32 `model.onnx`
  resolves fine on the API and 404s at load.
- Every catalogue entry carries `params` (millions) driving the **size-before-load guardrail**
  (`model/size.ts` + `components/model/ModelPicker.tsx`): the picker quotes the download for both
  backends and warns past `LARGE_MODEL_BYTES`. Add measured `bytes` when the params estimate would
  mislead (ASR's fp32 decoder makes WASM ~3x the estimate).
- **ASR timestamps are take-relative.** The live loop re-transcribes only the tail 30 s, so the
  model's timestamps restart at 0 on a longer take; `useLiveAsr`'s `shiftChunks()` offsets them by
  the window start before the route renders `m:ss`. Don't render worker `chunks` unshifted.
- **Roadmaps are filtered by a two-part feasibility bar.** A task becomes a page only if it
  **runs client-side** *and* its **cheapest usable checkpoint is under ~500 MB** (measured off
  the Hub, never estimated). The size test is about the **floor, not the ceiling** — a cheap
  default plus a gated heavy option is fine; a page whose every entry is heavy is the failure.
- **"In the browser" ≠ "on the GPU".** Prefer WebGPU, but CPU-only is right when CPU is faster:
  `/vad` is pinned to WASM (Silero's LSTM/`If` ops have no WebGPU coverage, ~100x real time on
  CPU), and Tabular's decision trees are TypeScript in a Worker. A task needing a *server* is
  what fails the test.
- **Measure a download; never estimate it.** Five of six sizes in the NLP roadmap were wrong,
  one by 4x on the recommended default (`deberta-v3-base-zeroshot-v2.0`: quoted ~180 MB, really
  **738.6 MB** — one fp32 `model.onnx`, no q8). An official in-repo export is not automatically
  a quantized one, and for seq2seq sum `encoder_model` + `decoder_model_merged` only.
- **Gate anything heavy** — state the size and speed cost, download nothing until an explicit
  opt-in, assert zero Hub requests before the click in an E2E spec. But **a gate is not a
  substitute for a size that fits**: `/text-to-audio` did all of that in front of MusicGen and
  was cut anyway, because a 599 MB minimum was the *whole* page.
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
  specs (`e2e/specs/audio-models.spec.ts`, `e2e/specs/vision-models.spec.ts`) are the guard, and
  they must assert a **known answer to a known input** — "a result appeared" passes while a
  quantized model calls a tiger a snake. `e2e/specs/model-ids.spec.ts` is the seconds-long half.
  Shared page-object verbs live on `ModelPageObject`; a modality subclass adds only its own.
- Adding a task: catalogue entry → worker (reuse the generic one) → hook → route → `REAL_ROUTES`.
  See `docs/guides/adding-a-model.md` §8 (Transformers.js) or §9 (a bare ONNX graph — DFN3 is the
  reference, `src/audio/vad/` the smaller one to read first).

**Graph machine learning (`/graph`, `/link-prediction`, `/graph-classification`) — the opposite carve-out: no checkpoint at all.**

- A GNN is one sparse gather repeated a few times, so the model is **written as WGSL and trained in
  the tab**. `lib/cora.ts` + `lib/data/cora.bin`, `lib/graphLayout.ts`, `webgpu/gnn.ts` +
  `webgpu/gat.ts`, `webgpu/shaders/gnn_aggregate.wgsl` + `webgpu/gnnRuntime.ts`,
  `webgpu/graphSession.ts`. Full write-up in `docs/roadmaps/graph.md`.
- **Message passing is one scaled gather**, `out[i] = α_i Σ_{j ∈ N(i) ∪ {i}} β_j x[j]`; GCN /
  GraphSAGE / GIN are a choice of the two scale vectors. Three invariants that fail **silently**:
  the **self-loop is not stored** and is added by the kernel (storing it double-counts), the graph
  **must be symmetric** because that is what makes the backward pass's `Âᵀ` the same kernel with
  α and β swapped, and **project before you gather** (`Â(XW)`, never `(ÂX)W`). `lib/cora.test.ts`
  asserts the first two against the real committed binary.
- **GAT is not a scale vector.** Its coefficients are learned per *edge* from the features, so it
  goes through the `Propagator` seam and its gather deliberately does **not** use the shader —
  O(|E|·d) ≈ 0.2 ms beside a projection that already runs on the GPU. The page says which half runs
  where.
- **A wrong aggregation still produces a falling loss and a plausible accuracy curve.** The guards
  are a **finite-difference gradient check** per architecture (`webgpu/gnn.test.ts`) and a GPU-vs-CPU
  kernel cross-check (`e2e/specs/webgpu/graph.spec.ts`). The check must run at a **generic point**
  (zero-init biases put preactivations exactly on ReLU's kink) and must include a **dropout pass**
  (the masked input transpose is unreachable without one).
- **Cora is bundled sparse-encoded, 161 KB** (dense f32 would be 15.5 MB). Input dropout is worth
  several points and is affordable only via the nonzero pattern in both layouts. **The layout is the
  expensive part, not the model** — computed once in the worker, never recomputed on a
  hyperparameter change. Features never cross `postMessage`.
- **Oversmoothing needs a number, and the obvious one is wrong**: use similarity between *adjacent*
  nodes, not all pairs. **GIN's collapse is overflow, not oversmoothing** — `deadFraction` is
  reported separately so the page cannot conflate them.
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
- **A gradient check does not pin a forward pass, and a canvas hides its geometry.** Two
  gaps worth knowing before writing the next kernel or the next visualization: GAT's
  finite-difference checks pass even if the softmax is normalised over the wrong set, so
  `gat.test.ts` asserts the property (every output is a convex combination of its closed
  neighbourhood) and calibrates it by zeroing the attention vectors, which must collapse the
  layer to an exact mean. And happy-dom gives a canvas **no 2D context and no
  `ResizeObserver`**, so a component test reaches only the guarded early-returns — pull the
  geometry out as a pure function (`GraphCanvas.layoutToPixels`) or it is untested. Doing that
  immediately surfaced a drawing centred and then shifted again by half the padding.
- **`e2e/specs/webgpu/` used to skip on every machine**: the fixture probed `navigator.gpu` from
  `about:blank`, an opaque origin and therefore not a secure context. It now probes a served origin,
  and the webgpu project passes `--enable-unsafe-swiftshader` so a runner with no `/dev/dri` still
  executes real WGSL.

**In-browser vision (`src/vision/`) — the second modality on the same path:**

- Same shape as `src/audio/`: one generic worker for every discriminative task (`vision.worker.ts`,
  task carried in the `load` message), a pure `engine.ts` owing the same three behaviours, thin task
  hooks over `useVisionPipeline`. Shipped: `/image-classification`, `/depth`, `/object-detection`,
  `/segmentation`, `/zero-shot-image-classification`, `/zero-shot-object-detection`,
  `/image-features`, `/mask-generation`, `/pose`, `/video-classification`,
  `/background-removal`, `/super-resolution`, `/image-to-3d` — fourteen of twenty. The rest stays
  on a server, with a reason per task — `docs/roadmaps/vision.md` §3.12.
- **The shared page pieces already exist — reuse them, do not re-copy.** `useImagePick` (file / drop
  / sample, and the one-object-URL-at-a-time rule), `useCameraFrames` (open → grab → `downscale` →
  one frame in flight), `components/vision/ImageSourcePanel` (the RUN slot's input surface) and
  `OverlayCanvas` (a canvas at source resolution, painted by a callback). `ImageSourcePanel` renders
  the *input* only: an overlay is a result, and results live in OUTPUT.
- **A `RawImage` does not survive `postMessage`** (class instance → methodless clone → the pipeline
  rejects it). Send `toPayload(image)` and rebuild with `fromPayload` in the worker. `toPayload`
  copies by default, because transferring the buffer blanks the preview the page is still showing;
  `{ copy: false }` is for a spent webcam frame.
- **Nor does a `Tensor`, and that one throws**: its `data`/`dims` are prototype getters over an
  internal ORT tensor, so structured clone refuses it (`#<_Tensor> could not be cloned`). Every
  result passes through `toCloneable` (`vision/serialize.ts`) before the engine posts it. Depth
  estimation is the task that hits it, and only a *real* load surfaces it.
- **Never resize or normalise for the model** — `AutoProcessor` reads the model's own
  `preprocessor_config.json`, and that file *is* the input contract. `downscale()` caps the *source*
  resolution (1280x720 costs ~4x 640x480) and is not preprocessing.
- **A quantized export can be wrong rather than merely worse.**
  `onnx-community/mobilenetv4_conv_small` at q8 calls a tiger "sidewinder, horned rattlesnake" (44%);
  at fp32 it says "tiger" (62%) — depthwise-separable convs are the classic int8 casualty. Pin
  precision per backend with `VisionModel.dtypes` (`{ wasm: "fp32" }`) and supply measured `bytes`.
  Assert a known label on a known image in the `@slow` spec (`just fe-e2e-vision`), and say in
  the comment whether the pin is a measurement or a precaution.
- **Never queue frames.** `useLiveFrames` grabs the next frame only when the previous result is back;
  `useCamera` owns teardown (a leaked `MediaStream` leaves the webcam light on).
- **Normalise a single-channel map before painting it** (`drawHeatmap`) — relative depth has no fixed
  scale, and without it the canvas is uniformly black or white. Say in the UI that the depths are
  *relative*, not metres; the direction of the ramp is read from the catalogue entry, because Depth
  Anything emits inverse depth (big = near) and a metric model emits metres (big = far).
  `DepthModel.metric` is that seam and no entry sets it — Depth Pro was cut at 1009 MB.
- **`percentage: false` on detection**, pinned in `useObjectDetector`. The default returns 0-1
  fractions and `drawBoxes` wants pixels; backwards, every box lands in the top-left corner. Boxes
  then need `scaleDetections` to map from the downscaled inference frame back onto the source.
- **`/zero-shot-image-classification` owns an engine instead of using a pipeline** (`src/vision/zeroshot/`):
  the CLIP/SigLIP towers are driven separately so label embeddings are encoded once and reused, which
  the pipeline cannot do. Splitting them means owning normalise/scale/softmax in `zeroshot/scoring.ts`
  — `exp(logit_scale)` and `logit_bias` are read from each checkpoint's weights, and a wrong value
  fails **silently** (scores stay in [0,1], ranking unchanged). Pinned by `just fe-e2e-zeroshot`,
  which compares against the full graph.
- **The zero-shot pipeline templates your prompts again.** Its default `hypothesis_template` is
  `"This is a photo of {}"`; pass `hypothesis_template: "{}"` when you template your own, and keep
  catalogue labels as bare nouns.
- **A threshold, an opacity or a class toggle re-derives; it never re-runs.** The model is asked once
  and the controls filter what it returned — the same pure-derivation trick `/vad` uses. `/pose` is
  the documented exception and says so on the page: filtering people *after* the fact would mean
  running the pose model on people already excluded, and that pass is the expensive half.
- **Four vision tasks own an engine rather than riding the generic worker**, always on the same
  criterion — it is not a plain `pipeline()` call. `vision/zeroshot/` (split towers, cached label
  embeddings), `vision/sam/` (two graphs, encode once / decode many), `vision/pose/`
  (two models live).
- **`/pose` is the single deliberate exception to "one model live at a time."** Its catalogue entry
  names both checkpoints and quotes the **combined** download. Two consequences: `model/progress.ts`
  is keyed on **repo + file** (both publish an `onnx/model_fp16.onnx`, and the second was
  overwriting the first — the bar reached 100% halfway through), and both models are disposed with
  `Promise.allSettled`, so a detector whose teardown throws does not skip the larger one.
- **`VisionModel.backends` is enforced, not merely declared.** `model/useBackendProbe.ts` answers
  "what would this load on" before anything downloads; `ModelPicker` disables a model the machine
  cannot run with the reason on the row. A `null` probe gates nothing.
- **`VisionModel.graphs` names the ONNX files an entry downloads** (default `["model"]`). CLIP as a
  feature extractor loads `vision_model.onnx`, SAM ships two graphs, the VLM entries three —
  `just fe-e2e-models` checks *those* files, since a check hard-coded to `model.onnx` looks at the
  wrong file and passes.
- **Two coordinate round-trips fail silently, and geometry is the only assertion that catches them.**
  SAM: a click is in CSS pixels while the canvas is sized to the source, so `OverlayCanvas`'s
  `onPick` converts — a mis-mapped point returns a plausible mask of whatever is there. Pose:
  `post_process_pose_estimation` scales the heatmap peak by the box's *size* and never adds its
  *origin*, so `vision/pose/pose.ts` does. The `@slow` specs assert a mask **coverage band** and the
  nose **above** the ankles.
- **`/video-classification` is a frame-level baseline and says so as a correctness requirement.**
  No real video transformer has an ONNX export; the limitation sits next to the result and an E2E
  spec asserts the copy.
- **Three routes cover *part* of a taxonomy slug, and each page says which part.**
  `/super-resolution` is `image-to-image`'s single-pass half (editing is diffusion);
  `/image-to-3d` is `image-to-3d`'s depth-to-cloud half (reconstruction is SD-derived). A route
  that quietly answers a smaller question than its name promises is the failure mode; the header
  sentence is the fix, and a test asserts it.
- **`/background-removal` added a Computer Vision row the Hub does not have** (Transformers.js
  invented the `background-removal` pipeline as a segmentation subclass), taking the category from
  19 rows to 20. It is also the only route whose blocking question was a **licence**:
  `briaai/RMBG-1.4` is Creative Commons **non-commercial** in an MIT repo, so Apache-2.0
  `Xenova/modnet` is the default and RMBG is offered with the restriction rendered beside the
  choice (`components/vision/LicenceNote.tsx`). Read the model card before writing the catalogue
  entry — it is the one thing that can invalidate a finished route.
- **Never threshold a matte.** `vision/matte.ts` blends (`src*a + bg*(1-a)`) and the PNG keeps its
  alpha; a hard threshold makes a matting model a segmenter with extra steps. Two traps behind it:
  `ImageSegmentationPipeline` takes an **argmax** branch whenever the processor exposes a
  `post_process_*_segmentation` method (both entries publish a plain `ImageFeatureExtractor`, so
  they escape it — check a third one's `preprocessor_config.json`), and **MODNet is a *portrait*
  matting model** which returns a near-empty matte rather than an error on anything else. The
  `@slow` spec measured 0.2% coverage on a car before `PORTRAIT_SAMPLES` existed.
- **Tiling is `/super-resolution`'s whole correctness surface, and it fails silently.**
  `vision/tile.ts` is pure so it can be pinned: normalising by *accumulated weight* makes the
  identity round-trip exact, the `+0.5` in the feather stops a zero-weight seam painting a black
  line, and only the top-left `scale x tile` region of a patch is read because
  `Swin2SRImageProcessor` pads up to a multiple of 8 and a padded patch drifts every later tile.
  A mis-assembled upscale is perfectly sharp — `just fe-e2e-superres` scores it by PSNR against a
  ground truth the spec constructs itself. A run is many inferences, so the page quotes tiles and
  seconds **before** the button and offers Stop; `useSuperRes.running` covers the whole sequence,
  because the pipeline's own inflight count drops to zero between tiles.
- **`/image-to-3d` is the one page where both runtimes appear, and they still do not mix.**
  Inference is `src/vision/`, the unprojection is pure arithmetic (`vision/pointCloud.ts`), the
  render pass is hand-written WGSL in `src/webgpu/` (`pointRenderer.ts` + `shaders/points.wgsl` —
  the repo's first *render* pipeline). They meet in the route as a `Float32Array`; neither module
  imports the other. Three things it settled: **inverse depth means a big value is *near***, so
  distance is its reciprocal and getting it backwards turns the scene inside out while still
  looking like a point cloud; **bounds are read back out of the float32 buffer**, not from the
  doubles that wrote them; and WebGPU's `point-list` is always one pixel, so points are
  **instanced quads** with depth testing. Vertex buffer once per inference, camera uniform once
  per frame. The focal length is an **assumption** — relative depth carries no intrinsics — and the
  page says so beside the slider. It degrades to the depth map plus an explanation when
  `detectWebGPU()` is not `ready`.
- **When you pin `dtypes`, say whether it is a measurement or a precaution.** They are different
  claims and only one is evidence. Both current pins are measurements now
  (`/image-classification`'s tiger, `/super-resolution`'s PSNR), but a precaution is fine to
  ship as long as it says so and names the spec that would settle it. Either way the entry
  then owes **measured** `bytes`.

**In-browser vision-language models (`src/multimodal/` — `/image-text-to-text`, `/visual-question-answering`, `/video-text-to-text`) — the third modality, the first streaming one, and three routes over one engine:**

- **There is no `image-text-to-text` pipeline in transformers.js 4.2.0.** `SUPPORTED_TASKS` has 25
  entries and that is not one, so the worker drives `AutoModelForImageTextToText` + `AutoProcessor`
  directly. This is the **third** page planned around a pipeline that could not carry it (MusicGen
  needed `MusicgenForConditionalGeneration`; Florence-2 needed `Florence2ForConditionalGeneration`
  because `image-to-text` resolves via `AutoModelForVision2Seq`, whose registry has no `florence2`).
  **Check `SUPPORTED_TASKS` before planning around a pipeline.** Full write-up in
  `docs/roadmaps/multimodal.md`.
- **`q4f16` via `vlmLoadOpts()`, never a literal in a worker** — `asrLoadOpts` is the precedent.
  `loadOpts()`'s fp16 is 514 MB for SmolVLM-256M against 189 MB at q4f16. `Dtype` gained `q4f16`/`q4`
  and `BYTES_PER_PARAM` gained entries for both: a widened `Dtype` without them renders **"NaN MB"**
  on a real page with nothing failing on the way there.
- **A q4f16 size estimate is wrong, and worse the smaller the model is.** SmolVLM-256M's
  `embed_tokens_q4f16.onnx` is 56.8 MB — the *same size as its fp16 build*, because the embedding
  table is not 4-bit quantized at all, which is 30% of the download. Every VLM entry carries
  **measured `bytes`**, and `just fe-e2e-models` re-checks them against the Hub. Two related traps:
  Qwen3-VL keeps its weights in external `.onnx_data` files (summing only the `.onnx` stubs measures
  a 1373 MB model at 1.2 **MB**), and the roadmap's Qwen2-VL-2B is 2668 MB at q4f16, not ~1.1 GB.
- **`ModelResponse<TResult, TPartial = never>` gained a `partial` variant** — progress *inside* one
  run, correlated to the request id. Machine A stays `ready`, `running` stays an inflight count, and
  a partial whose request already settled is **dropped** so a late chunk cannot repaint a finished
  answer. The default generic is what made it free: the arm is uninhabited for every non-streaming
  worker, so no existing engine, switch or test changed. NLP text-generation needs the same thing,
  which is why it is in the shared envelope rather than a private protocol.
- **`apply_chat_template` is not decoration, and getting it wrong has no error attached.** Each
  checkpoint has its own image placeholder and turn markers; a hand-built prompt produces a fluent,
  confident sentence that does not answer the question. Three silent traps: `{ type: "image" }` is a
  **slot** filled positionally from the image list, `add_generation_prompt` is what makes the model
  *answer* rather than continue the question, and `generate` returns **prompt + answer** so the
  prompt's tokens must be sliced off before decoding.
- **512 is SmolVLM's own tile size, not a round number.** Its `preprocessor_config.json` has
  `do_image_splitting: true` with `max_image_size.longest_edge: 512`, so a 2048px input is cut into
  up to a 4x4 grid **plus a global view** — seventeen encodes for one question. Downscaling the
  source to 512 produces one tile: ~64 image tokens instead of over a thousand. §3.3's DocVQA page
  must set its **own** number (a document needs pixels), not inherit this one.
- **The encode gets its own state in OUTPUT.** The pause before the first token is seconds, and an
  unlabelled pause is indistinguishable from a hang. The question is held INPUT — typing it, tapping
  a preset and picking an image all run nothing; only GENERATE spends. The answer is labelled with
  the question it was **actually** asked, captured inside the run so editing the box afterwards
  cannot relabel a result on screen.
- **One VLM live at a time, no exception** — these are the largest downloads in the app and a leaked
  session ends the tab. Null the reference *first*, then dispose.
- **"Has a GPU" is not the gate — `shader-f16` is.** An adapter without it loads `q4f16` weights
  happily, reports `ready`, then fails on the **first operator** of every run (`Program Gather
  requires f16 but the device does not support it`) — the worst outcome available, because the
  download is already paid for. `supportsShaderF16()` is the real probe;
  `useBackendProbe({ requireShaderF16: true })` folds it into the answer so the picker disables the
  row *before* anything is fetched, and the page names the missing feature rather than saying
  "resolved to wasm" on a machine that plainly has a GPU.
- **Two of the repo's own documents disagreed about Qwen3-VL-2B**, and the ceiling won: 1373 MB is
  past `adding-a-task-page.md` §0's ~1 GB line and past `size.test.ts`'s budget, so it is absent and
  gets its own plan. SmolVLM-500M (358 MB) is the second rung.
- **`components/Markdown.tsx` silently drops every prop but `{children, className}`** — a
  `data-testid` passed to it never reaches the DOM (the deleted `/image-to-text` passed one that never
  resolved). Put the testid on a wrapper.
- **`just fe-e2e-vlm` is the only test that can catch a broken chat template** and it needs a real
  GPU — the models are WebGPU-only by catalogue declaration. It asserts a **known answer on a known
  image**; "some text appeared" would pass straight through the failure.
- **Three routes, one engine — the point and the hazard.** All three share one worker, one
  `engine.ts`, one `useVlm` and one `VlmRun` envelope; they are separate routes because the Hub has
  separate tags and a user looking for VQA does not click "Image Text to Text". The rule that keeps
  them from drifting into three catalogues and three hooks: **a change one page needs belongs in the
  shared hook, and the other two get it.** A reviewer should be able to diff two route files and see
  only the question-shaping difference.
- **`/visual-question-answering` downloads nothing new; its whole mechanism is
  `multimodal/prompt.ts`.** `composePrompt(question, { terse })` returns the exact string that will
  be sent *and* its cap — the question as typed at 128 tokens, or plus `Answer in one word.` at 16.
  The terse cap is **16, not 1**: the instruction is the mechanism and the cap is a backstop, and a
  model truncated mid-word would make the demonstration indistinguishable from the scissors. A
  question that already asks for brevity is **not instructed again** (ordering a small decoder twice
  makes it answer the instruction). The composed prompt is on screen **before** the click and labels
  the answer after it — rewriting a prompt silently is the `hypothesis_template` problem again. The
  toggle looks like a filter, which is why flipping it must run nothing.
- **`/video-text-to-text` is a frame sampler plus an image model and says so beside the result** —
  a correctness requirement, the same one `/video-classification` carries, asserted by an E2E spec.
  `SmolVLM2-256M-Video-Instruct`, 189.2 MB measured, `model_type: smolvlm` (4.2.0 defines it as a
  subclass of `idefics3`), in its own `VIDEO_VLM_MODELS` array — the pages share the worker and the
  type, not the list.
- **N images, not one, and the widening was strictly additive.** `VlmRun.image` → `VlmRun.images`,
  an **ordered list**, with the chat template's `{ type: "image" }` slots filled **positionally**;
  the count is derived from the list's own length at the call site rather than passed beside it. A
  single-image run is a one-element list, and `/image-text-to-text`'s route tests are what proved the
  shipped page unchanged.
- **Frames multiply the tile problem rather than adding to it.** A 640x360 frame is five tiles, so
  eight of them is forty encodes for one question. `MAX_FRAME_SIDE` is 512 — the model's own
  `video_sampling.video_size.longest_edge`, quoted rather than inherited from `MAX_INFERENCE_SIDE`.
  The frame count is the cost dial (64 image tokens each, attended over for every generated word),
  capped at **8**; the model's config allows 64, which is a number for a server.
- **Sampling is by count, not by rate, and it is shown.** `multimodal/frames.ts` samples the centre
  of each of N equal slices (0 is usually a black frame; `duration` is past the last decodable one).
  Deliberately *not* `vision/video.ts`'s `frameTimes`, which samples at fps because
  `/video-classification`'s frames are independent passes — here they share one prompt, so eight
  means eight whatever the clip's length. The decode is still `sampleVideo`: it grew a `times` option
  (a **function of the duration**, which only the decoder has read by then) rather than a second copy.
  The filmstrip in OUTPUT is the frames the model was actually given.
- **The reverse toggle is the one control on these pages that legitimately spends.** If the answer
  does not change, the model is describing a picture rather than reading a sequence — the usual
  result at this size, and the finding rather than the failure. It cannot re-derive, so flipping runs
  nothing and the next GENERATE is a real second inference, which the page says before the click. The
  `@slow` spec asserts the **re-run**, not a difference in the answer. Decoding is cached on
  **(clip, frame count)** and deliberately not on order; `useVideoPick` owns that and the
  one-object-URL rule.
- **`just fe-e2e-videovlm` is the only guard on the multi-image template**, as `fe-e2e-vlm` is for
  the single-image one: N frames out of step with N slots produces a fluent answer about the wrong
  pictures, with no error anywhere. Both need a real GPU with `shader-f16`.

**Three routes were built and then cut for size — read this before adding one:**

`/text-to-audio` (MusicGen: **599 MB** WASM / **1127 MB** WebGPU), `/image-to-text`
(Florence-2 544 MB, vit-gpt2 482 MB) and `/document-question-answering` (Donut:
411 MB WebGPU / 597 MB WASM) all shipped, ran correctly, and were removed. Each failed
the same test — `docs/guides/adding-a-task-page.md` §0 question 2: **every checkpoint
the task has is a several-hundred-megabyte download, with no lighter entry to fall back
on.** Two heavy *entries* went with them: Depth Pro (1009 MB) off `/depth` and SigLIP 2
(751 MB) off `/zero-shot-image-classification`.

- **A gate is not a substitute for a size that fits.** All three stated the cost,
  downloaded nothing until an explicit click, and had an E2E spec asserting zero Hub
  requests before it. They were cut anyway. Gate a heavy entry that sits *beside* light
  ones; do not gate a page into existence.
- **The taxonomy row stays; the route goes.** All three fall through to `/tasks/$slug`,
  asserted in `taskTaxonomy.test.ts` so none can be re-mapped without a smaller model.
  The sidebar mirrors the Hub, not our build state.
- **Rewrite a mechanism rather than deleting it with its subject.** `isHeavy` was a Depth
  Pro id check and is now a size threshold (`HEAVY_MODEL_BYTES`), so `/depth`'s gate
  outlived the entry; `DepthModel.metric` stays unset for the same reason.

**Findings they paid for, kept because they outlive them:**

- **Any encoder-decoder whose decoder is quantized cannot open a WASM session** on the ORT
  bundled with 4.2.0 (`qdq_actions.cc:137 … Missing required scale`). ASR hit it first;
  **Donut proved it is not ASR-specific**. Pinned per entry via `dtypes`
  (`{ encoder_model: "q8", decoder_model_merged: "fp32" }`) at ~3x the download — the
  alternative is *no* CPU path. The note lives in `model/backend.ts` beside `asrLoadOpts`;
  generalise that function if a third family hits it. **Only a real browser load catches
  it** — `onnxruntime-node` loads the identical call fine.
- **Check `SUPPORTED_TASKS` before planning around a pipeline**, not after. Three pages
  needed a model class instead: MusicGen (`MusicgenForConditionalGeneration`), Florence-2
  (`Florence2ForConditionalGeneration` — `image-to-text` resolves via
  `AutoModelForVision2Seq`, no `florence2`), `/image-text-to-text` (no such task). Donut
  was the one that passed the check.
- **A pipeline can hardcode one model's prompt.** `DocumentQuestionAnsweringPipeline` bakes
  in `<s_docvqa><s_question>…</s_question><s_answer>`, so a second architecture would be
  prompted with tokens it has never seen — which is why DocVQA could never have had a
  lighter second entry.
- **Never resize for the model, in both directions.** Donut's processor is `do_resize` +
  `do_thumbnail` + `do_pad` at a fixed 2560x1920 and `thumbnail()` **never upscales**, so
  inference cost was constant and a smaller source was *padded*: the plan's resolution
  slider would have traded legibility for no speed. `AutoProcessor` reads the model's own
  config and that file *is* the input contract.
- **`components/Markdown.tsx` drops every prop but `{children, className}`** — put a
  `data-testid` on a wrapper.
- **OCR now has no page**, and **metric depth has no path** (ZoeDepth has no export).

**In-browser NLP (`src/text/` — `/text-classification`) — the fourth modality, and the cheapest module in the app:**

- **There is no decode step, and that is the point.** No text equivalent of `audio/io.ts`
  or `vision/image.ts` exists: the input is already a string, so there is no
  preprocessing and no transport problem — a `RawImage` does not survive `postMessage`
  and a `Tensor` throws outright, while a string crosses as itself. That absence is why
  `/text-classification` establishes the module rather than a more impressive page. Same
  shape as the other three: one generic worker (`pipeline.worker.ts`, task in the `load`
  message), a pure `engine.ts` owing the same three behaviours, a `client.ts`, thin hooks
  over `useTextPipeline`. Add a task by adding its arm to `TextTask` and its warm-up args
  to `warmupArgs()` — a union arm with no caller is an untested branch.
- **Every entry carries measured `bytes` for both backends**, stricter than vision's
  "measure where an estimate would mislead", and a finding rather than a preference: the
  NLP roadmap's size tables were all `q8` figures while `loadOpts()` asks for **fp16 on
  WebGPU**. Roughly double, all the way down the category — it moves several entries
  across `LARGE_MODEL_BYTES` and Summarization's floor from 283.9 MB to 563.6 MB, over
  the feasibility bar. `just fe-e2e-models` re-checks the quoted numbers against the Hub.
- **`q4` is not a lever for an encoder**: `model_q4.onnx` measures *larger* than
  `model_quantized.onnx` on every encoder tried. 4-bit is a decoder format.
- **No `textLoadOpts()`, deliberately** — the `asrLoadOpts`/`vlmLoadOpts` precedent is for
  a family-wide decision and the measurements do not support one. Pins go per entry in
  `dtypes`, each saying whether it is a **measurement or a precaution**.
- **A base model is not a classifier, and it fails silently.**
  `onnx-community/ModernBERT-base-ONNX` was cut from the catalogue against the roadmap's
  own table: no trained head means `LABEL_0`/`LABEL_1` from random weights — confident,
  fluent, meaningless. Read what a checkpoint was fine-tuned *for* before writing its
  entry.
- **`ScoreList` refuses to render a single row.** A classifier's argmax is its least
  informative output — "POSITIVE" looks identical at 0.99 and at 0.51 — so callers pass
  the full label set and a near-tie is stated in words. `SpanOverlay` + `highlight()` is
  the other shared component, built by `/token-classification` against a real model's
  offsets: **slice the original string by character offset, never rebuild from tokens**,
  or the highlight lands a character off and reads as a styling problem.
- **Nothing here is debounced.** The roadmap wants live classification on a 200–300 ms
  pause; the page-pattern rule wins. Typing is INPUT, only GENERATE spends — a debounced
  auto-run is the five-samples-five-inferences failure with a timer in front of it.
- **The head-to-head is a second LOAD, not a toggle**, and the samples are chosen so the
  three models **disagree**. `just fe-e2e-text` asserts a known label on a known sentence
  and pins the comparison *structurally* (SST-2 has two classes, FinBERT three) — asserting
  the rankings differ would pin a property neither model promises.

**Env vars:** Prefix with `VITE_`. Access via `import.meta.env.VITE_*`.

**Commands:**
- Dev server: `just fe-dev`
- Build: `just fe-build`
- Lint: `just fe-lint`
- Test: `just fe-test`
- Test UI: `just fe-test-ui`
- End-to-end: `just fe-e2e` (browsers: `just fe-e2e-install`; UI: `just fe-e2e-ui`)
- Real model loads: `just fe-e2e-slow` (minutes, needs network); ids only: `just fe-e2e-models`;
  speech enhancement only: `just fe-e2e-enhance`; voice activity detection only: `just fe-e2e-vad`;
  vision only: `just fe-e2e-vision` (tens of minutes cold — one route at a time with
  `just fe-e2e-vision-one /pose`); zero-shot scoring parity: `just fe-e2e-zeroshot`;
  super-resolution vs bicubic by PSNR: `just fe-e2e-superres`; link prediction by AUC band:
  `just fe-e2e-link`; graph classification above its baseline: `just fe-e2e-graphcls`;
  text classification by a known label: `just fe-e2e-text`
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
| `just fe-e2e-vision` | Run the @slow vision specs: real loads across all 14 routes (tens of minutes cold) |
| `just fe-e2e-superres` | Run the @slow super-resolution spec: Swin2SR vs a bicubic baseline, by PSNR |
| `just fe-e2e-vision-one <route>` | One @slow vision route at a time |
| `just fe-e2e-link` | @slow link prediction, pinned by an AUC **band** — leakage pushes it up, so a floor would pass with the bug |
| `just fe-e2e-graphcls` | @slow graph classification, pinned **above its majority baseline**, not above chance |
| `just fe-e2e-models` | Check every model id (audio + vision) resolves on the HF Hub (seconds) |
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
│   │   ├── audio/             # Pretrained audio models (Transformers.js; enhance/ + vad/ on bare ONNX)
│   │   ├── vision/            # Pretrained vision models (image I/O, canvas overlays, one worker)
│   │   ├── model/             # Shared task-page plumbing (backend probe, size, worker lifecycle)
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
│   ├── roadmaps/              # Per-category roadmaps, once the category has shipped code
│   └── explanations/          # Concept explanations, design rationale, background context
├── justfile                   # Task runner (use `just --list`)
├── docker-compose.yml
└── README.md
```

---

## Docker & deployment

Two compose files, and they are not interchangeable:

| File | What it is |
|---|---|
| `docker-compose.yml` | **Development.** Bind-mounts source, runs the Vite dev server and `runserver`, throwaway credentials. Frontend on `5180` (HTTPS, self-signed), backend on `8006`. Never deploy it. |
| `docker-compose.prod.yml` | **Production.** Built images, no bind mounts, secrets from a root `.env`, and only Caddy publishes a port. |

Don't inline either file's contents into docs — read the file. An out-of-date
copy in prose is worse than no copy.

Images: `frontend/Dockerfile` is multi-stage (node build → nginx) and
`frontend/Dockerfile.dev` is the dev server; `backend/Dockerfile` is multi-stage
and runs gunicorn. The backend venv lives at **`/venv`, outside `/app`**, because
the dev compose bind-mounts over `/app`.

**The deployment is single-origin over HTTPS, and both halves are requirements,
not preferences:**

- `src/api/client.ts` ships an empty base URL, so the app calls `/api` on its own
  origin. The Vite proxy does that in dev; **nginx does it in production**
  (`frontend/docker/nginx.conf`). Without it the design inverts into cross-origin
  calls that fail on mixed content, CORS and Local Network Access.
- `navigator.gpu` only exists in a secure context. On plain HTTP every model page
  reports `unsupported` on hardware that works — indistinguishable from an old
  browser. Caddy terminates TLS and obtains certificates automatically.

Consequences worth knowing before changing any of it:

- **`SECURE_PROXY_SSL_HEADER` + `X-Forwarded-Proto` is a chain** (Caddy → nginx →
  Django). Break it and `/admin/` rejects every POST on CSRF, or
  `SECURE_SSL_REDIRECT` loops forever. The prod compose sets
  `SECURE_SSL_REDIRECT=False` because Caddy already redirects at the edge.
- **Migrations are opt-in** (`RUN_MIGRATIONS=1`), set on the `migrate` service
  only — replicas racing `migrate` on boot is a real failure mode.
- **WhiteNoise serves Django's static files** from inside the backend container,
  collected at build time. Without it `/admin/` renders unstyled at `DEBUG=False`.
- **nginx must serve `.wasm` as `application/wasm`** — ONNX Runtime's streaming
  compilation refuses anything else — and must keep `index.html` `no-cache` while
  `/assets/` is `immutable`, or a deploy strands browsers on chunk names that no
  longer exist.
- **Model weights never touch the server**; the browser fetches them from the HF
  CDN. Any CSP you add must not block it.

Full procedure, backups and troubleshooting: `docs/guides/deployment.md`.

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
| Merged | Every phase ticked, tests green — `develop` merges into `main` | `git switch main && git merge --no-ff develop` |
| Closed | Complete — the closed issue is the permanent record | `gh issue close <n> --comment "<what landed>"` |

- Write the plan body to a scratch file first, then pass it with `--body-file` — it keeps long
  markdown intact. The scratch file is temporary; **never commit it**.
- Reference the issue from the work: `Closes #<n>` in the commit message or PR body.
- **An issue is a unit of merging, not of branching.** Its commits accumulate on `develop`; the
  merge into `main` is what completes it, and the issue is closed after that merge — not when the
  last commit is written. See the git section under "Absolute Don'ts".
- Roadmap/research issues (label `roadmap`) are **not plans** — they are the per-category research a
  plan gets written from, and they stay open **while the category is still unbuilt**.
- **A roadmap graduates to a file once its first route ships.** At that point it stops being
  research and starts documenting shipped code, which has to be reviewable in the same pull
  request as the code it describes — impossible in an issue body. Move it to
  `docs/roadmaps/<category>.md`, convert the `blob/main` URLs to relative links, and close the
  issue with a comment pointing at the file. `docs/roadmaps/audio.md` is the worked example.

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

**Git operations — the fence is graded, because commits and pushes are not the same risk.**

*Unattended (reversible, and the reflog recovers them):* `git add`, `git commit`,
`git switch`, `git stash` — **on `develop`, never on `main`.** The assistant commits because
it was asked to; the permission only removes the prompt.

*Work lands on `develop`; do not open a branch per issue.* `develop` is the integration
branch and the issues in this repo are phased plans that touch the same files days apart, so
a branch per issue produces branches that conflict with each other rather than with `main`.
Before committing, check `git branch --show-current`; if it isn't `develop`, `git switch
develop` — switch, don't branch. A `<type>/<topic>` branch (`feat/`, `fix/`, `chore/`) is for
when the **user asks** for one — a throwaway spike, or work that must be reviewed as its own
PR — and it branches from `develop`.

*Merging `develop` into `main` is how an issue finishes.* When every phase of the issue is
ticked and its tests are green: `git switch main`, `git merge --no-ff develop` (so the
issue's commits land as one legible unit), push, then `gh issue close <n>`. Merge whole
issues only — half an issue sitting on `develop` is why the merge waits, not a reason to
cherry-pick — and never commit directly on `main`. Both the merge and the push prompt.

*Never run autonomously — confirm first:*
- `git push` — do not push to any remote
- `git rebase` / `git merge`

*Never at all (denied in `.claude/settings.json`):*
- `git push --force` / `git push -f` — drops commits nobody else has fetched
- `git reset --hard` — destructive, cannot be undone
- `git clean` — untracked files were never in the object store; the reflog cannot help
- `git branch -D` — do not delete branches
- `git checkout .` — discards the working tree wholesale

A pattern deny is not a security boundary (`git push origin +main` force-pushes without the
string `--force`). The real protection is branch protection on `main` plus working on `develop`
— see [`docs/guides/ai-guardrails.md`](../docs/guides/ai-guardrails.md).

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
