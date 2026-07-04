# Architecture

## Overview

**Model Playground** runs machine-learning models — LLMs, computer vision, and
custom networks — **directly in the browser on the user's GPU via raw WebGPU**.
Inference never touches the server: weights are fetched to the client and all
compute happens in WGSL shaders on the local device.

It is a **decoupled monorepo**: a Django REST Framework backend and a React SPA
frontend, developed and deployed independently, communicating over HTTP.

```
┌────────────────────────────────────────────────────────────────────┐
│                           Monorepo root                            │
│                                                                    │
│  ┌───────────────────────────┐   HTTP/JSON   ┌──────────────────┐ │
│  │   frontend/  React SPA     │ ───────────► │  backend/  DRF   │ │
│  │                            │ ◄─────────── │  model registry  │ │
│  │  ┌──────────────────────┐  │  catalog +    └────────┬─────────┘ │
│  │  │  WebGPU runtime      │  │  run metadata          │           │
│  │  │  (device → pipeline  │  │               ┌────────▼────────┐  │
│  │  │   → dispatch), in a  │  │               │   PostgreSQL    │  │
│  │  │  Web Worker          │  │               └─────────────────┘  │
│  │  └──────────┬───────────┘  │                                    │
│  └─────────────┼──────────────┘                                    │
│         ┌──────▼───────┐  weights fetched from a model host / CDN  │
│         │  User's GPU  │  (not the Django backend)                 │
│         └──────────────┘                                           │
└────────────────────────────────────────────────────────────────────┘
```

The backend is a **model registry**: it stores the catalog (metadata, weights
URLs, per-model config) and records inference-run metrics reported by the
client. It does **not** run inference. The two applications **share no code**;
the API contract (`docs/standards/api-contracts.md`) is the only interface
between them. How in-browser inference works is covered in
`docs/explanations/webgpu-inference.md`.

---

## Backend Structure

```
backend/
├── core/                  Django project package
│   ├── settings/
│   │   ├── base.py        Shared settings for all environments
│   │   ├── dev.py         Development overrides (SQLite fallback, DEBUG=True)
│   │   ├── prod.py        Production overrides
│   │   └── test.py        Test runner settings
│   ├── urls.py            Root URL configuration
│   ├── wsgi.py
│   └── asgi.py
├── apps/                  All domain applications
│   ├── accounts/          User model, registration, profile
│   │   ├── models.py      CustomUser (UUID PK, extends AbstractUser)
│   │   ├── serializers.py Request/response shapes
│   │   ├── services.py    Business logic (user creation, etc.)
│   │   ├── views.py       Class-based API views
│   │   └── urls.py
│   ├── pages/             Infrastructure endpoints
│   │   └── views.py       /api/health/ liveness check
│   └── registry/          Model catalog + inference-run metadata
│       ├── models.py      ModelCard (catalog entry), InferenceRun (run record)
│       ├── serializers.py Catalog / run request-response shapes
│       ├── services.py    record_inference_run()
│       ├── views.py       ModelCardViewSet, InferenceRunViewSet (DRF router)
│       └── urls.py        /api/registry/…
├── manage.py
├── pyproject.toml         Python dependencies (managed by uv)
└── .env.example
```

### Key design decisions

**One app per domain.** Each `apps/<name>/` package owns a single bounded domain. Cross-domain dependencies go through service functions, not direct model imports from another app.

**Business logic in `services.py`.** Views delegate to service functions — they never contain business rules. This keeps views thin and services testable in isolation.

**Split settings.** `base.py` contains all environment-agnostic config. Each environment file imports from `base` and overrides only what it needs. The active settings module is selected via `DJANGO_SETTINGS_MODULE`.

**UUID primary keys.** All models use `UUIDField(default=uuid4)` as the primary key to avoid exposing sequential integer IDs in the API.

**Registry, not inference server.** The `registry` app is deliberately thin: it
serves catalog metadata and stores client-reported run metrics. Weights live on
a model host/CDN (`ModelCard.weights_url`); GPU compute happens in the browser.
This keeps the backend cheap and stateless with respect to ML workloads.

---

## Frontend Structure

```
frontend/
├── src/
│   ├── api/
│   │   ├── client.ts      Axios instance — base URL, JWT interceptor
│   │   ├── auth.ts        Auth API call functions
│   │   ├── models.ts      Registry API (list/get models, record runs)
│   │   └── queryKeys.ts   Centralised TanStack Query key constants
│   ├── components/        Shared / reusable UI components
│   │   └── ui/            shadcn/ui copy-paste components (Button, Input, Form, Card…)
│   ├── webgpu/            Raw-WebGPU runtime (no ML framework — see below)
│   │   ├── capabilities.ts  detectWebGPU() — adapter/features/limits probe
│   │   ├── device.ts        Shared GPUDevice (memoised, device-lost aware)
│   │   ├── buffers.ts       Storage/uniform buffer + readback helpers
│   │   ├── pipeline.ts      WGSL → GPUComputePipeline
│   │   ├── runtime.ts       runMatmul() reference kernel + benchmark
│   │   ├── worker.ts        Web Worker that owns the device, runs jobs
│   │   ├── workerClient.ts  Main-thread promise API over the worker
│   │   └── shaders/         WGSL compute shaders (matmul.wgsl, relu.wgsl)
│   ├── hooks/             Custom hooks encapsulating business logic
│   │   ├── useAuth.ts     Auth state, login, logout
│   │   ├── useWebGPU.ts   WebGPU capability probe (for the UI)
│   │   ├── useGpuBenchmark.ts  Runs the matmul kernel via the worker
│   │   └── useModels.ts   Model catalog query
│   ├── lib/
│   │   ├── utils.ts       cn() helper (clsx + tailwind-merge)
│   │   └── date.ts        date-fns wrappers (formatDate, formatRelative)
│   ├── routes/            TanStack Router file-based routes
│   │   ├── __root.tsx     Root layout
│   │   ├── index.tsx      Home page
│   │   ├── login.tsx      Login page
│   │   └── playground.tsx GPU capabilities + benchmark + model catalog
│   ├── schemas/           Zod validation schemas (one file per domain)
│   │   └── auth.ts        Login and register schemas
│   ├── store/             Zustand global state (one file per concern)
│   │   ├── ui.ts          UI flags (sidebar, modals)
│   │   └── auth.ts        Client-side auth flags
│   ├── test/
│   │   └── setup.ts       Vitest setup (imports @testing-library/jest-dom)
│   ├── types/
│   │   └── auth.ts        TypeScript types matching API contracts
│   └── main.tsx           App entry point (QueryClient, RouterProvider)
├── vite.config.ts
└── package.json
```

### Key design decisions

**TanStack Router for routing.** Routes are file-based under `src/routes/`. The router generates a fully type-safe route tree. Route loaders prefetch data via the QueryClient before the component renders.

**TanStack Query for server state.** All data fetched from the API lives in the Query cache. Components never manage async loading/error state manually — they call `useQuery` or `useMutation`.

**Axios interceptor for JWT.** A single Axios instance in `client.ts` attaches `Authorization: Bearer <token>` headers and handles silent token refresh on 401 responses. No component ever touches tokens directly.

**No business logic in components.** Components render UI. All logic (auth checks, data transformation, API calls) lives in hooks under `src/hooks/`.

**Tailwind CSS v4 + shadcn/ui for styling.** All components use Tailwind utility classes. shadcn/ui components are copied into `src/components/ui/` via `npx shadcn@latest add <component>` and never modified directly. The `cn()` helper in `src/lib/utils.ts` (backed by `clsx` + `tailwind-merge`) handles conditional class merging.

**React Hook Form + Zod for forms.** Form schemas are defined in `src/schemas/` (one file per domain) using Zod. Components use `useForm` with `zodResolver`. shadcn/ui `Form`, `FormField`, `FormItem`, and `FormMessage` primitives wrap the RHF context.

**Zustand for global UI state.** Lightweight slices in `src/store/` (one file per concern) hold UI flags that don't belong in TanStack Query (e.g. sidebar open/close, logout-in-progress). The `immer` middleware is used for all mutations. Server-fetched data stays in TanStack Query — never in Zustand.

**Vitest + React Testing Library for tests.** Tests run in a `happy-dom` environment configured in `vite.config.ts`. Test files are co-located with the source file they test (e.g. `useAuth.test.tsx` next to `useAuth.ts`).

**Raw WebGPU for inference — no ML framework.** The `src/webgpu/` module talks to
the WebGPU API directly: acquire a `GPUDevice`, compile WGSL into a compute
pipeline, upload inputs to storage buffers, dispatch, read results back. There is
deliberately no Transformers.js / ONNX Runtime / WebLLM dependency — models are
expressed as WGSL kernels for maximum control and a minimal bundle. Heavy compute
runs in a **Web Worker** (`worker.ts`) so the UI thread stays responsive. See
`docs/explanations/webgpu-inference.md` for the full pipeline and
`docs/guides/adding-a-model.md` to add a model.

---

## Data Flow

### In-browser inference flow (WebGPU)

```
Playground route
  └─► useModels() ─► GET /api/registry/models/  (catalog metadata only)
  └─► useGpuBenchmark() / model runner
        └─► workerClient.ts  (postMessage, transfer input buffers)
              └─► Web Worker (worker.ts)
                    └─► runtime.ts
                          ├─ getGPUDevice()            (device.ts)
                          ├─ createComputePipeline()   (pipeline.ts, WGSL)
                          ├─ upload storage buffers     (buffers.ts)
                          ├─ dispatchWorkgroups()       → User's GPU
                          └─ readBackFloat32()          (buffers.ts)
        └─◄ result posted back (output buffer transferred)
  └─► POST /api/registry/runs/  (optional: report metrics)
```

Weights referenced by `ModelCard.weights_url` are fetched by the client from a
model host/CDN — not proxied through Django.


### Authenticated request flow

```
Component
  └─► useQuery / useMutation
        └─► API function (src/api/)
              └─► Axios client (src/api/client.ts)
                    ├─ attaches Authorization header
                    └─► /api/<endpoint>/ (Django backend)
                              └─► DRF view
                                    └─► service function
                                          └─► Django ORM → PostgreSQL
```

### Response flows back up the same chain, populating the Query cache, which triggers React re-renders.

---

## Environment Configuration

| Variable | Where | Purpose |
|----------|-------|---------|
| `SECRET_KEY` | `backend/.env` | Django secret key |
| `DATABASE_URL` | `backend/.env` | PostgreSQL connection string |
| `DJANGO_SETTINGS_MODULE` | `backend/.env` | Which settings file to load |
| `CORS_ALLOWED_ORIGINS` | `backend/.env` | Frontend origin(s) allowed cross-origin |
| `VITE_API_BASE_URL` | `frontend/.env` | Backend base URL for Axios |

All secrets and environment-specific config live in `.env` files that are **never committed**. Use `.env.example` as the template.
