# model_playground

Run machine-learning models — LLMs, computer vision, and custom networks —
**directly in the browser on your GPU via raw WebGPU** (WGSL compute shaders, no
ML framework). A decoupled monorepo:

- **`backend/`** — Django REST Framework API (Python 3.13, PostgreSQL, Celery).
  A **model registry**: serves the catalog and stores inference-run metadata.
  It does **not** run inference.
- **`frontend/`** — React 19 SPA (TypeScript, Vite, TanStack Router, TanStack
  Query). Owns the **WebGPU runtime** (`src/webgpu/`); inference runs client-side
  in a Web Worker.

Model weights are fetched by the browser from a model host/CDN — never proxied
through Django. See [`docs/explanations/webgpu-inference.md`](docs/explanations/webgpu-inference.md).

---

## Quick Start

```bash
# 1. Clone
git clone <repo-url> && cd model_playground

# 2. Create env files
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env

# 3. Start everything
docker compose up

# 4. Run migrations (first time)
docker compose exec backend python manage.py migrate
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000/api/ |
| Django admin | http://localhost:8000/admin/ |
| Health check | http://localhost:8000/api/health/ |

---

## Documentation

| Doc | Description |
|-----|-------------|
| [docs/explanations/webgpu-inference.md](docs/explanations/webgpu-inference.md) | How in-browser inference works (raw WebGPU pipeline) |
| [docs/guides/adding-a-model.md](docs/guides/adding-a-model.md) | Add a model: WGSL kernel + registry entry |
| [docs/guides/local-setup.md](docs/guides/local-setup.md) | Full local dev setup (Docker + without Docker) |
| [docs/guides/onboarding.md](docs/guides/onboarding.md) | New developer orientation |
| [docs/standards/api-contracts.md](docs/standards/api-contracts.md) | All API endpoints, request/response shapes |
| [docs/explanations/architecture.md](docs/explanations/architecture.md) | Monorepo structure and design decisions |
| [docs/explanations/auth-flow.md](docs/explanations/auth-flow.md) | JWT auth flow end to end |
| [docs/guides/celery_setup.md](docs/guides/celery_setup.md) | Celery + Redis async task setup |
| [docs/plans/](docs/plans/) | Phased feature plans and ADRs — [`in-progress/`](docs/plans/in-progress/) (active) and [`completed/`](docs/plans/completed/) (kept as a record) |

**AI assistants:** [CLAUDE.md](CLAUDE.md) (Claude Code) and [.github/copilot-instructions.md](.github/copilot-instructions.md) (GitHub Copilot) describe the project conventions for AI tooling. Keep both in sync when conventions change.

---

## Project Structure

```
/
├── backend/          Django REST API
│   ├── core/         Settings, URLs, WSGI
│   ├── apps/         Domain applications
│   │   ├── accounts/ User model (email-based), registration, JWT auth
│   │   ├── pages/    Health check
│   │   └── registry/ Model catalog (ModelCard) + inference-run metadata
│   └── manage.py
├── frontend/         React SPA
│   └── src/
│       ├── api/      Axios client, query keys, API functions
│       ├── components/ui/  shadcn/ui components
│       ├── webgpu/   Raw-WebGPU runtime (device, buffers, pipeline, worker, shaders/)
│       ├── hooks/    Custom hooks (auth, useWebGPU, useGpuBenchmark, useModels)
│       ├── lib/      cn() helper, date-fns wrappers
│       ├── routes/   TanStack Router file-based routes (incl. /playground)
│       ├── schemas/  Zod validation schemas
│       ├── store/    Zustand global state slices
│       └── types/    TypeScript types from API contracts
├── docs/             Project knowledge base
└── docker-compose.yml
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| In-browser inference | Raw **WebGPU** + WGSL compute shaders (Web Worker) |
| Backend language | Python 3.13 |
| Backend framework | Django 5 + Django REST Framework |
| Auth | JWT (`djangorestframework-simplejwt`) |
| Database | PostgreSQL 16 |
| Async tasks | Celery + Redis (`django-celery-results`, `django-celery-beat`) |
| Dependency manager | [uv](https://github.com/astral-sh/uv) |
| Frontend language | TypeScript |
| Frontend bundler | Vite |
| UI framework | React 19 |
| Routing | TanStack Router |
| Server state | TanStack Query v5 |
| HTTP client | Axios |
| Styling | Tailwind CSS v4 + shadcn/ui |
| Forms | React Hook Form + Zod |
| Global state | Zustand |
| Testing (frontend) | Vitest + React Testing Library |
| GPU types | @webgpu/types |
| Charts | ECharts (lazy) + Recharts |
| Utilities | date-fns |
| Container | Docker Compose |
| Task runner | [just](https://github.com/casey/just) |
