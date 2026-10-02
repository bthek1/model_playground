# Local Development Setup

This guide walks through getting the full stack running on your local machine using Docker Compose, or without Docker for rapid backend/frontend iteration.

---

## Prerequisites

- **Docker** ≥ 24 and **Docker Compose** ≥ 2.20
- **Python** 3.13 + [uv](https://github.com/astral-sh/uv) (for running the backend locally without Docker)
- **Node.js** 20 + **npm** (for running the frontend locally without Docker)

---

## Option A — Full Stack with Docker Compose (recommended)

### 1. Clone the repo

```bash
git clone <repo-url>
cd django_react_template
```

### 2. Create the environment file

```bash
cp .env.example .env      # or: just env-init
```

There is **one** `.env`, in the project root, read by both halves: Django
(`core/settings/base.py` reads `../.env`, and `backend/.envrc` sources it for
direnv), Vite (`envDir` points at the root) and every service in
`docker-compose.yml`. Only `VITE_*` keys reach the browser bundle, so the
backend secrets beside them are never shipped to the client.

Two tests keep the layout honest, because every way it drifts is silent:

- `backend/tests/test_env_config.py` reads the files themselves — the settings
  read `ENV_FILE` (the root `.env`) and nothing else; every variable a settings
  module requires *without a default* is in `.env.example` (dev) or
  `.env.prod.example` / the prod compose file (prod); neither example invites
  AWS credentials; no config file names `backend/.env` or `frontend/.env`; and
  every `docker-compose.prod.yml` command in the justfile and the docs' code
  blocks carries `--env-file .env.prod`.
- `frontend/src/__tests__/envExposure.test.ts` runs a real Vite build against a
  sentinel `.env` and asserts a backend key never reaches the output, and that
  every `import.meta.env.VITE_*` the app reads is documented in `.env.example`.

`just env` prints the file with every value masked; it holds real secrets, so
nothing in the toolchain `cat`s it.

The default values in `.env.example` are pre-configured to work with Docker Compose (`db` hostname, etc.). No changes are needed for local development.

Production does not use this file — it reads `.env.prod`; see
[deployment.md](deployment.md).

> **Upgrading an older checkout?** Before #61 there were two files,
> `backend/.env` and `frontend/.env`. Neither is read any more. Concatenate
> their values into the root `.env` (`cat backend/.env frontend/.env >> .env`,
> then remove duplicate keys), check the app still boots, and then delete the
> two old files by hand — both hold real secrets, so the deletion is
> deliberately not scripted.

### 3. Start all services

```bash
docker compose up
```

This starts:
- `db` — PostgreSQL 16 on port `5432`
- `backend` — Django dev server on `http://localhost:8000`
- `frontend` — Vite dev server on `http://localhost:5173`

### 4. Run database migrations (first time only)

In a separate terminal:

```bash
docker compose exec backend python manage.py migrate
```

### 5. Create a superuser (optional)

```bash
docker compose exec backend python manage.py createsuperuser
```

### 6. Verify

- Frontend: [http://localhost:5173](http://localhost:5173)
- Backend API: [http://localhost:8000/api/health/](http://localhost:8000/api/health/)
- Django admin: [http://localhost:8000/admin/](http://localhost:8000/admin/)
- Flower (Celery monitoring): `just flower` then open [http://localhost:5555](http://localhost:5555)

> **The Vite dev server runs over HTTPS** (self-signed cert via
> `@vitejs/plugin-basic-ssl`). WebGPU's `navigator.gpu` is only exposed in a
> [secure context](https://developer.mozilla.org/en-US/docs/Web/Security/Secure_Contexts)
> — HTTPS or `localhost` — so accessing the app from another machine over a plain
> `http://<lan-ip>:5180` origin would hide WebGPU entirely. Use
> `https://<lan-ip>:5180` and accept the one-time self-signed-cert warning
> (in Chrome: **Advanced → Proceed**, or type `thisisunsafe` on the warning page).

---

## Option B — Backend Only (no Docker)

### 1. Set up environment

```bash
cp .env.example .env      # in the project root, not in backend/
# Edit .env: set DATABASE_URL to SQLite or a local Postgres instance
cd backend
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Run migrations and start server

```bash
uv run python manage.py migrate
uv run python manage.py runserver
```

Backend is available at `http://localhost:8000`.

---

## Option C — Frontend Only (no Docker)

### 1. Set up environment

```bash
cp .env.example .env      # in the project root, not in frontend/
cd frontend
# VITE_API_BASE_URL stays empty — the app calls /api on its own origin and the
# Vite dev server proxies it to VITE_API_PROXY_TARGET (default localhost:8006).
```

### 2. Install dependencies and start

```bash
npm install
npm run dev
```

Frontend is available at `http://localhost:5173`.

---

## Running Tests

### Backend

```bash
cd backend
uv run pytest
```

### Frontend

```bash
# Run the test suite
just fe-test
# or directly:
cd frontend && npm test

# Type-check + bundle
just fe-build
# or directly:
cd frontend && npm run build
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `db` container not ready | Wait a few seconds and retry `migrate` — Postgres takes a moment to initialise |
| Port already in use | Change the host port in `docker-compose.yml` (e.g. `"8001:8000"`) |
| `uv: command not found` | Install uv: `curl -Ls https://astral.sh/uv/install.sh \| sh` |
| Hot reload not working in Docker | Ensure the volume mount for the source directory is configured in `docker-compose.yml` |
