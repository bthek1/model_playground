# Django + React Monorepo — Justfile
# Run `just` or `just --list` to see all available commands.

set dotenv-load := false

# Clear any shell-activated virtualenv so uv uses the project's own .venv
# (avoids "VIRTUAL_ENV does not match project environment path" warning)

export VIRTUAL_ENV := ""

# ── Default ────────────────────────────────────────────────────────────────────

# Show available commands
default:
    @just --list

# ── Docker ─────────────────────────────────────────────────────────────────────

# Start all services (db, backend, frontend)
up:
    docker compose up

# Start all services in the background
up-d:
    docker compose up -d

# Stop all services
down:
    docker compose down

# Stop all services and remove volumes
down-v:
    docker compose down -v

# Rebuild all images
build:
    docker compose build

# Rebuild a specific service: just build-svc backend
build-svc svc:
    docker compose build {{ svc }}

# Tail logs for all services
logs:
    docker compose logs -f

# Tail logs for a specific service: just logs-svc backend
logs-svc svc:
    docker compose logs -f {{ svc }}

# ── Production ─────────────────────────────────────────────────────────────────
# These drive docker-compose.prod.yml, which needs a `.env` beside it — copy
# .env.example and fill it in. See docs/guides/deployment.md.

# Build and start the production stack (Caddy + nginx + gunicorn + Celery)
up-prod:
    docker compose -f docker-compose.prod.yml up -d --build

# Stop the production stack (volumes are kept)
down-prod:
    docker compose -f docker-compose.prod.yml down

# Tail production logs
logs-prod:
    docker compose -f docker-compose.prod.yml logs -f

# Pull code, rebuild and restart — the routine deploy
deploy:
    git pull --ff-only
    docker compose -f docker-compose.prod.yml up -d --build
    docker compose -f docker-compose.prod.yml ps

# Check the production config without starting anything
prod-config:
    docker compose -f docker-compose.prod.yml config

# Generate a value for SECRET_KEY
secret-key:
    @cd backend && uv run python -c "from django.core.management.utils import get_random_secret_key as k; print(k())"

# Django's deployment checklist, under production settings
be-check-deploy:
    cd backend && uv run python manage.py check --deploy

# Open a shell in the running production backend
prod-shell:
    docker compose -f docker-compose.prod.yml exec backend python manage.py shell

# Back up the production database to ./backups/
prod-backup:
    mkdir -p backups
    # Single quotes on purpose: POSTGRES_USER/DB are expanded by the shell
    # *inside* the db container (the postgres image sets both). This justfile
    # sets `dotenv-load := false`, so on the host they would be empty.
    docker compose -f docker-compose.prod.yml exec -T db \
        sh -c 'pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB"' \
        | gzip > "backups/$(date +%Y%m%d-%H%M%S).sql.gz"
    @ls -lh backups | tail -1

# ── Backend ────────────────────────────────────────────────────────────────────

# Install backend dependencies (uv)
be-install:
    cd backend && uv sync --group dev

# Run the Django dev server locally
be-dev: be-makemigrations be-migrate
    cd backend && uv run python manage.py runserver 0.0.0.0:8006

# Apply database migrations
be-migrate:
    cd backend && uv run python manage.py migrate

# Create new migrations after model changes
be-makemigrations app="":
    cd backend && uv run python manage.py makemigrations {{ app }}

# Show pending migrations
be-showmigrations:
    cd backend && uv run python manage.py showmigrations

# Open the Django shell
be-shell:
    cd backend && uv run python manage.py shell

# Create a Django superuser
be-superuser:
    cd backend && uv run python manage.py createsuperuser --noinput

# Create/reset the user the Playwright E2E suite signs in as (dev only)
be-seed-e2e:
    cd backend && uv run python manage.py seed_e2e_user

# Collect static files
be-collectstatic:
    cd backend && uv run python manage.py collectstatic --noinput

# Run backend test suite
be-test:
    cd backend && uv run pytest

# Run backend tests with coverage
be-test-cov:
    cd backend && uv run pytest --cov=apps --cov-report=term-missing

# Lint backend code (ruff, if available)
be-lint:
    cd backend && uv run ruff check .

# Format backend code (ruff, if available)
be-fmt:
    cd backend && uv run ruff format .

# Create a new Django app: just be-startapp myapp
be-startapp name:
    mkdir -p backend/apps/{{ name }}
    cd backend && uv run python manage.py startapp {{ name }} apps/{{ name }}

# ── Frontend ───────────────────────────────────────────────────────────────────

# Install frontend dependencies
fe-install:
    cd frontend && npm install

# Run the Vite dev server locally
fe-dev:
    cd frontend && npm run dev -- --host 0.0.0.0 --port 5180

# Build the frontend for production
fe-build:
    cd frontend && npm run build

# Check the entry chunk against its size budget (needs a build first)
fe-check-bundle:
    cd frontend && npm run check:bundle

# Regenerate the raster brand assets from public/favicon.svg
fe-icons:
    cd frontend && npm run icons

# Preview the production build
fe-preview:
    cd frontend && npm run preview

# Lint frontend code
fe-lint:
    cd frontend && npm run lint

# Run frontend tests
fe-test:
    cd frontend && npm test

# Run frontend tests in watch mode with Vitest UI
fe-test-ui:
    cd frontend && npm run test:ui

# Install the Playwright browsers (run once after fe-install)
fe-e2e-install:
    cd frontend && npx playwright install --with-deps chromium firefox

# Run the Playwright end-to-end tests (mocked API — no backend needed)
fe-e2e:
    cd frontend && npm run test:e2e

# Run E2E tests against the real Django API (needs `just dev` + `just be-seed-e2e`)
fe-e2e-full: be-seed-e2e
    cd frontend && E2E_BACKEND=1 npm run test:e2e

# Run the @slow E2E specs: real model downloads + real ONNX sessions (minutes)
fe-e2e-slow:
    cd frontend && E2E_SLOW=1 npx playwright test --project=chromium --workers=1 audio-models.spec.ts

# Run the @slow speech-enhancement specs (DeepFilterNet3). The WebGPU half runs
# in the `webgpu` project and skips itself on a machine with no GPU device.
fe-e2e-enhance:
    cd frontend && E2E_SLOW=1 npx playwright test --project=chromium --project=webgpu --workers=1 --grep "speech enhancement"

# Run the @slow /graph specs: a real 200-epoch GNN training run on Cora, pinned
# by an accuracy floor, plus the depth sweep that measures oversmoothing. Nothing
# is downloaded — the dataset is bundled — but on SwiftShader budget ~10 minutes.
fe-e2e-graph:
    cd frontend && E2E_SLOW=1 npx playwright test --project=webgpu --workers=1 webgpu/graph.spec.ts

# Rebuild the bundled Cora dataset from the LINQS release (needs curl + tar).
# Only needed to change the binary format; the dataset itself has not moved.
fe-data-cora:
    cd frontend && node scripts/prepare-cora.mjs

# Run the @slow voice-activity-detection specs (Silero VAD, ~2 MB, seconds)
fe-e2e-vad:
    cd frontend && E2E_SLOW=1 npx playwright test --project=chromium --grep "voice activity detection"

# Run the @slow vision specs: real loads of MobileNetV4, Depth Anything V2,
# D-FINE nano, SegFormer-B0, CLIP, OWLv2, DINOv2, SlimSAM, the D-FINE+ViTPose
# pair, and the Wave 3 carve-outs (MODNet, Swin2SR, depth-to-point-cloud), each
# asserting a known answer or a real measurement on a known input. OWLv2 alone is
# ~155 MB, so budget tens of minutes on a cold cache. The Florence-2 half runs in
# the `webgpu` project and skips itself on a machine with no GPU device.
fe-e2e-vision:
    cd frontend && E2E_SLOW=1 npx playwright test --project=chromium --project=webgpu --workers=1 vision-models.spec.ts

# Run the @slow super-resolution spec on its own: a real Swin2SR load, a real
# tiled upscale, and the measurement that separates a working reconstruction
# from a mis-assembled one — the model's output, downscaled by 2, must differ
# materially from the bicubic baseline rather than matching it.
#
# **This is also the spec that decides `SUPER_RES_MODELS[0].dtypes`.** WASM is
# pinned to fp32 as a precaution (super-resolution is dense regression, where
# int8 error lands straight in the picture). If q8 clears this bar, drop the
# override — it costs 31 MB of download.
fe-e2e-superres:
    cd frontend && E2E_SLOW=1 npx playwright test --project=chromium --workers=1 --grep "super-resolution"

# Run one @slow vision route at a time — the whole file is tens of minutes cold.
#   just fe-e2e-vision-one /mask-generation
fe-e2e-vision-one route:
    cd frontend && E2E_SLOW=1 npx playwright test --project=chromium --project=webgpu --workers=1 vision-models.spec.ts --grep "{{route}}"

# Run the @slow zero-shot parity spec: the split-tower path (which caches the
# label embeddings) scored against the full CLIP graph on the same image. The
# only check that can catch a wrong `logit_scale` — see vision/zeroshot/scoring.ts
# Runs at fp32 on purpose (~1.2 GB, minutes on a cold cache): at q8 the two paths
# are separately quantized exports and the comparison measures the quantizer.
fe-e2e-zeroshot:
    cd frontend && E2E_SLOW=1 npx playwright test --project=chromium --workers=1 zero-shot-parity.spec.ts

# Run the @slow link-prediction spec: a real GCN encoder trained on a Cora with
# 10% of its citations removed, pinned by an AUC **band**. A floor alone would
# pass with the bug this page can actually have — leakage makes the number go up.
fe-e2e-link:
    cd frontend && E2E_SLOW=1 npx playwright test --project=webgpu --workers=1 link-prediction.spec.ts

# Run the @slow graph-classification spec: a real GCN over 1113 protein graphs,
# pinned **above the majority baseline**. PROTEINS is 663/450, so a model that
# ignores the molecule scores 0.598 — "above chance" would pass with it broken.
fe-e2e-graphcls:
    cd frontend && E2E_SLOW=1 npx playwright test --project=webgpu --workers=1 graph-classification.spec.ts

# Run the @slow VLM spec: a real SmolVLM-256M load (~189 MB at q4f16) and a real
# generation, asserting a **known answer on a known image**.
#
# This is the only test that can catch a broken chat template, which is this
# page's characteristic failure — the unit suite mocks the worker away and a
# mocked E2E run never loads weights, so both stay green while the model is
# prompted with a string it has never seen. The symptom is not an error: it is a
# fluent, confident sentence that does not answer the question. "Some text
# appeared" would pass straight through it.
#
# WebGPU only, by catalogue declaration — an autoregressive decoder on WASM is
# seconds per token, so the picker disables the row and there is nothing to run.
fe-e2e-vlm:
    cd frontend && E2E_SLOW=1 npx playwright test --project=webgpu --workers=1 webgpu/vlm.spec.ts

# @slow: a real SmolVLM2-Video load and generation — the only guard on the
# *multi-image* chat template, and the only place the reverse-frames toggle is
# shown to be a real second inference rather than a re-render.
#
# N frames means N `{ type: "image" }` slots filled positionally from the image
# list; one short, or the two out of step, and the model answers fluently about
# the wrong pictures with no error anywhere. Hence a known answer about a known
# clip. Four frames of image tokens through a 256M decoder is the slowest run in
# the app, so this is minutes rather than seconds.
#
# Needs a GPU with shader-f16 (it skips without one), same as fe-e2e-vlm.
fe-e2e-videovlm:
    cd frontend && E2E_SLOW=1 npx playwright test --project=webgpu --workers=1 webgpu/video-vlm.spec.ts

# Run the @slow text spec: a real DistilBERT SST-2 load (~128 MB on WebGPU /
# 68 MB on WASM) and a real classification, asserting a **known label on a known
# sentence** rather than "a ranked list appeared" — the latter is exactly what a
# model with a broken tokenizer also produces.
#
# The second test loads FinBERT alongside it and pins the head-to-head
# structurally: SST-2's head has two classes and FinBERT's has three, so a
# comparison that quietly renders one model's answer twice cannot pass.
fe-e2e-text:
    cd frontend && E2E_SLOW=1 npx playwright test --project=chromium --workers=1 text-models.spec.ts

# Run only the @slow fill-mask specs: two real loads (BERT base, 219 MB on
# WebGPU / 111 MB on WASM, then RoBERTa base at 250 / 126) asking the same
# question through two different tokenizers.
#
# **The RoBERTa half is the test.** Its mask is `<mask>`, not `[MASK]`, so a
# page that hard-codes the literal passes the BERT test and fails this one —
# loudly, because `FillMaskPipeline` looks `mask_token_id` up in the token ids
# and raises "Mask token (<mask>) not found in text.". Both halves assert the
# *word* (paris) rather than "a ranked list appeared".
fe-e2e-fillmask:
    cd frontend && E2E_SLOW=1 npx playwright test --project=chromium --workers=1 text-models.spec.ts -g "fill-mask"

# Run only the @slow question-answering specs: a real DistilBERT-SQuAD load
# (~125 MB on WebGPU / 63 MB on WASM) and real extractive answers.
#
# **The assertion is a character range, not a string**, and that is the whole
# reason the spec is worth its minutes. "A span appeared" passes while the
# token->character alignment is off by one; "the span reads Gustave Eiffel"
# passes while it marks the *second* mention of a name. Only the offsets pin it.
#
# The second test is the one that justifies `text/offsets.ts` existing at all:
# the model's own decode of that answer is `general - purpose compute shaders`,
# which does not occur in the passage, so the obvious shortcut — searching the
# passage for the answer string — highlights nothing. The third asserts that the
# model answers a question its passage cannot answer, which is the page's
# subject rather than a failure.
fe-e2e-qa:
    cd frontend && E2E_SLOW=1 npx playwright test --project=chromium --workers=1 text-models.spec.ts -g "question answering"

# Run the @slow zero-shot spec on its own: a real DeBERTa-v3-xsmall load
# (~136 MB on WebGPU / 83 MB on WASM) and real entailment runs.
#
# Three things only a real model can check, and every one of them produces a
# plausible-looking page when it is broken:
#
#   * a **known ranking on a known sentence** — a billing complaint must put
#     `billing` first against labels the model has never been trained on;
#   * that the **hypothesis template reaches the model** — two different
#     templates producing identical scores is exactly what a page that lets the
#     pipeline apply its own default looks like, and nothing else can catch it;
#   * that **multi-label is a second inference**, not a re-derivation: the
#     single-label scores sum to 1 and the multi-label ones do not.
#
# It is one forward pass per label, so a three-label run is three inferences.
fe-e2e-zeroshot-text:
    cd frontend && E2E_SLOW=1 npx playwright test --project=chromium --workers=1 text-models.spec.ts -g "zero-shot"

# Run only the @slow embedding specs: a real all-MiniLM-L6-v2 load (43 MB on
# WebGPU / 22 MB on WASM — the cheapest floor in the app) and real cosines.
#
# **The assertion is a spread, not a threshold.** Dropping the pooling and
# normalise options — or pooling a CLS-trained checkpoint by the mean — does not
# fail: it produces embeddings whose cosines all sit in a narrow band near 0.9,
# so every pair looks alike and the page looks like it works. "The paraphrase
# scores above 0.5" passes comfortably on exactly those collapsed vectors; a
# *gap* between the paraphrase and the unrelated pair does not.
#
# It also pins the truncation control, which has two halves that fail silently:
# the ranking must survive the cut, and both vectors must still read
# 1.000 afterwards — a missing renormalisation leaves a working-looking slider
# that scales every similarity by an arbitrary factor.
fe-e2e-embed:
    cd frontend && E2E_SLOW=1 npx playwright test --project=chromium --workers=1 text-models.spec.ts -g "embedding"

# Check every model id (audio + vision + multimodal + text) still resolves on
# the Hugging Face Hub, that each vision entry publishes the dtypes both backends
# ask for, that the VLM entries publish their three q4f16 graphs, and that the
# VLM and text catalogues still match the download sizes they quote, and that
# each embedding entry's pooling matches its upstream training config (seconds)
fe-e2e-models:
    cd frontend && E2E_SLOW=1 npx playwright test --project=chromium model-ids.spec.ts

# Run E2E tests in the interactive Playwright UI
fe-e2e-ui:
    cd frontend && npm run test:e2e:ui

# Run E2E tests in a headed browser
fe-e2e-headed:
    cd frontend && npm run test:e2e:headed

# Run only the WebGPU E2E project: real WGSL, each kernel cross-checked against a
# CPU reference. On a machine with no GPU device node Chromium falls back to
# SwiftShader — slow, but numerically real — so these run on a CI runner too.
fe-e2e-webgpu:
    cd frontend && npx playwright test --project=webgpu

# Open the last Playwright HTML report
fe-e2e-report:
    cd frontend && npm run test:e2e:report

# ── Dev ────────────────────────────────────────────────────────────────────────

# Install all dependencies (backend + frontend)
install: be-install fe-install

# Start the db container if not already running
db-up:
    @docker compose ps --status running db | grep -q db \
        && echo "DB already running." \
        || (echo "Starting DB..." && docker compose up -d db && echo "Waiting for DB to be ready..." && sleep 3)

# Start the Redis + Celery worker + beat containers if not already running
celery-up:
    #!/usr/bin/env bash
    _redis_up=$(docker compose ps --status running redis | grep -c redis || true)
    _worker_up=$(docker compose ps --status running celery_worker | grep -c celery_worker || true)
    if [[ "$_redis_up" -gt 0 && "$_worker_up" -gt 0 ]]; then
        echo "Redis + Celery worker already running."
    else
        echo "Starting Redis + Celery worker + beat..."
        docker compose up -d redis celery_worker celery_beat
    fi

# Run the Celery worker locally (outside Docker) — useful for debugging tasks
celery-worker:
    cd backend && uv run celery -A core worker --loglevel=info --concurrency=2

# Run Flower monitoring UI locally (port 5555)
flower:
    cd backend && uv run celery -A core flower --port=5555

# Tail Celery worker logs
celery-logs:
    docker compose logs -f celery_worker

# Run backend and frontend dev servers concurrently (uses overmind if available)
dev: db-up celery-up
    #!/usr/bin/env bash
    echo "Starting backend and frontend dev servers..."
    echo "  Backend : http://localhost:8006"
    echo "  Frontend: http://localhost:5180"
    _pids=()
    _cleanup() {
        echo ""
        for pid in "${_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
        wait 2>/dev/null || true
    }
    trap _cleanup INT TERM EXIT
    if command -v overmind &>/dev/null; then
        overmind start
    else
        just be-dev &
        _pids+=($!)
        just fe-dev &
        _pids+=($!)
        wait "${_pids[@]}"
    fi

# ── Database ───────────────────────────────────────────────────────────────────

# Open a psql session via Docker
db-shell:
    docker compose exec db psql -U appuser -d appdb

# Reset the database (drops volumes and re-applies migrations)
db-reset: down-v up-d be-migrate
    @echo "Database reset complete."

# ── Utilities ──────────────────────────────────────────────────────────────────

# Print configured environment files
env:
    @echo "=== backend/.env ===" && cat backend/.env 2>/dev/null || echo "(not found)"
    @echo "=== frontend/.env ===" && cat frontend/.env 2>/dev/null || echo "(not found)"

# Copy .env.example files to .env (safe — skips if already exists)
env-init:
    @[ -f backend/.env ] || cp backend/.env.example backend/.env && echo "Created backend/.env"
    @[ -f frontend/.env ] || cp frontend/.env.example frontend/.env && echo "Created frontend/.env"

# Clean Python bytecode and cache files
clean:
    find backend -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find backend -name "*.pyc" -delete 2>/dev/null || true

# Clean everything including node_modules and build artefacts
clean-all: clean
    rm -rf frontend/node_modules frontend/dist
