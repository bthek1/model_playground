# Production image for the Django API.
#
# Two stages: `uv` installs into /venv in the builder, and the runtime is a plain
# python:slim with no uv and no build cache. The venv lives at **/venv, outside
# /app on purpose** — the development compose file bind-mounts ./backend over
# /app, and a venv inside the project directory would be shadowed by that mount.
#
# The same image serves development: docker-compose.yml overrides the command
# with `runserver`, and the entrypoint below only migrates when asked to.

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/venv

WORKDIR /app

# Dependencies first, so a source change doesn't reinstall the world.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/venv/bin:$PATH" \
    DJANGO_SETTINGS_MODULE=core.settings.prod

RUN useradd --create-home --uid 1000 app

COPY --from=builder /venv /venv
COPY --from=builder --chown=app:app /app /app

WORKDIR /app

# Collect admin + DRF static assets into STATIC_ROOT so WhiteNoise can serve
# them. Done at build time, not on boot: it is the same for every replica, and
# `SECRET_KEY` is not available yet — hence the throwaway one. No other setting
# is read, because collectstatic only walks the apps' static directories.
RUN SECRET_KEY=build-only ALLOWED_HOSTS=localhost CSRF_TRUSTED_ORIGINS=https://localhost \
    DATABASE_URL=sqlite:///tmp/build.db \
    python manage.py collectstatic --noinput --clear \
    && chown -R app:app /app/staticfiles

USER app

EXPOSE 8000

ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Threads, not more workers: this process serves a metadata API and does no
# inference, so it is I/O bound on Postgres.
CMD ["gunicorn", "core.wsgi:application", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "3", \
     "--threads", "4", \
     "--timeout", "60", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
