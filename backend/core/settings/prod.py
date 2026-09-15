"""Production settings.

The deployment this is written for serves the SPA and the API from **one
origin** behind a TLS-terminating reverse proxy (see docs/guides/deployment.md).
That is not a preference: `src/api/client.ts` ships an empty base URL so the
browser only ever talks to the origin that served the page, and `navigator.gpu`
is undefined outside a secure context — an HTTP deploy makes every model page
report "WebGPU unsupported" on hardware that supports it perfectly well.
"""

import environ

from .base import *

env = environ.Env()
environ.Env.read_env(BASE_DIR / ".env")

DEBUG = False

SECRET_KEY = env("SECRET_KEY")

ALLOWED_HOSTS = env.list("ALLOWED_HOSTS")

DATABASES = {
    "default": env.db("DATABASE_URL"),
}

EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"
EMAIL_HOST = env("EMAIL_HOST", default="localhost")
EMAIL_PORT = env.int("EMAIL_PORT", default=25)
EMAIL_HOST_USER = env("EMAIL_HOST_USER", default="")
EMAIL_HOST_PASSWORD = env("EMAIL_HOST_PASSWORD", default="")
EMAIL_USE_TLS = env.bool("EMAIL_USE_TLS", default=False)
DEFAULT_FROM_EMAIL = env("DEFAULT_FROM_EMAIL", default="noreply@example.com")

# ── Origins ────────────────────────────────────────────────────────────────────
# Same-origin is the intended deployment, and in that shape CORS is not involved
# at all — so this defaults to empty rather than to something permissive. Set it
# only if you split the SPA onto its own host, and read the note in the
# deployment guide about what that costs first.
CORS_ALLOWED_ORIGINS = env.list("CORS_ALLOWED_ORIGINS", default=[])

# Required behind a TLS-terminating proxy. Django compares the Origin header of
# an unsafe request against this list, and a proxied request arrives claiming
# http:// unless SECURE_PROXY_SSL_HEADER (below) is honoured — so /admin/ logins
# fail with "CSRF verification failed" without it. Must include the scheme.
CSRF_TRUSTED_ORIGINS = env.list("CSRF_TRUSTED_ORIGINS")

# ── HTTPS / security hardening ────────────────────────────────────────────────
# The proxy terminates TLS and forwards over plain HTTP, so Django must be told
# how to recognise a request that *arrived* encrypted. Without this,
# SECURE_SSL_REDIRECT sees http://, redirects, and the proxy forwards the
# redirected request right back — an infinite loop.
#
# It trusts a header the client cannot set only because the proxy overwrites it.
# Exposing this container directly to the internet would make the header
# forgeable and the redirect bypassable.
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
USE_X_FORWARDED_HOST = True

SECURE_SSL_REDIRECT = env.bool("SECURE_SSL_REDIRECT", default=True)
SECURE_HSTS_SECONDS = env.int("SECURE_HSTS_SECONDS", default=31536000)
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = "DENY"

# ── Logging ───────────────────────────────────────────────────────────────────
# Containers log to stdout; the platform collects it. Without this Django's
# default only mails admins on 500s, so a crash leaves no trace in `docker logs`.
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "{asctime} {levelname} {name} {message}",
            "style": "{",
        },
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "verbose"},
    },
    "root": {"handlers": ["console"], "level": env("LOG_LEVEL", default="INFO")},
    "loggers": {
        "django.request": {
            "handlers": ["console"],
            "level": "WARNING",
            "propagate": False,
        },
    },
}
