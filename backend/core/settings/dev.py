from .base import *

DEBUG = True
ALLOWED_HOSTS = env.list("ALLOWED_HOSTS", default=["localhost", "127.0.0.1"])

DATABASES = {
    "default": env.db(
        "DATABASE_URL", default="postgres://appuser:apppassword@localhost:5432/appdb"
    ),
}

EMAIL_BACKEND = "django.core.mail.backends.console.EmailBackend"
DEFAULT_FROM_EMAIL = env("DEFAULT_FROM_EMAIL", default="noreply@example.com")

CORS_ALLOWED_ORIGINS = env.list(
    "CORS_ALLOWED_ORIGINS", default=["http://localhost:5174"]
)

# Allow LAN origins over both http and https — the Vite dev server runs over
# HTTPS (WebGPU needs a secure context), so browsers on the LAN hit the API from
# an https://192.168.x.x origin.
CORS_ALLOWED_ORIGIN_REGEXES = [
    r"^https?://192\.168\.\d+\.\d+:\d+$",
]

# JWT auth uses the Authorization header (not cookies), but keep CSRF-trusted
# origins in sync in case any session/CSRF-protected view is added later.
CSRF_TRUSTED_ORIGINS = [
    *env.list(
        "CSRF_TRUSTED_ORIGINS",
        default=["http://localhost:5174", "https://localhost:5174"],
    ),
    # Regex isn't supported here; the wildcard covers the LAN /24 blocks.
    "https://192.168.*.*:*",
    "http://192.168.*.*:*",
]
