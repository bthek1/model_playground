from .base import *

DEBUG = False
ALLOWED_HOSTS = ["localhost", "127.0.0.1"]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

EMAIL_BACKEND = "django.core.mail.backends.locmem.EmailBackend"
DEFAULT_FROM_EMAIL = "root@localhost"

PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]

# Tests never run `collectstatic`, so neither half of the WhiteNoise setup in
# base.py applies here: the hashed-manifest storage raises on any {% static %}
# lookup until staticfiles.json exists, and the middleware warns once per test
# that STATIC_ROOT is missing. Serving static files is not what these tests are
# for — that behaviour is checked against a built image instead.
STORAGES = {
    **STORAGES,
    "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
}

MIDDLEWARE = [m for m in MIDDLEWARE if "whitenoise" not in m]

# Run Celery tasks synchronously in tests — no broker required.
CELERY_TASK_ALWAYS_EAGER = True
CELERY_TASK_EAGER_PROPAGATES = True
