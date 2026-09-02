"""Create (or reset) the user the Playwright E2E suite signs in as.

Guarded to refuse to run when DEBUG is off, so it can never mint a known-password
account on a production database. See docs/guides/e2e-testing.md.
"""

import os

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError

DEFAULT_EMAIL = "e2e@example.com"
DEFAULT_PASSWORD = "e2e-password-123"


class Command(BaseCommand):
    help = "Create or reset the end-to-end test user (development only)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--email",
            default=os.environ.get("E2E_USER_EMAIL", DEFAULT_EMAIL),
            help="Email address for the test user.",
        )
        parser.add_argument(
            "--password",
            default=os.environ.get("E2E_USER_PASSWORD", DEFAULT_PASSWORD),
            help="Password for the test user.",
        )

    def handle(self, *args, **options):
        if not settings.DEBUG:
            raise CommandError(
                "seed_e2e_user refuses to run with DEBUG=False — it creates an "
                "account with a well-known password and is for local/CI use only."
            )

        email = options["email"]
        password = options["password"]

        user_model = get_user_model()
        user, created = user_model.objects.get_or_create(
            email=email,
            defaults={"first_name": "Eee", "last_name": "Tootoo"},
        )
        user.set_password(password)
        user.is_active = True
        user.save()

        verb = "Created" if created else "Reset password for"
        self.stdout.write(self.style.SUCCESS(f"{verb} E2E user {email}"))
