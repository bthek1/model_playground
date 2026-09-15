#!/bin/sh
# Container entrypoint for the Django API.
#
# Migrations are opt-in via RUN_MIGRATIONS=1 rather than automatic, for two
# reasons: the development compose file uses this same image and should not
# touch the database on every `up`, and with more than one replica an automatic
# migrate races itself. The production compose file sets it on the web service
# only.
set -e

if [ "$RUN_MIGRATIONS" = "1" ]; then
  echo "==> Applying migrations"
  python manage.py migrate --noinput
fi

exec "$@"
