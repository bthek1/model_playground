#!/usr/bin/env bash
# Post-deploy smoke test for the static frontend (#57). Fails on the first
# broken property; CI runs it after scripts/deploy-frontend.sh.
#
#   scripts/smoke-frontend.sh [url] [dist-dir]
#
# url defaults to https://playground.benedictthekkel.com; dist-dir (default
# frontend/dist) names a real hashed .js and .wasm to probe. $BUCKET (or the
# `site` stack output) is the bucket whose direct URL must refuse the request.
set -euo pipefail

URL="${1:-https://playground.benedictthekkel.com}"
DIST="${2:-frontend/dist}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${URL#https://}"
BUCKET="${BUCKET:-$(pulumi -C "$ROOT/infra/site" stack output --stack production bucketName)}"
REGION="${AWS_REGION:-ap-southeast-2}"

fail() { echo "✗ $*" >&2; exit 1; }
ok() { echo "✓ $*"; }
# Headers only, lower-cased names; no -k anywhere, so every https request here
# also verifies the certificate chain and that it names $HOST.
head_of() { curl -sS -o /dev/null -D - "$@" | tr -d '\r' | tr 'A-Z' 'a-z'; }
header() { grep -m1 "^$1:" | cut -d' ' -f2-; }

# A fresh distribution or an in-flight invalidation can take minutes to settle.
ATTEMPTS="${SMOKE_ATTEMPTS:-30}"
for i in $(seq 1 "$ATTEMPTS"); do
  status="$(curl -sS -o /dev/null -w '%{http_code}' "$URL/" || true)"
  [[ "$status" == "200" ]] && break
  echo "  / answered ${status:-nothing}; waiting ($i/$ATTEMPTS)"
  sleep "${SMOKE_WAIT:-10}"
done
[[ "$status" == "200" ]] || fail "/ is $status, not 200"
ok "/ is 200 over a certificate valid for $HOST"

cache="$(head_of "$URL/" | header cache-control)"
[[ "$cache" == *no-cache* ]] || fail "/ cache-control is '$cache', want no-cache"
ok "/ is no-cache"

js="$(cd "$DIST" && ls assets/index-*.js | head -1)"
cache="$(head_of "$URL/$js" | header cache-control)"
[[ "$cache" == *immutable* ]] || fail "/$js cache-control is '$cache', want immutable"
ok "/$js is immutable"

wasm="$(cd "$DIST" && ls assets/*.wasm | head -1)"
ctype="$(head_of "$URL/$wasm" | header content-type)"
[[ "$ctype" == application/wasm* ]] || fail "/$wasm content-type is '$ctype'"
ok "/$wasm is application/wasm"

deep="$(curl -sS "$URL/asr")"
[[ "$deep" == *'<div id="root">'* ]] || fail "/asr did not return the app shell"
ok "/asr (a deep link) returns the app shell"

code="$(curl -sS -o /dev/null -w '%{http_code} %{redirect_url}' "http://$HOST/asr")"
[[ "$code" == 301\ https://$HOST/asr || "$code" == 308\ https://$HOST/asr ]] ||
  fail "http:// answered '$code', want a redirect to https"
ok "http:// redirects to https://"

code="$(curl -sS -o /dev/null -w '%{http_code}' \
  "https://$BUCKET.s3.$REGION.amazonaws.com/index.html")"
[[ "$code" == "403" ]] || fail "the bucket answered $code directly, want 403"
ok "the bucket refuses direct requests (403)"
