#!/usr/bin/env bash
# Upload a built frontend to the site bucket and make it live (#57).
#
#   scripts/deploy-frontend.sh [dist-dir]      # default: frontend/dist
#
# Bucket and distribution come from $BUCKET / $DISTRIBUTION_ID, or from the
# `site` stack's outputs (`pulumi -C infra/site stack output --stack production ...`) when unset.
# Run it after `pulumi up`, never instead of it.
#
# The ORDER is the point. A browser tab holding the previous index.html will
# still ask for the previous build's hashed chunks, possibly days later, so:
#
#   1. hashed /assets first (immutable) — the new build's chunks exist before
#      anything references them;
#   2. .wasm with an explicit application/wasm — ONNX Runtime's streaming
#      compilation refuses anything else, and the CLI's MIME guess is not a
#      contract;
#   3. index.html and the manifest LAST (no-cache) — the switch-over;
#   4. never `--delete`. An asset this build no longer references is re-written
#      in place tagged `stale=true`, once, and the bucket's lifecycle rule
#      expires it N days later (infra/site: staleAssetDays). Re-writing resets
#      its age, so the clock starts when it went stale, not when it was built;
#   5. invalidate only /index.html and the manifest: hashed names never change
#      meaning, so they never need invalidating.
set -euo pipefail

DIST="${1:-frontend/dist}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

[[ -f "$DIST/index.html" ]] || {
  echo "no $DIST/index.html — build first (VITE_BACKEND=off npm run build)" >&2
  exit 1
}
[[ -d "$DIST/assets" ]] || { echo "no $DIST/assets" >&2; exit 1; }

output() { pulumi -C "$ROOT/infra/site" stack output --stack production "$1"; }
BUCKET="${BUCKET:-$(output bucketName)}"
DISTRIBUTION_ID="${DISTRIBUTION_ID:-$(output distributionId)}"

IMMUTABLE="public, max-age=31536000, immutable"
REVALIDATE="no-cache, must-revalidate"
DAY="public, max-age=86400"

echo "→ s3://$BUCKET (distribution $DISTRIBUTION_ID)"

# 1. Hashed assets (everything under assets/ except .wasm).
aws s3 cp "$DIST/assets" "s3://$BUCKET/assets" --recursive --only-show-errors \
  --exclude "*.wasm" \
  --cache-control "$IMMUTABLE"

# 2. WebAssembly, with its content type stated rather than guessed.
aws s3 cp "$DIST/assets" "s3://$BUCKET/assets" --recursive --only-show-errors \
  --exclude "*" --include "*.wasm" \
  --content-type "application/wasm" \
  --cache-control "$IMMUTABLE"

# Un-hashed files from public/ (icons, og-image): stable names, so a day.
aws s3 cp "$DIST" "s3://$BUCKET" --recursive --only-show-errors \
  --exclude "assets/*" --exclude "index.html" --exclude "site.webmanifest" \
  --cache-control "$DAY"

# 3. The switch-over.
if [[ -f "$DIST/site.webmanifest" ]]; then
  aws s3 cp "$DIST/site.webmanifest" "s3://$BUCKET/site.webmanifest" --only-show-errors \
    --content-type "application/manifest+json" \
    --cache-control "$REVALIDATE"
fi
aws s3 cp "$DIST/index.html" "s3://$BUCKET/index.html" --only-show-errors \
  --content-type "text/html; charset=utf-8" \
  --cache-control "$REVALIDATE"

# 4. Mark what this build no longer references. Never delete.
declare -A current=()
while IFS= read -r -d '' f; do
  current["assets/${f#"$DIST/assets/"}"]=1
done < <(find "$DIST/assets" -type f -print0)

marked=0
while IFS= read -r key; do
  [[ "$key" == "None" || -n "${current[$key]:-}" ]] && continue
  stale="$(aws s3api get-object-tagging --bucket "$BUCKET" --key "$key" \
    --query "TagSet[?Key=='stale'].Value" --output text </dev/null)"
  # Already stale: leave it, or its expiry clock restarts on every deploy.
  [[ "$stale" == "true" ]] && continue
  IFS=$'\t' read -r ctype cache < <(aws s3api head-object --bucket "$BUCKET" \
    --key "$key" --query '[ContentType,CacheControl]' --output text)
  aws s3api copy-object --bucket "$BUCKET" --key "$key" \
    --copy-source "$BUCKET/$key" \
    --metadata-directive REPLACE \
    --content-type "$ctype" --cache-control "$cache" \
    --tagging-directive REPLACE --tagging "stale=true" </dev/null >/dev/null
  marked=$((marked + 1))
done < <(aws s3api list-objects-v2 --bucket "$BUCKET" --prefix assets/ \
  --query 'Contents[].Key' --output text | tr '\t' '\n')
echo "  marked $marked superseded asset(s) stale"

# 5. Only the un-hashed switch-over files need the edge to forget them.
aws cloudfront create-invalidation --distribution-id "$DISTRIBUTION_ID" \
  --paths "/index.html" "/site.webmanifest" \
  --query 'Invalidation.Id' --output text
echo "✓ deployed"
