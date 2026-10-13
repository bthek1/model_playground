#!/usr/bin/env bash
# Tell IndexNow search engines which pages changed in this deploy (#69).
#
#   scripts/indexnow-submit.sh [url] [dist-dir]
#
# url defaults to https://playground.benedictthekkel.com; dist-dir (default
# frontend/dist) is the artifact just deployed. CI runs it after the smoke
# test. IndexNow is Bing, Yandex, Seznam, Naver and others — not Google.
#
#   1. The key file (`<32 hex>.txt` at the site root) must be live and hold
#      its own name: that is how an engine knows the submitter owns the host.
#   2. The *previous* seo-manifest.json comes from the live site — the upload
#      script never writes it, this script does — and is diffed against the
#      new one. The build's digests cover what a crawler reads, not the hashed
#      chunk names every deploy changes, so an unchanged page is not resent.
#      No previous manifest means a first submission: every page goes.
#   3. New, changed and removed pages are POSTed in one request. A removed
#      page goes too, so an engine re-fetches it and sees its 404.
#   4. Only on 200/202 is the new manifest uploaded. A failed submission
#      leaves the old one live, so the next deploy retries the same diff.
#
# It also keeps the sitemap's <lastmod> honest (#72). The recorded manifest is
# `{ path: { digest, modified } }`: a page whose digest did not move keeps the
# date it had, a new or changed one gets today's (UTC), and a manifest from
# before this (`{ path: digest }`) seeds today for every page, once. The built
# sitemap — which has no dates, because the build cannot know when a page last
# changed — is re-uploaded with a <lastmod> per URL, on every deploy that gets
# that far, whether or not anything is submitted. If the step cannot run, the
# deploy's own sitemap stays live without dates, which is what it always was;
# a wrong date is worse than none, because Google stops trusting them.
#
# It NEVER fails the deploy: by now the site is live, and a notification is an
# optimisation — crawlers still have the sitemap. Every problem is a warning in
# the job log, naming what went wrong, and the script exits 0.
set -uo pipefail

URL="${1:-https://playground.benedictthekkel.com}"
DIST="${2:-frontend/dist}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${URL#https://}"
ENDPOINT="${INDEXNOW_ENDPOINT:-https://api.indexnow.org/IndexNow}"

warn() {
  echo "⚠ indexnow: $*" >&2
  [[ -n "${GITHUB_ACTIONS:-}" ]] && echo "::warning title=IndexNow::$*"
  exit 0
}
ok() { echo "✓ indexnow: $*"; }

# 1. The key, from the artifact, and the same key live.
mapfile -t keys < <(cd "$DIST" 2>/dev/null && ls | grep -E '^[0-9a-f]{32}\.txt$')
[[ ${#keys[@]} -eq 1 ]] || warn "want exactly one <32 hex>.txt key file in $DIST, found ${#keys[@]}"
KEY="${keys[0]%.txt}"
KEY_URL="$URL/$KEY.txt"
live_key="$(curl -sS --max-time 20 "$KEY_URL" 2>/dev/null)"
[[ "${live_key//[[:space:]]/}" == "$KEY" ]] ||
  warn "$KEY_URL does not serve the key — engines would reject the submission; skipped"

note() {
  echo "⚠ indexnow: $*" >&2
  [[ -n "${GITHUB_ACTIONS:-}" ]] && echo "::warning title=IndexNow::$*"
  return 0
}

# 2. Previous manifest (live) against the new one (artifact).
NEXT="$DIST/seo-manifest.json"
[[ -f "$NEXT" ]] || warn "no $NEXT in the artifact; skipped"
PREV="$(mktemp)"
MERGED="$(mktemp)"
SITEMAP="$(mktemp)"
trap 'rm -f "$PREV" "$MERGED" "$SITEMAP"' EXIT
TODAY="${SEO_TODAY:-$(date -u +%F)}"
code="$(curl -sS --max-time 20 -o "$PREV" -w '%{http_code}' "$URL/seo-manifest.json" 2>/dev/null)"
case "$code" in
  200) ;;
  403 | 404) echo "{}" >"$PREV"; echo "  no previous manifest ($code): first submission, every page" ;;
  *) warn "fetching the previous manifest answered '${code:-nothing}'; skipped rather than resubmit every page" ;;
esac

# The diff, the dated manifest to record, and the sitemap with <lastmod>.
urls="$(PREV="$PREV" NEXT="$NEXT" ORIGIN="$URL" TODAY="$TODAY" MERGED="$MERGED" \
  SITEMAP_IN="$DIST/sitemap.xml" SITEMAP_OUT="$SITEMAP" node -e '
  const fs = require("fs");
  const env = process.env;
  let prev;
  try { prev = JSON.parse(fs.readFileSync(env.PREV, "utf8")); } catch { prev = {}; }
  const next = JSON.parse(fs.readFileSync(env.NEXT, "utf8"));
  // A recorded entry is { digest, modified }; one from before #72 is the digest.
  const digest = (e) => (typeof e === "string" ? e : e && e.digest);
  const dated = (e) => (e && typeof e === "object" && /^\d{4}-\d{2}-\d{2}$/.test(e.modified) ? e.modified : null);
  const merged = {};
  for (const [p, d] of Object.entries(next)) {
    merged[p] = { digest: d, modified: digest(prev[p]) === d ? dated(prev[p]) || env.TODAY : env.TODAY };
  }
  fs.writeFileSync(env.MERGED, JSON.stringify(merged, null, 2) + "\n");
  if (fs.existsSync(env.SITEMAP_IN)) {
    const xml = fs.readFileSync(env.SITEMAP_IN, "utf8").replace(
      /<url><loc>([^<]*)<\/loc><\/url>/g,
      (whole, loc) => {
        let path;
        try { path = new URL(loc.replace(/&amp;/g, "&")).pathname; } catch { return whole; }
        const m = merged[path];
        return m ? `<url><loc>${loc}</loc><lastmod>${m.modified}</lastmod></url>` : whole;
      },
    );
    fs.writeFileSync(env.SITEMAP_OUT, xml);
  }
  const paths = new Set([
    ...Object.keys(next).filter((p) => digest(prev[p]) !== next[p]),   // new or changed
    ...Object.keys(prev).filter((p) => !(p in next)),                  // removed
  ]);
  for (const p of paths) if (/^\/[a-z0-9\/-]*$/.test(p)) console.log(env.ORIGIN + p);
')" || warn "could not diff the manifests; skipped"

BUCKET="${BUCKET:-$(pulumi -C "$ROOT/infra/site" stack output --stack production bucketName 2>/dev/null)}"

# The dated sitemap goes up whatever happens next: its dates are true now.
if [[ -s "$SITEMAP" && -n "$BUCKET" ]]; then
  if aws s3 cp "$SITEMAP" "s3://$BUCKET/sitemap.xml" --only-show-errors \
    --content-type "application/xml; charset=utf-8" \
    --cache-control "public, max-age=0, s-maxage=300, must-revalidate"; then
    ok "sitemap dated"
  else
    note "uploading the dated sitemap failed; the deploy's undated one stays live"
  fi
fi

record() {
  [[ -n "$BUCKET" ]] || warn "$1, but no bucket to record the manifest in; the next deploy resubmits"
  aws s3 cp "$MERGED" "s3://$BUCKET/seo-manifest.json" --only-show-errors \
    --content-type "application/json; charset=utf-8" \
    --cache-control "no-cache, must-revalidate" ||
    warn "$1, but recording the manifest failed; the next deploy resubmits"
  ok "manifest recorded"
}

if [[ -z "$urls" ]]; then
  ok "no page changed what a crawler reads; nothing to submit"
  # Nothing to tell the engines, but a manifest from before #72 has no dates
  # yet: record the seeded ones, or every deploy would re-seed them as today.
  if ! cmp -s <(node -e 'console.log(JSON.stringify(JSON.parse(require("fs").readFileSync(process.argv[1], "utf8"))))' "$PREV" 2>/dev/null) \
             <(node -e 'console.log(JSON.stringify(JSON.parse(require("fs").readFileSync(process.argv[1], "utf8"))))' "$MERGED"); then
    record "dates seeded"
  fi
  exit 0
fi
count="$(grep -c . <<<"$urls")"

# 3. One request for the lot.
payload="$(HOST="$HOST" KEY="$KEY" KEY_URL="$KEY_URL" URLS="$urls" node -e '
  const urlList = process.env.URLS.split("\n").filter(Boolean);
  console.log(JSON.stringify({
    host: process.env.HOST, key: process.env.KEY,
    keyLocation: process.env.KEY_URL, urlList,
  }));
')"
status="$(curl -sS --max-time 30 -o /dev/null -w '%{http_code}' -X POST \
  -H "Content-Type: application/json; charset=utf-8" \
  --data "$payload" "$ENDPOINT" 2>/dev/null)"
case "$status" in
  200 | 202) ok "submitted $count URL(s) ($status)" ;;
  *) warn "IndexNow answered '${status:-nothing}' for $count URL(s); manifest not advanced, the next deploy retries" ;;
esac

# 4. Advance the manifest, now the engines have the diff.
record "submitted"
