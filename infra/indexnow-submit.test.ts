// scripts/indexnow-submit.sh (#69) against stub `curl` and `aws` on PATH. The
// live site is a directory the stub serves from; IndexNow is a status code the
// stub answers with and a payload it records. The contract: submit only what
// changed, never off-host, advance the manifest only on success, and never
// fail the deploy.

import { spawnSync } from "node:child_process";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { beforeEach, describe, expect, it } from "vitest";

const SCRIPT = join(__dirname, "../scripts/indexnow-submit.sh");
const URL = "https://playground.example.com";
const KEY = "0123456789abcdef0123456789abcdef";

// GET: serve $STUB_LIVE/<path>, 404 when absent (403 if STUB_MISSING_CODE).
// POST: record the body, answer $STUB_POST_CODE.
const CURL = `#!/usr/bin/env bash
url=""; out=""; fmt=""; data=""; post=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o) out="$2"; shift 2 ;;
    -w) fmt="$2"; shift 2 ;;
    --data) data="$2"; shift 2 ;;
    -X) [[ "$2" == POST ]] && post=1; shift 2 ;;
    -H|--max-time) shift 2 ;;
    http*) url="$1"; shift ;;
    *) shift ;;
  esac
done
if [[ $post == 1 ]]; then
  printf '%s' "$data" > "$STUB_POST"
  code="\${STUB_POST_CODE:-200}"
else
  [[ -n "\${STUB_DOWN:-}" ]] && { printf '000'; exit 7; }
  [[ -n "\${STUB_MANIFEST_DOWN:-}" && "$url" == */seo-manifest.json ]] && { [[ -n "$fmt" ]] && printf '000'; exit 28; }
  file="$STUB_LIVE/\${url#${URL}/}"
  if [[ -f "$file" ]]; then code=200; body="$(cat "$file")"; else code="\${STUB_MISSING_CODE:-404}"; body="<Error/>"; fi
fi
if [[ -n "$out" ]]; then printf '%s' "\${body:-}" > "$out"; [[ "$out" == /dev/null ]] || true; else printf '%s' "\${body:-}"; fi
[[ -n "$fmt" ]] && printf '%s' "$code"
exit 0
`;
// Logs each call, and keeps a copy of what an \`s3 cp\` uploaded, by its key.
const AWS = `#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$STUB_AWS"
[[ -n "\${STUB_AWS_FAIL:-}" ]] && exit 1
[[ "$1 $2" == "s3 cp" ]] && cp "$3" "$STUB_UPLOADS/\${4##*/}"
exit 0
`;

let dir: string, dist: string, live: string, bin: string, uploads: string;

const manifest = (m: Record<string, string>) => `${JSON.stringify(m)}\n`;
const NEXT = { "/home": "h1", "/asr": "a1", "/vad": "v1" };
const TODAY = "2026-10-20";
/** The manifest as the script records it (#72): digest and last-changed date. */
const dated = (m: Record<string, string>, date: string) =>
  Object.fromEntries(Object.entries(m).map(([p, digest]) => [p, { digest, modified: date }]));
const uploaded = (name: string) => readFileSync(join(uploads, name), "utf8");
const manifestUploads = (aws: string[]) => aws.filter((l) => l.includes("s3://site-bucket/seo-manifest.json"));

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "indexnow-"));
  dist = join(dir, "dist");
  live = join(dir, "live");
  bin = join(dir, "bin");
  uploads = join(dir, "uploads");
  for (const d of [dist, live, bin, uploads]) mkdirSync(d);
  writeFileSync(join(bin, "curl"), CURL);
  writeFileSync(join(bin, "aws"), AWS);
  chmodSync(join(bin, "curl"), 0o755);
  chmodSync(join(bin, "aws"), 0o755);
  writeFileSync(join(dist, `${KEY}.txt`), KEY);
  writeFileSync(join(live, `${KEY}.txt`), KEY);
  writeFileSync(join(dist, "seo-manifest.json"), manifest(NEXT));
});

function submit(env: Record<string, string> = {}) {
  const r = spawnSync("bash", [SCRIPT, URL, dist], {
    env: {
      ...process.env,
      PATH: `${bin}:${process.env.PATH}`,
      STUB_LIVE: live,
      STUB_POST: join(dir, "post.json"),
      STUB_AWS: join(dir, "aws.log"),
      STUB_UPLOADS: uploads,
      BUCKET: "site-bucket",
      GITHUB_ACTIONS: "",
      SEO_TODAY: TODAY,
      ...env,
    },
    encoding: "utf8",
  });
  const postFile = join(dir, "post.json");
  const awsFile = join(dir, "aws.log");
  return {
    code: r.status,
    out: r.stdout + r.stderr,
    post: existsSync(postFile) ? JSON.parse(readFileSync(postFile, "utf8")) : null,
    aws: existsSync(awsFile) ? readFileSync(awsFile, "utf8").trim().split("\n") : [],
  };
}

describe("indexnow-submit.sh", () => {
  it("submits every page on the first run, and records the manifest", () => {
    const { code, out, post, aws } = submit();
    expect(code).toBe(0);
    expect(post).toEqual({
      host: "playground.example.com",
      key: KEY,
      keyLocation: `${URL}/${KEY}.txt`,
      urlList: [`${URL}/home`, `${URL}/asr`, `${URL}/vad`],
    });
    expect(out).toContain("first submission");
    expect(out).toContain("✓ indexnow: submitted 3 URL(s) (200)");
    expect(aws).toHaveLength(1);
    expect(aws[0]).toContain("s3://site-bucket/seo-manifest.json");
    expect(aws[0]).toContain("no-cache, must-revalidate");
    expect(aws[0]).toContain("application/json");
    expect(JSON.parse(uploaded("seo-manifest.json"))).toEqual(dated(NEXT, TODAY));
  });

  it("treats a 403 for the manifest (S3, no object) as a first run too", () => {
    expect(submit({ STUB_MISSING_CODE: "403" }).post.urlList).toHaveLength(3);
  });

  it("submits nothing, and writes nothing, when no page changed", () => {
    writeFileSync(join(live, "seo-manifest.json"), `${JSON.stringify(dated(NEXT, "2026-10-01"))}\n`);
    const { code, out, post, aws } = submit();
    expect(code).toBe(0);
    expect(post).toBeNull();
    expect(aws).toEqual([]);
    expect(out).toContain("nothing to submit");
  });

  it("submits only the changed, new and removed pages", () => {
    writeFileSync(
      join(live, "seo-manifest.json"),
      manifest({ "/home": "h1", "/asr": "OLD", "/gone": "g1" }),
    );
    const { post } = submit();
    expect(new Set(post.urlList)).toEqual(
      new Set([`${URL}/asr`, `${URL}/vad`, `${URL}/gone`]),
    );
  });

  it("never submits a URL off the host, nor a path that is not a route", () => {
    writeFileSync(
      join(dist, "seo-manifest.json"),
      manifest({ "/asr": "a", "https://evil.test/x": "e", "//evil.test": "e", "/../x": "e" }),
    );
    const { post } = submit();
    expect(post.urlList).toEqual([`${URL}/asr`]);
  });

  it.each(["202"])("accepts %s as success", (c) => {
    const { aws } = submit({ STUB_POST_CODE: c });
    expect(manifestUploads(aws)).toHaveLength(1);
  });

  // The site is already live: a failed notification is a warning, and the
  // manifest stays put so the next deploy retries the same diff.
  it.each(["400", "403", "422", "429", "500"])(
    "on %s warns, exits 0 and does not advance the manifest",
    (c) => {
      const { code, out, aws } = submit({ STUB_POST_CODE: c });
      expect(code).toBe(0);
      expect(out).toContain(`IndexNow answered '${c}'`);
      expect(out).toContain("the next deploy retries");
      expect(manifestUploads(aws)).toEqual([]);
    },
  );

  it("skips, without submitting, when the live key file is missing", () => {
    rmSync(join(live, `${KEY}.txt`));
    const { code, out, post } = submit();
    expect(code).toBe(0);
    expect(out).toContain("does not serve the key");
    expect(post).toBeNull();
  });

  it("skips when the live key file holds a different key", () => {
    writeFileSync(join(live, `${KEY}.txt`), "f".repeat(32));
    expect(submit().post).toBeNull();
  });

  it("skips when the artifact has no key file, or two", () => {
    rmSync(join(dist, `${KEY}.txt`));
    expect(submit().out).toContain("found 0");
    writeFileSync(join(dist, `${KEY}.txt`), KEY);
    writeFileSync(join(dist, `${"a".repeat(32)}.txt`), "a".repeat(32));
    expect(submit().out).toContain("found 2");
  });

  // A network failure is not "no previous manifest": treating it as one would
  // resubmit every page on every flaky deploy.
  it("skips, rather than resubmitting everything, when the manifest fetch times out", () => {
    const { code, out, post } = submit({ STUB_MANIFEST_DOWN: "1" });
    expect(code).toBe(0);
    expect(out).toContain("answered '000'; skipped rather than resubmit every page");
    expect(post).toBeNull();
  });

  it("skips when the site cannot be reached at all", () => {
    const { code, out, post } = submit({ STUB_DOWN: "1" });
    expect(code).toBe(0);
    expect(out).toContain("does not serve the key");
    expect(post).toBeNull();
  });

  it("warns, still exits 0, when recording the manifest fails", () => {
    const { code, out } = submit({ STUB_AWS_FAIL: "1" });
    expect(code).toBe(0);
    expect(out).toContain("recording the manifest failed");
  });

  it("raises a GitHub Actions warning annotation in CI", () => {
    const { out } = submit({ STUB_POST_CODE: "429", GITHUB_ACTIONS: "true" });
    expect(out).toMatch(/^::warning title=IndexNow::IndexNow answered '429'/m);
  });

  // #72: the sitemap's <lastmod> is the date a page's crawlable content last
  // changed — carried forward while its digest holds still, never the build date.
  describe("dates", () => {
    const SITEMAP = [
      '<?xml version="1.0" encoding="UTF-8"?>',
      '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
      `  <url><loc>${URL}/home</loc></url>`,
      `  <url><loc>${URL}/asr</loc></url>`,
      `  <url><loc>${URL}/vad</loc></url>`,
      "</urlset>",
      "",
    ].join("\n");
    beforeEach(() => writeFileSync(join(dist, "sitemap.xml"), SITEMAP));
    const lastmod = (xml: string, path: string) =>
      xml.match(new RegExp(`<loc>${URL}${path}</loc><lastmod>([^<]*)</lastmod>`))?.[1];

    it("keeps an unchanged page's date and stamps a changed or new one with today", () => {
      writeFileSync(
        join(live, "seo-manifest.json"),
        JSON.stringify({ "/home": { digest: "h1", modified: "2026-10-01" }, "/asr": { digest: "OLD", modified: "2026-10-01" } }),
      );
      submit();
      expect(JSON.parse(uploaded("seo-manifest.json"))).toEqual({
        "/home": { digest: "h1", modified: "2026-10-01" },
        "/asr": { digest: "a1", modified: TODAY },
        "/vad": { digest: "v1", modified: TODAY },
      });
      const xml = uploaded("sitemap.xml");
      expect(lastmod(xml, "/home")).toBe("2026-10-01");
      expect(lastmod(xml, "/asr")).toBe(TODAY);
      expect(lastmod(xml, "/vad")).toBe(TODAY);
    });

    it("uploads the dated sitemap with the deploy's own headers", () => {
      const { aws } = submit();
      const line = aws.find((l) => l.includes("s3://site-bucket/sitemap.xml"))!;
      expect(line).toContain("application/xml; charset=utf-8");
      expect(line).toContain("public, max-age=0, s-maxage=300, must-revalidate");
    });

    // A manifest from before #72 has digests and no dates: today, once, and
    // recorded even though nothing is submitted — or every deploy re-seeds.
    it("seeds today for a manifest without dates, and records it with nothing to submit", () => {
      writeFileSync(join(live, "seo-manifest.json"), manifest(NEXT));
      const { post, out, aws } = submit();
      expect(post).toBeNull();
      expect(out).toContain("nothing to submit");
      expect(manifestUploads(aws)).toHaveLength(1);
      expect(JSON.parse(uploaded("seo-manifest.json"))).toEqual(dated(NEXT, TODAY));
    });

    it("re-dates the sitemap on a deploy that changed nothing, so the deploy's undated one does not stick", () => {
      writeFileSync(join(live, "seo-manifest.json"), JSON.stringify(dated(NEXT, "2026-10-01")));
      const { aws } = submit();
      expect(manifestUploads(aws)).toEqual([]);
      expect(lastmod(uploaded("sitemap.xml"), "/asr")).toBe("2026-10-01");
    });

    it("still dates the sitemap when the submission fails, and leaves the manifest alone", () => {
      const { aws } = submit({ STUB_POST_CODE: "500" });
      expect(manifestUploads(aws)).toEqual([]);
      expect(lastmod(uploaded("sitemap.xml"), "/home")).toBe(TODAY);
    });

    it("treats a malformed date as none, rather than publishing it", () => {
      writeFileSync(join(live, "seo-manifest.json"), JSON.stringify({ "/home": { digest: "h1", modified: "soon" } }));
      submit();
      expect(lastmod(uploaded("sitemap.xml"), "/home")).toBe(TODAY);
    });

    it("leaves the sitemap alone when the artifact has none", () => {
      rmSync(join(dist, "sitemap.xml"));
      const { aws } = submit();
      expect(aws.some((l) => l.includes("sitemap.xml"))).toBe(false);
    });

    it("dates nothing when the previous manifest cannot be fetched", () => {
      const { aws } = submit({ STUB_MANIFEST_DOWN: "1" });
      expect(aws).toEqual([]);
    });
  });
});
