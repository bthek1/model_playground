/**
 * Bundle budget for the production build.
 *
 * Two regressions this catches, both of which look fine locally and only hurt
 * on a cold first paint:
 *
 *  1. A heavy library imported at the top level of a module the entry reaches.
 *     `echarts` (1.1 MB) is supposed to stay behind the lazy EChart wrapper; an
 *     accidental static import folds it into the entry chunk.
 *  2. Drift: the entry chunk growing a little at a time until first paint is
 *     measured in megabytes.
 *
 *     npm run check:bundle      (runs against an existing dist/)
 *
 * Budgets are deliberately close to the measured size — a few hundred KB of
 * headroom, not a few megabytes — so they fail on the import that caused it
 * rather than a year later.
 */

import { gzipSync } from "node:zlib";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const assets = join(root, "dist", "assets");

/**
 * Measured 2026-09-14: 1,589 KB raw / 487 KB gzip.
 * Measured 2026-09-25: 1,803 KB raw / 543 KB gzip, after the three Tabular
 * routes landed. Raw raised 1800 → 1860; gzip left where it was, because that
 * is the number a user actually waits for and it still has headroom.
 *
 * The raise is app code, not a leaked library: the three routes add ~46 KB raw
 * and ~13 KB gzip between them, `echarts` is still split, and the bundled CSVs
 * and series are behind dynamic imports for exactly this reason — a static
 * `?raw` import of the six sample files put 97 KB of incompressible text into
 * this chunk and is what caught them.
 * Measured 2026-09-28: 1,861 KB raw / 561 KB gzip, after `/rl` (#51, #52) and
 * `/robotics` (#53). Raw raised 1860 → 1900 and gzip 560 → 575. Again app code:
 * `echarts` is still split, the CartPole fixture is test-only, and the RL
 * module is plain arithmetic with no library behind it. Every route is in this
 * chunk — there is no route-level code splitting — so each new page moves it;
 * lazy route components are the structural fix, and the next raise should be
 * that instead.
 * Measured 2026-10-09: 1,896 KB raw / 577 KB gzip, after the SEO work
 * (#63–#65), against 1,884 / 573 at the commit before it. Gzip raised
 * 575 → 580, which is the raise the note above says should not happen: the
 * structural fix is planned as #66 and was deliberately left out of that
 * batch. What grew is text the pages need rather than code — a one-line
 * description per task (it is each route's meta description and its line on
 * the home page's task index), the head sync and the not-found page. When #66
 * lands, take this back below 575.
 */
const ENTRY_BUDGET_GZIP_KB = 580;
const ENTRY_BUDGET_RAW_KB = 1900;

/**
 * Markers that must not appear in the entry chunk. Each is a string the library
 * emits into its own bundled output, not something app code would write.
 */
const MUST_STAY_SPLIT = [
  ["echarts", /\becharts\b/],
  // The analytics SDK (#60) is ~49 KB gzip and behind `src/analytics/`'s
  // dynamic import; `posthog-js` is the library's own `$lib` name.
  ["posthog-js", /\bposthog-js\b/],
];

/**
 * A PostHog **personal** API key (`phx_…`) can read and delete every project in
 * the organisation. The browser only ever needs the public project key
 * (`phc_…`). Vite keeps a non-`VITE_` variable out of the bundle by prefix
 * alone, so this is the guard against the day someone renames one: any
 * `phx_` token anywhere in `dist/` fails the build — in CI and in the image.
 */
const SECRET_PATTERNS = [["PostHog personal API key (phx_…)", /phx_[A-Za-z0-9]{16,}/]];

/**
 * `onnxruntime-web` is deliberately NOT in the list above, because today it *is*
 * in the entry chunk and asserting otherwise would just be a failing test.
 *
 * The chain: every vision route uses `hooks/useImagePick.ts`, which imports
 * `vision/image.ts`, which takes `RawImage` from the `@huggingface/transformers`
 * package root — and that pulls the library, and ORT with it, into the eagerly
 * imported route graph. The 21 MB `.wasm` files are still fetched on demand, so
 * the cost is JS weight on first paint, not a multi-megabyte download.
 *
 * Getting `RawImage` without the rest of the library is an app-architecture
 * change that touches the documented toPayload/fromPayload contract, so it is
 * tracked separately rather than done inside a deployment change. Until then the
 * size budget below is what keeps it from growing.
 */

let failed = false;
function fail(message) {
  console.error(`  ✗ ${message}`);
  failed = true;
}
function pass(message) {
  console.log(`  ✓ ${message}`);
}

let entries;
try {
  entries = readdirSync(assets);
} catch {
  console.error(`No build found at ${assets}. Run \`npm run build\` first.`);
  process.exit(1);
}

// Vite names the entry chunk `index-<hash>.js`; the CSS shares the prefix.
const entryName = entries.find((f) => /^index-[\w-]+\.js$/.test(f));
if (!entryName) {
  console.error(`Could not find an entry chunk (index-*.js) in ${assets}.`);
  process.exit(1);
}

const entryPath = join(assets, entryName);
const source = readFileSync(entryPath);
const rawKb = statSync(entryPath).size / 1024;
const gzipKb = gzipSync(source).length / 1024;

console.log(`Entry chunk: ${entryName}`);

if (rawKb > ENTRY_BUDGET_RAW_KB) {
  fail(`raw ${rawKb.toFixed(0)} KB exceeds budget ${ENTRY_BUDGET_RAW_KB} KB`);
} else {
  pass(`raw ${rawKb.toFixed(0)} KB (budget ${ENTRY_BUDGET_RAW_KB} KB)`);
}

if (gzipKb > ENTRY_BUDGET_GZIP_KB) {
  fail(`gzip ${gzipKb.toFixed(0)} KB exceeds budget ${ENTRY_BUDGET_GZIP_KB} KB`);
} else {
  pass(`gzip ${gzipKb.toFixed(0)} KB (budget ${ENTRY_BUDGET_GZIP_KB} KB)`);
}

const text = source.toString("utf8");
for (const [name, pattern] of MUST_STAY_SPLIT) {
  if (pattern.test(text)) {
    fail(`${name} is in the entry chunk — it must stay lazily loaded`);
  } else {
    pass(`${name} stays out of the entry chunk`);
  }
}

/** Every text file the deploy uploads — not just the entry chunk. */
function* textFiles(dir) {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) yield* textFiles(path);
    else if (/\.(js|mjs|css|html|json|map|txt|webmanifest|svg)$/.test(name)) yield path;
  }
}
const dist = join(root, "dist");
for (const [label, pattern] of SECRET_PATTERNS) {
  const hits = [...textFiles(dist)].filter((path) => pattern.test(readFileSync(path, "utf8")));
  if (hits.length > 0) {
    fail(`a ${label} is in the build: ${hits.map((h) => h.slice(dist.length + 1)).join(", ")}`);
  } else {
    pass(`no ${label} anywhere in dist/`);
  }
}

// The bundled packages' licences require their notices to ship with them
// (#62). `scripts/third-party-notices.mjs` writes the file during the build;
// this proves it did, and that the **worker** builds — where Transformers.js
// and ONNX Runtime live — were counted, not only the main one.
const notices = join(dist, "THIRD-PARTY-NOTICES.txt");
if (!existsSync(notices)) {
  fail("dist/THIRD-PARTY-NOTICES.txt is missing — the notices plugin did not run");
} else {
  const text = readFileSync(notices, "utf8");
  const missing = ["react", "onnxruntime-web", "@huggingface/transformers"].filter(
    (name) => !text.includes(`\n${name}@`),
  );
  if (missing.length) fail(`THIRD-PARTY-NOTICES.txt omits ${missing.join(", ")}`);
  else pass("THIRD-PARTY-NOTICES.txt covers the main and worker bundles");
}

if (failed) {
  console.error("\nBundle check failed. See scripts/check-bundle.mjs.");
  process.exit(1);
}
console.log("\nBundle budget OK.");
