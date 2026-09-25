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
import { readdirSync, readFileSync, statSync } from "node:fs";
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
 */
const ENTRY_BUDGET_GZIP_KB = 560;
const ENTRY_BUDGET_RAW_KB = 1860;

/**
 * Markers that must not appear in the entry chunk. Each is a string the library
 * emits into its own bundled output, not something app code would write.
 */
const MUST_STAY_SPLIT = [["echarts", /\becharts\b/]];

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

if (failed) {
  console.error("\nBundle budget exceeded. See scripts/check-bundle.mjs.");
  process.exit(1);
}
console.log("\nBundle budget OK.");
