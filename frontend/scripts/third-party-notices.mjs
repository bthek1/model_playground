/**
 * THIRD-PARTY-NOTICES.txt, generated from exactly what the build ships (#62).
 *
 * MIT, Apache-2.0 and BSD all require the copyright and licence notice to
 * travel with the code, and the deployed bundle *is* a redistribution of
 * React, Transformers.js, ONNX Runtime Web and ~everything else in
 * `dependencies`. So the build writes the notices itself, from the modules
 * Rolldown actually rendered into an output chunk — not from `package.json`,
 * which lists things the bundle tree-shakes away, and not from `npm ls`, which
 * cannot see that most of the weight lives in **worker** bundles.
 *
 * That last point is why the plugin is registered twice in `vite.config.ts`:
 * once in `plugins` and once in `worker.plugins`. Vite bundles each worker as
 * a separate build *during* the main one, so the worker instances record into
 * the shared `seen` map and the main build's `generateBundle` — which runs
 * after every worker has been bundled — writes the file.
 *
 * It also fails the build on a licence outside `ALLOWED`, so a copyleft or
 * unlicensed package cannot reach the deployed bundle quietly. The pure
 * helpers are exported for `src/__tests__/thirdPartyNotices.test.ts`.
 */

import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

/**
 * Licences a dependency may carry and still ship. All permissive: none asks
 * more of a redistribution than keeping its notice, which this file does.
 * Add to it deliberately, never to make a build pass.
 */
export const ALLOWED = new Set([
  "MIT",
  "Apache-2.0",
  "BSD-2-Clause",
  "BSD-3-Clause",
  "ISC",
  "0BSD",
  "CC0-1.0",
  "BlueOak-1.0.0",
  "Unlicense",
  "Python-2.0",
  "CC-BY-4.0",
]);

const NOTICE_FILE = /^(licen[cs]e|copying|notice)(\.[a-z]+)?$/i;

/**
 * The package directory a module id lives in, or null for app code. Uses the
 * **last** `node_modules/` segment, so a nested copy resolves to itself.
 */
export function packageRoot(id) {
  const clean = id.replace(/^\0/, "").split("?")[0].replace(/\\/g, "/");
  const at = clean.lastIndexOf("/node_modules/");
  if (at < 0) return null;
  const rest = clean.slice(at + "/node_modules/".length).split("/");
  const depth = rest[0].startsWith("@") ? 2 : 1;
  if (rest.length <= depth) return null;
  return clean.slice(0, at) + "/node_modules/" + rest.slice(0, depth).join("/");
}

/** The SPDX expression a package.json declares, whatever shape it used. */
export function declaredLicence(pkg) {
  const l = pkg.license ?? pkg.licenses;
  if (typeof l === "string") return l;
  if (Array.isArray(l)) return l.map((x) => (typeof x === "string" ? x : x.type)).join(" OR ");
  if (l && typeof l === "object" && l.type) return l.type;
  return "UNKNOWN";
}

/**
 * Whether an SPDX expression is acceptable: `OR` needs one allowed operand,
 * `AND` needs all of them. Parentheses are flattened, which is right for every
 * expression npm packages actually use (`(MIT OR Apache-2.0)`).
 */
export function isAllowed(expression, allowed = ALLOWED) {
  const expr = expression.replace(/[()]/g, " ").trim();
  return expr
    .split(/\s+OR\s+/)
    .some((alt) => alt.split(/\s+AND\s+/).every((id) => allowed.has(id.trim())));
}

/** Read one package's name, version, licence and notice files. */
export function readPackage(root) {
  const pkg = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));
  const texts = readdirSync(root)
    .filter((f) => NOTICE_FILE.test(f))
    .sort()
    .map((f) => ({ file: f, text: readFileSync(join(root, f), "utf8").trim() }));
  const repo = typeof pkg.repository === "string" ? pkg.repository : pkg.repository?.url;
  return {
    name: pkg.name,
    version: pkg.version,
    licence: declaredLicence(pkg),
    repository: repo,
    texts,
  };
}

/**
 * Resolve a package root to the directory holding its real `package.json`.
 * Some packages ship a stub `package.json` in a subpath (`{ "type": "module" }`)
 * — the root found by `packageRoot` is always the real one, but guard anyway.
 */
function hasManifest(root) {
  return existsSync(join(root, "package.json"));
}

/** The notices file's text, packages sorted by name. */
export function renderNotices(packages) {
  const header = [
    "THIRD-PARTY SOFTWARE NOTICES",
    "",
    "This site's code is MIT-licensed (https://github.com/bthek1/model_playground).",
    "It bundles the open-source packages below. Each is listed with its licence and",
    "the notice files it ships, as those licences require. Generated at build time",
    "from the modules actually present in the bundle.",
    "",
    `${packages.length} packages.`,
    "",
  ].join("\n");
  const body = [...packages]
    .sort((a, b) => a.name.localeCompare(b.name) || a.version.localeCompare(b.version))
    .map((p) => {
      const lines = [`${p.name}@${p.version}`, `License: ${p.licence}`];
      if (p.repository) lines.push(`Source: ${p.repository}`);
      if (!p.texts.length) lines.push("", "(No licence file is included in the package.)");
      for (const t of p.texts) lines.push("", t.text);
      return lines.join("\n");
    })
    .join("\n\n" + "-".repeat(78) + "\n\n");
  return `${header}\n${"=".repeat(78)}\n\n${body}\n`;
}

/** Package roots seen across the main build and every worker build. */
const seen = new Set();

/**
 * @param {{ emit?: boolean, fileName?: string, allowed?: Set<string> }} options
 *   `emit: false` for the `worker.plugins` instance — it only records.
 */
export function thirdPartyNotices({
  emit = true,
  fileName = "THIRD-PARTY-NOTICES.txt",
  allowed = ALLOWED,
} = {}) {
  return {
    name: "third-party-notices",
    apply: "build",
    generateBundle(_options, bundle) {
      for (const output of Object.values(bundle)) {
        if (output.type !== "chunk") continue;
        for (const [id, info] of Object.entries(output.modules)) {
          if (!info.renderedLength) continue;
          const root = packageRoot(id);
          if (root && hasManifest(root)) seen.add(root);
        }
      }
      if (!emit) return;

      const packages = [...seen].map(readPackage);
      // One entry per name@version, however many copies resolved.
      const unique = [...new Map(packages.map((p) => [`${p.name}@${p.version}`, p])).values()];
      const refused = unique.filter((p) => !isAllowed(p.licence, allowed));
      if (refused.length) {
        this.error(
          "third-party-notices: licence not on the allowlist (scripts/third-party-notices.mjs):\n" +
            refused.map((p) => `  ${p.name}@${p.version} — ${p.licence}`).join("\n"),
        );
      }
      this.emitFile({ type: "asset", fileName, source: renderNotices(unique) });
    },
  };
}
