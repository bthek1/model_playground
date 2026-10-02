// @vitest-environment node
//
// #61 put the backend's secrets and the frontend's VITE_* keys in one root
// `.env`, and pointed Vite's `envDir` at it. That is safe only because Vite
// inlines `import.meta.env.X` for keys matching `envPrefix` (default `VITE_`)
// and nothing else. A config that widened `envPrefix` — `""` looks harmless,
// and `loadEnv(mode, dir, "")` sits three lines away doing exactly that for
// the *proxy* config — would ship SECRET_KEY to every browser with no error
// anywhere. So: build a module that asks for both kinds of key, against a
// sentinel `.env`, with this repo's own `envDir`/`envPrefix`, and read the
// bytes that come out.

import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { build, type UserConfig } from "vite";
import { afterAll, describe, expect, it } from "vitest";
import viteConfig from "../../vite.config";

const SECRET = "backend-only-sentinel-6c1f0e";
const PUBLIC = "vite-public-sentinel-93ab2d";

const resolved = (
  viteConfig as (env: { mode: string; command: "build" }) => UserConfig
)({ mode: "production", command: "build" });

describe("the shared root .env (#61)", () => {
  const dir = mkdtempSync(join(tmpdir(), "env-exposure-"));
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  it("is read from the project root", () => {
    expect(resolved.envDir).toBe(resolve(__dirname, "../../.."));
  });

  it("keeps Vite's default VITE_ prefix", () => {
    // Unset means `VITE_`; anything else must be argued for in this test.
    expect(resolved.envPrefix).toBeUndefined();
  });

  it("inlines VITE_* keys and never a backend key", async () => {
    writeFileSync(
      join(dir, ".env"),
      `SECRET_KEY=${SECRET}\nDATABASE_URL=postgres://u:${SECRET}@db/x\nVITE_PROBE=${PUBLIC}\n`,
    );
    writeFileSync(
      join(dir, "entry.js"),
      "export const pub = import.meta.env.VITE_PROBE;\n" +
        "export const sec = import.meta.env.SECRET_KEY;\n" +
        "export const all = JSON.stringify(import.meta.env);\n",
    );
    await build({
      root: dir,
      configFile: false,
      logLevel: "silent",
      envDir: dir,
      envPrefix: resolved.envPrefix,
      build: {
        outDir: join(dir, "dist"),
        lib: { entry: join(dir, "entry.js"), formats: ["es"], fileName: "out" },
        minify: false,
      },
    });
    const out = readdirSync(join(dir, "dist"))
      .map((f) => readFileSync(join(dir, "dist", f), "utf8"))
      .join("\n");

    expect(out).toContain(PUBLIC); // the probe is live, so an absence means something
    expect(out).not.toContain(SECRET);
  }, 60_000);

  it("documents every VITE_* key the app reads in the root .env.example", () => {
    const src = resolve(__dirname, "..");
    const used = new Set<string>();
    for (const file of readdirSync(src, { recursive: true, encoding: "utf8" })) {
      if (!/\.(ts|tsx)$/.test(file)) continue;
      const text = readFileSync(join(src, file), "utf8");
      for (const m of text.matchAll(/import\.meta\.env\.(VITE_[A-Z0-9_]+)/g)) used.add(m[1]);
    }
    // Live probe: these two are read today, so an empty set means a broken scan.
    expect(used).toContain("VITE_BACKEND");
    expect(used).toContain("VITE_API_BASE_URL");
    used.delete("VITE_PROBE"); // this file's own sentinel

    const example = readFileSync(resolve(__dirname, "../../../.env.example"), "utf8");
    const documented = new Set([...example.matchAll(/^(VITE_[A-Z0-9_]+)=/gm)].map((m) => m[1]));
    expect([...used].filter((k) => !documented.has(k))).toEqual([]);
  });

  it("leaves no per-half .env.example behind", () => {
    expect(existsSync(resolve(__dirname, "../../.env.example"))).toBe(false);
  });
});
