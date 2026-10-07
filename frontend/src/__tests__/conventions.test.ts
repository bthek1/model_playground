// The task-page conventions from #58, checked against the source.
//
// Each helper (useTaskSlots, RunButton, useCatalogueEntry, the route-test
// contract) replaced a pattern that had been copied 20–35 times. Nothing about
// a copy fails when a new route writes the old pattern again — it renders, and
// its tests pass — so the drift would be invisible until the next audit. These
// assertions make the old pattern fail in CI instead, with the file named.
//
// They read source text, deliberately: the property is "how the file is
// written", which no render can observe. Each one is narrow enough that a
// legitimate exception has to be added to its list below, by name.

import { describe, expect, it } from "vitest";

const routes = import.meta.glob<string>("../routes/*.tsx", {
  query: "?raw",
  import: "default",
  eager: true,
});
const routeTests = import.meta.glob<string>("./routes/*.test.tsx", {
  query: "?raw",
  import: "default",
  eager: true,
});
const hooks = import.meta.glob<string>("../hooks/*.ts", {
  query: "?raw",
  import: "default",
  eager: true,
});

const name = (path: string) => path.split("/").pop()!.replace(/\.(test\.)?tsx?$/, "");
const entries = (files: Record<string, string>) =>
  Object.entries(files).map(([path, src]) => [name(path), src] as const);

/** A route that lets the user pick a model to download is a task page. */
const taskRoutes = entries(routes).filter(([, src]) => src.includes("useModelSelection("));

describe("task-page conventions (#58)", () => {
  it("finds the routes it is meant to check", () => {
    // Guards the globs themselves: a moved directory would make every
    // assertion below pass vacuously.
    expect(Object.keys(routes).length).toBeGreaterThan(30);
    expect(taskRoutes.length).toBeGreaterThan(25);
    expect(Object.keys(hooks).length).toBeGreaterThan(20);
  });

  it("wires SELECT and LOAD through useTaskSlots on every task route", () => {
    const missing = taskRoutes
      .filter(([, src]) => !src.includes("useTaskSlots("))
      .map(([n]) => n);
    expect(missing).toEqual([]);
  });

  it("leaves the error split and the cache re-probe to useTaskSlots", () => {
    const handWritten = taskRoutes
      .filter(
        ([, src]) =>
          /\bconst loadError = [\w.]*status === "error"/.test(src) ||
          src.includes("useCacheRefresh("),
      )
      .map(([n]) => n);
    expect(handWritten).toEqual([]);
  });

  it("uses RunButton for a trigger, not a hand-written spinner button", () => {
    // The pattern RunButton replaced: a Button whose children switch between a
    // spinner + running label and an icon + verb on a run/training flag.
    // `/asr`'s Upload button spins on `decoding` — an input source, not a
    // trigger — so the flag is what the check keys on.
    // The props may hold `=>`, so they are matched up to the next <Button
    // rather than to the first ">".
    const spinnerTrigger =
      /<Button\b(?:(?!<\/?Button)[\s\S])*?>\s*\{[^?{}]*\b(running|training)\b[^?{}]*\?\s*\(\s*<>\s*<Loader2/;
    const handWritten = entries(routes)
      .filter(([, src]) => spinnerTrigger.test(src))
      .map(([n]) => n);
    expect(handWritten).toEqual([]);
  });

  it("looks catalogue entries up with useCatalogueEntry, not by hand", () => {
    const handWritten = entries(hooks)
      .filter(([n]) => !n.endsWith(".test"))
      .filter(([, src]) => /\.find\(\(m\) => m\.id === \w+\)\s*\?\?/.test(src))
      .map(([n]) => n);
    expect(handWritten).toEqual([]);
  });

  it("registers the §8 contract in every task route's test", () => {
    const tests = new Map(entries(routeTests));
    const missing = taskRoutes
      .map(([n]) => n)
      .filter((n) => !tests.get(n)?.includes("describeTaskPageContract("));
    expect(missing).toEqual([]);
  });
});

// ── The static build (#57) ───────────────────────────────────────────────────
//
// `VITE_BACKEND=off` ships to S3 + CloudFront, where a request to /api does not
// fail: it is rewritten to index.html and answered 200. So an ungated /api
// caller added later would not break the deploy — it would hand a JSON parser a
// page of HTML, on the live site only. `staticBuild.test.tsx` and
// `static-build.spec.ts` catch it on the pages they visit; this catches it at
// the import, on every page, and names the file.

const source = import.meta.glob<string>("../**/*.{ts,tsx}", {
  query: "?raw",
  import: "default",
  eager: true,
});
/** App source only: no tests, no test harness, no generated route tree, no
 *  declarations (`vite-env.d.ts` types the flag; it does not read it). */
const app = Object.entries(source)
  .map(([path, src]) => [path.replace(/^\.\.\//, ""), src] as const)
  .filter(
    ([path]) =>
      !/\.(test|spec)\.tsx?$/.test(path) &&
      !path.endsWith(".d.ts") &&
      !path.startsWith("test/") &&
      !path.startsWith("__tests__/") &&
      path !== "routeTree.gen.ts",
  );
const srcOf = (path: string) => app.find(([p]) => p === path)?.[1] ?? "";

/**
 * Every module outside `src/api/` that imports from it, and why the static
 * build never reaches its request. A new one fails the test below: gate it on
 * `BACKEND_ENABLED` from `@/lib/features`, then add it here with the reason.
 */
const API_CALLERS: Record<string, string> = {
  "hooks/useAuth.ts": "useMe is disabled; login/register run only on /login and /signup, which redirect",
  "hooks/useModels.ts": "only ModelCatalogCard uses it, and it is not rendered",
  "components/ui/BackendStatus.tsx": "only /login and /signup render it, and they redirect",
  "hooks/useTaskPoller.ts": "only TaskTrigger uses it, and no route renders that",
  "components/TaskTrigger.tsx": "no route renders it",
};

describe("static build conventions (#57)", () => {
  it("finds the source it is meant to check", () => {
    expect(app.length).toBeGreaterThan(200);
    expect(srcOf("lib/features.ts")).toContain("BACKEND_ENABLED");
  });

  it("reads VITE_BACKEND in exactly one place", () => {
    const readers = app.filter(([, src]) => src.includes("VITE_BACKEND")).map(([p]) => p);
    expect(readers).toEqual(["lib/features.ts"]);
  });

  it("reaches /api only through the named, gated callers", () => {
    const importsApi = /from\s+["'](?:@\/api\/|(?:\.\.\/)+api\/)/;
    const callers = app
      .filter(([p, src]) => !p.startsWith("api/") && importsApi.test(src))
      .map(([p]) => p)
      .sort();
    expect(callers).toEqual(Object.keys(API_CALLERS).sort());
  });

  it("never calls /api around the client — no raw fetch or axios outside src/api", () => {
    const raw = app
      .filter(([p]) => !p.startsWith("api/"))
      .filter(([, src]) => /fetch\(\s*[`"']\/api|from\s+["']axios["']/.test(src))
      .map(([p]) => p);
    expect(raw).toEqual([]);
  });

  it("gates useMe on the flag", () => {
    expect(srcOf("hooks/useAuth.ts")).toMatch(/enabled:\s*BACKEND_ENABLED\s*&&/);
  });

  it.each(["routes/index.tsx", "routes/login.tsx", "routes/signup.tsx"])(
    "%s redirects before it renders when the flag is off",
    (path) => {
      expect(srcOf(path)).toMatch(
        /beforeLoad:[\s\S]*?if \(!BACKEND_ENABLED\) throw redirect\(\{ to: ['"]\/home['"]/,
      );
    },
  );

  it("renders the registry catalogue only behind the flag", () => {
    const renderers = app.filter(([, src]) => src.includes("<ModelCatalogCard"));
    expect(renderers.map(([p]) => p).sort()).toEqual(["routes/home.tsx", "routes/playground.tsx"]);
    for (const [path, src] of renderers) {
      const ungated = src.match(/<ModelCatalogCard/g)!.length;
      const gated = src.match(/BACKEND_ENABLED && <ModelCatalogCard/g)?.length ?? 0;
      expect(gated, path).toBe(ungated);
    }
  });

  it("renders TaskTrigger nowhere", () => {
    // `<TaskTrigger` as JSX — not `TaskTriggerResponse>` in a generic.
    const jsx = /<TaskTrigger[\s/>]/;
    expect(app.filter(([, src]) => jsx.test(src)).map(([p]) => p)).toEqual([]);
  });
});

// Analytics (#60). Each rule here fails silently if broken: a second reader of
// the key can enable analytics where the flag says it is off; one static
// import of the SDK folds ~50 KB gzip into the entry chunk and undoes the lazy
// load; a page that calls the SDK directly skips the allowlist.
describe("analytics conventions (#60)", () => {
  it("reads VITE_POSTHOG_* only in src/lib/features.ts", () => {
    const readers = app.filter(([, src]) => /import\.meta\.env\.VITE_POSTHOG_/.test(src));
    expect(readers.map(([p]) => p)).toEqual(["lib/features.ts"]);
  });

  it("imports posthog-js only in src/analytics/client.ts", () => {
    const importers = app.filter(([, src]) => /from\s+["']posthog-js|import\(\s*["']posthog-js/.test(src));
    expect(importers.map(([p]) => p)).toEqual(["analytics/client.ts"]);
  });

  it("reaches the client only by dynamic import, from the facade", () => {
    // `./client` means the analytics client only inside analytics/; every
    // modality has a `client.ts` of its own.
    const statically = (path: string, src: string) =>
      /^import\s+(?!type\b)[^;]*from\s+["']@\/analytics\/client["']/m.test(src) ||
      (path.startsWith("analytics/") &&
        /^import\s+(?!type\b)[^;]*from\s+["']\.\/client["']/m.test(src));
    expect(app.filter(([p, src]) => statically(p, src)).map(([p]) => p)).toEqual([]);
    const dynamic = app.filter(([, src]) => /import\(\s*["'](\.\/client|@\/analytics\/client)["']\s*\)/.test(src));
    expect(dynamic.map(([p]) => p)).toEqual(["analytics/index.ts"]);
  });

  it("sends events only through the facade's track/pageview", () => {
    const direct = app
      .filter(([p]) => !p.startsWith("analytics/"))
      .filter(([, src]) => /\bposthog\.(capture|init|identify)\b/.test(src));
    expect(direct.map(([p]) => p)).toEqual([]);
  });
});

// ── Legal (#62) ─────────────────────────────────────────────────────────────
//
// Each claim the legal pages make is only as true as the list it is rendered
// from. These read the source for the three ways a list falls behind the code:
// a catalogue `/licences` cannot see, a storage key the privacy notice does not
// name, and a generative page whose output is not labelled machine-made.

import { MODEL_CATALOGUES } from "@/legal/catalogues";
import { STORED_ITEMS } from "@/legal/storage";

/** Resolve a `CONST_NAME` to the string literal it is assigned in `src`. */
function resolveName(src: string, token: string): string | null {
  if (/^["'`]/.test(token)) return token.slice(1, -1);
  const m = new RegExp(`\\b${token}\\s*=\\s*["'\`]([^"'\`]+)["'\`]`).exec(src);
  return m ? m[1] : null;
}

/** Every key the source writes to device storage, with the file that does it. */
function storageKeys(): { key: string; file: string }[] {
  const found: { key: string; file: string }[] = [];
  const patterns = [
    /localStorage\.setItem\(\s*(["'`][^"'`]+["'`]|[A-Z_][A-Z0-9_]*)/g,
    /indexedDB\.open\(\s*(["'`][^"'`]+["'`]|[A-Z_][A-Z0-9_]*)/g,
    /caches\.open\(\s*(["'`][^"'`]+["'`]|[A-Z_][A-Z0-9_]*)/g,
  ];
  for (const [file, src] of app) {
    for (const re of patterns) {
      for (const m of src.matchAll(re)) {
        const key = resolveName(src, m[1]);
        found.push({ key: key ?? `<unresolved ${m[1]}>`, file });
      }
    }
    // A persisted Zustand store writes under its `name`.
    if (/\bpersist\(/.test(src)) {
      for (const m of src.matchAll(/\bname:\s*["'`]([^"'`]+)["'`]/g)) found.push({ key: m[1], file });
    }
  }
  return found;
}

describe("legal conventions (#62)", () => {
  it("names every device-storage key in the privacy notice's table", () => {
    const keys = storageKeys();
    // Guards the scan: a refactor that hid every key would pass vacuously.
    expect(keys.map((k) => k.key)).toEqual(expect.arrayContaining(["theme", "model-prefs", "transformers-cache"]));
    const disclosed = new Set(STORED_ITEMS.map((s) => s.key));
    const undisclosed = keys.filter((k) => !disclosed.has(k.key)).map((k) => `${k.file}: ${k.key}`);
    expect(undisclosed).toEqual([]);
  });

  it("lists every model catalogue on /licences", () => {
    // A catalogue is an exported array of entries; one off MODEL_CATALOGUES is
    // a set of models with no attribution and no licence coverage test.
    const listed = new Set(MODEL_CATALOGUES.map((c) => c.name));
    const declared = app.flatMap(([file, src]) =>
      [...src.matchAll(/^export const ([A-Z0-9_]+(?:_MODELS|_PAIRS|_ENTRIES)):\s*[A-Za-z]+\[\]\s*=/gm)].map(
        (m) => ({ file, name: m[1] }),
      ),
    );
    expect(declared.length).toBeGreaterThan(25);
    // GROUNDING_ENTRIES is a filter of ROBOTICS_ENTRIES, which is listed.
    const unlisted = declared
      .filter((d) => !listed.has(d.name) && d.name !== "GROUNDING_ENTRIES")
      .map((d) => `${d.file}: ${d.name}`);
    expect(unlisted).toEqual([]);
  });

  it("labels the output of every generative page as machine-made", () => {
    const GENERATIVE = [
      "text-generation",
      "summarization",
      "translation",
      "image-text-to-text",
      "visual-question-answering",
      "video-text-to-text",
      "text-to-speech",
    ];
    // Within the OutputPanel's opening props — bounded by distance, since a
    // prop's own JSX (a title with an icon) puts `>` inside the tag.
    const labelled = (src: string) => {
      const at = src.indexOf("<OutputPanel");
      return at >= 0 && /\bgenerated=/.test(src.slice(at, at + 800));
    };
    const unlabelled = GENERATIVE.filter((r) => !labelled(srcOf(`routes/${r}.tsx`)));
    expect(unlabelled).toEqual([]);
  });

  it("renders the legal footer from the app shell", () => {
    expect(srcOf("components/layout/AppLayout.tsx")).toContain("<LegalFooter");
  });
});
