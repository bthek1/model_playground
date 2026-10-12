/**
 * `<link rel="modulepreload">` for a page's own route chunk (#66).
 *
 * With `autoCodeSplitting` each route's component is a separate chunk, which
 * the browser would otherwise discover only after the entry has run and the
 * router has matched — three round trips (entry → route chunk → its imports)
 * before the page can render. Every indexable page has its own HTML file, so
 * the build knows which chunk that page will ask for and can name it, and
 * everything it imports statically, in the head.
 *
 * Pure, with relative imports only: `scripts/seoPages.ts` runs it in Node over
 * the real bundle, and `preload.test.ts` over a hand-built one.
 */

/** The part of a Rollup output chunk this needs. */
export interface ChunkLike {
  type: "chunk" | "asset";
  fileName: string;
  isEntry?: boolean;
  facadeModuleId?: string | null;
  imports?: readonly string[];
}

export type BundleLike = Readonly<Record<string, ChunkLike>>;

/** `/asr` → `asr`, `/privacy` → `privacy` (whose file is `privacy.lazy.tsx`). */
export function routeFileStem(path: string): string {
  return path.replace(/^\//, "");
}

/** Every chunk `start` imports statically, itself included, in load order. */
function closure(bundle: BundleLike, start: string, into: Set<string>): Set<string> {
  if (into.has(start)) return into;
  const chunk = bundle[start];
  if (!chunk || chunk.type !== "chunk") return into;
  for (const dep of chunk.imports ?? []) closure(bundle, dep, into);
  into.add(start);
  return into;
}

/**
 * The split chunk holding the component of `src/routes/<stem>.tsx` (or
 * `<stem>.lazy.tsx`), or `undefined` if there is none — the router plugin
 * marks the split module with a `tsr-split` query on the route file's id.
 */
export function routeChunk(bundle: BundleLike, stem: string): ChunkLike | undefined {
  const file = new RegExp(`/src/routes/${stem.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(\\.lazy)?\\.tsx(\\?|$)`);
  const candidates = Object.values(bundle).filter(
    (c) => c.type === "chunk" && !c.isEntry && c.facadeModuleId && file.test(c.facadeModuleId),
  );
  // A `.lazy.tsx` route is split by hand and carries no query; prefer the
  // plugin's split chunk when both exist.
  return (
    candidates.find((c) => c.facadeModuleId!.includes("tsr-split")) ?? candidates[0]
  );
}

/**
 * The files to preload for a page: its route chunk and that chunk's static
 * imports, minus anything the entry already loads (Vite preloads those in
 * every HTML file). Throws when the route has no chunk of its own, which means
 * either the page has no route file or the route stopped splitting.
 */
export function routePreloads(bundle: BundleLike, path: string): string[] {
  const entry = Object.values(bundle).find((c) => c.type === "chunk" && c.isEntry);
  if (!entry) throw new Error("seo: the bundle has no entry chunk");
  const chunk = routeChunk(bundle, routeFileStem(path));
  if (!chunk) {
    throw new Error(
      `seo: no split chunk for ${path} — expected src/routes/${routeFileStem(path)}.tsx to be code-split (#66)`,
    );
  }
  const already = closure(bundle, entry.fileName, new Set());
  return [...closure(bundle, chunk.fileName, new Set())].filter((f) => !already.has(f));
}

/** Insert one `modulepreload` per file before `</head>`, skipping any already there. */
export function addModulePreloads(html: string, files: readonly string[], base = "/"): string {
  const links = files
    .map((f) => `${base}${f}`)
    .filter((href) => !html.includes(`href="${href}"`))
    .map((href) => `<link rel="modulepreload" crossorigin href="${href}">`);
  if (links.length === 0) return html;
  return html.replace("</head>", () => `  ${links.join("\n    ")}\n  </head>`);
}
