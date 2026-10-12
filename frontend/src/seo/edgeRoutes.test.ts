import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { APP_ONLY_PATHS, SITE_PAGES } from "./pages";

// The CloudFront function serves an indexable route its own index.html and
// everything else the root shell (#64). It runs at the edge with no module
// system, so it cannot import SITE_PAGES — it carries a copy, and this is what
// keeps the copy honest. Evaluated from the file's exact text, as CloudFront
// runs it.
const source = readFileSync(join(__dirname, "../../../infra/site/spa-rewrite.js"), "utf8");
const { routes, appRoutes } = new Function(
  `${source}\nreturn { routes: ROUTES, appRoutes: APP_ROUTES };`,
)() as { routes: Record<string, boolean>; appRoutes: Record<string, boolean> };

describe("infra/site/spa-rewrite.js ROUTES and APP_ROUTES", () => {
  it("lists exactly the pages the build writes a file for", () => {
    const listed = Object.keys(routes);
    const pages = SITE_PAGES.map((p) => p.path);
    // Named both ways: a page missing from the function is served the generic
    // shell; a route listed with no file answers 403 from the bucket.
    expect(pages.filter((p) => !listed.includes(p)), "pages missing from ROUTES").toEqual([]);
    expect(listed.filter((p) => !pages.includes(p)), "ROUTES with no page").toEqual([]);
  });

  // Every other path gets /404.html and a real 404 (#63), so an app-only route
  // missing here would 404 at the edge while working in dev.
  it("lists the same app-only routes as APP_ONLY_PATHS", () => {
    expect(Object.keys(appRoutes).sort()).toEqual([...APP_ONLY_PATHS].sort());
  });
});
