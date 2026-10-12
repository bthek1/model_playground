import { expect, test } from "../fixtures/base";

// Route-level code splitting (#66), against the built bundle. Every route's
// component is its own chunk, so a deep load now depends on a second request
// the entry used to make unnecessary: the page's HTML names that chunk in a
// `modulepreload` (scripts/seoPages.ts), and the router then imports it. A
// chunk that 404s, or a route that silently stopped splitting, leaves the app
// shell standing with an empty content area — no error, a page that "loads".
test.skip(!process.env.E2E_STATIC, "static-build pass only (E2E_STATIC=1)");

const preloadsOf = (html: string) =>
  [...html.matchAll(/<link rel="modulepreload"[^>]*href="([^"]+)"/g)].map((m) => m[1]);

test("every sitemap URL deep-loads its own route chunk and mounts it", async ({
  page,
  request,
}) => {
  test.setTimeout(180_000);
  const sitemap = await (await request.get("/sitemap.xml")).text();
  const paths = [...sitemap.matchAll(/<loc>([^<]*)<\/loc>/g)].map((m) => new URL(m[1]).pathname);
  expect(paths.length).toBeGreaterThan(40);

  const shared = new Set(preloadsOf(await (await request.get("/")).text()));
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const fetched = new Map<string, number>();
  page.on("response", (r) => fetched.set(new URL(r.url()).pathname, r.status()));

  for (const path of paths) {
    const own = preloadsOf(await (await request.get(path)).text()).filter((f) => !shared.has(f));
    // The last one is the route chunk itself; the rest are what it imports.
    expect(own.length, `${path} preloads no chunk of its own`).toBeGreaterThan(0);

    fetched.clear();
    errors.length = 0;
    await page.goto(path);
    // The router's outlet rendered something: the shell alone has an empty
    // <main>, which is what a route that failed to load leaves behind.
    await expect(page.locator("main > *").first(), path).toBeVisible();
    await expect(page.getByTestId("not-found"), path).toHaveCount(0);
    for (const file of own) expect(fetched.get(file), `${path} → ${file}`).toBe(200);
    expect(errors, path).toEqual([]);
  }
});
