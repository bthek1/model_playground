import { expect, test } from "../fixtures/base";

// The site's SEO (#63, #64, #65), against the built bundle: the per-route HTML
// files, the sitemap and robots.txt are build output, so only the static pass
// has them. `vite preview` serves a route its own file the way the CloudFront
// function does (scripts/seoPages.ts); the edge's redirects are pinned by
// infra/site/spa-rewrite.test.ts and the post-deploy smoke test instead.
test.skip(!process.env.E2E_STATIC, "static-build pass only (E2E_STATIC=1)");

const ORIGIN = "https://playground.benedictthekkel.com";

const tag = (html: string, re: RegExp) => html.match(re)?.[1];

test.describe("SEO (static build)", () => {
  // What a crawler that runs no JavaScript gets — every page, from the sitemap
  // it is told about, each with its own head and a heading in the body.
  test("every sitemap URL serves its own title, canonical and heading without JS", async ({
    request,
  }) => {
    const robots = await (await request.get("/robots.txt")).text();
    expect(robots).toContain(`Sitemap: ${ORIGIN}/sitemap.xml`);

    const sitemap = await request.get("/sitemap.xml");
    expect(sitemap.status()).toBe(200);
    const locs = [...(await sitemap.text()).matchAll(/<loc>([^<]*)<\/loc>/g)].map((m) => m[1]);
    expect(locs.length).toBeGreaterThan(40);

    const titles = new Set<string>();
    for (const loc of locs) {
      const path = new URL(loc).pathname;
      const res = await request.get(path);
      expect(res.status(), path).toBe(200);
      const html = await res.text();
      expect(tag(html, /<link rel="canonical" href="([^"]*)"/), path).toBe(loc);
      expect(tag(html, /<meta property="og:url" content="([^"]*)"/), path).toBe(loc);
      expect(html, path).toMatch(/<div id="root"><main[^>]*>.*<h1[^>]*>[^<]+<\/h1>/s);
      titles.add(tag(html, /<title>([^<]*)<\/title>/)!);
    }
    // One title per page: the shell's single title everywhere was the finding.
    expect(titles.size).toBe(locs.length);
  });

  test("the app mounts over the static summary and keeps its head", async ({ page }) => {
    await page.goto("/asr");
    await expect(page.getByRole("heading", { level: 1 })).toHaveText("Automatic Speech Recognition");
    // The static <h1> is gone: createRoot replaced #root's children.
    await expect(page.locator("h1")).toHaveCount(1);
    await expect(page).toHaveTitle("Automatic Speech Recognition in your browser · Model Playground");
    await expect(page.locator('link[rel="canonical"]')).toHaveAttribute("href", `${ORIGIN}/asr`);
  });

  // Client-side navigation never fetches the next route's file, so the head
  // has to follow by hand.
  test("the head follows a client-side navigation", async ({ page }) => {
    await page.goto("/home");
    await page.getByRole("link", { name: "Depth Estimation" }).click();
    await expect(page).toHaveURL(/\/depth$/);
    await expect(page).toHaveTitle("Depth Estimation in your browser · Model Playground");
    await expect(page.locator('link[rel="canonical"]')).toHaveAttribute("href", `${ORIGIN}/depth`);
    await expect(page.locator('meta[name="description"]')).toHaveAttribute(
      "content",
      /Depth Anything/,
    );
    const data = JSON.parse((await page.locator("#structured-data").textContent())!);
    expect(data.itemListElement.at(-1).item).toBe(`${ORIGIN}/depth`);
  });

  test("home lists every task, and a breadcrumb leads back to its category", async ({ page }) => {
    await page.goto("/home");
    await expect(page.getByRole("heading", { name: "What you can run" })).toBeVisible();
    await expect(page.locator("#computer-vision").getByRole("link", { name: "Object Detection", exact: true })).toBeVisible();

    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto("/object-detection");
    await page.getByRole("navigation", { name: "Breadcrumb" }).getByRole("link", { name: "Computer Vision" }).click();
    await expect(page).toHaveURL(/\/home#computer-vision$/);
  });

  // The edge answers an unknown path with the shell and a 200 — it cannot know
  // the router's routes — so the page has to say noindex itself (#63).
  test("an unknown path is a not-found page that asks not to be indexed", async ({ page }) => {
    await page.goto("/no-such-page");
    await expect(page.getByTestId("not-found")).toBeVisible();
    await expect(page.getByRole("heading", { name: "Page not found" })).toBeVisible();
    await expect(page.locator('meta[name="robots"]')).toHaveAttribute("content", "noindex");
    await expect(page.locator('link[rel="canonical"]')).toHaveCount(0);

    await page.getByRole("link", { name: "Browse every task" }).click();
    await expect(page).toHaveURL(/\/home$/);
    await expect(page.locator('meta[name="robots"]')).toHaveCount(0);
  });
});
