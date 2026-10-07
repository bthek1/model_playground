import { expect, test } from "../fixtures/base";

// The legal pages (#62), in the default mocked suite — which CI also runs
// against the shipped static build (`e2e-static`), so the deep links below are
// tested through the same `vite preview` the CloudFront rewrite stands in for.
//
// What a browser adds over the unit tests: the lazy route chunks really load,
// the footer is really reachable on a viewport-clamped task page, and a legal
// page really sends nothing — a privacy notice that phoned home to render
// would be the one page where that is unforgivable.

const PAGES = [
  { path: "/privacy", heading: "Privacy" },
  { path: "/terms", heading: "Terms of use" },
  { path: "/licences", heading: "Licences" },
  { path: "/accessibility", heading: "Accessibility" },
] as const;

test.describe("Legal pages", () => {
  test.beforeEach(async ({ mockApi }) => {
    await mockApi();
  });

  for (const { path, heading } of PAGES) {
    test(`${path} deep-links, and requests nothing off its own origin`, async ({ page, baseURL }) => {
      const origin = new URL(baseURL!).origin;
      const offOrigin: string[] = [];
      page.on("request", (req) => {
        const url = new URL(req.url());
        if (url.protocol.startsWith("http") && url.origin !== origin) offOrigin.push(req.url());
      });
      await page.goto(path);
      await expect(page.getByRole("heading", { level: 1, name: heading })).toBeVisible();
      expect(offOrigin).toEqual([]);
    });
  }

  test("the footer is on screen on a task page, and reaches each page", async ({ page }) => {
    await page.goto("/text-classification");
    const footer = page.getByTestId("legal-footer");
    await expect(footer).toBeInViewport();
    for (const { heading } of PAGES) {
      const name = heading === "Terms of use" ? "Terms" : heading;
      await footer.getByRole("link", { name, exact: true }).click();
      await expect(page.getByRole("heading", { level: 1, name: heading })).toBeVisible();
    }
  });

  test("a picker states the licence before anything loads", async ({ page }) => {
    await page.goto("/background-removal");
    await expect(page.getByTestId("model-licence")).toContainText("Apache-2.0");
    await expect(page.getByTestId("model-licence-badge-briaai/RMBG-1.4")).toHaveText("Non-commercial");
  });

  test("the built site serves its third-party notices", async ({ page, request }) => {
    // Only the build writes the file; the dev server has nothing to serve.
    test.skip(!process.env.E2E_STATIC, "THIRD-PARTY-NOTICES.txt exists only in a build");
    await page.goto("/licences");
    const href = await page.getByTestId("third-party-notices").getAttribute("href");
    const res = await request.get(href!);
    expect(res.ok()).toBe(true);
    const text = await res.text();
    expect(text).toContain("THIRD-PARTY SOFTWARE NOTICES");
    expect(text).toMatch(/\nonnxruntime-web@/);
  });
});
