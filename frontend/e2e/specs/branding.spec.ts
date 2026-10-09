import { expect, test } from "../fixtures/base";

/**
 * The tab title and the favicon are the two things nobody notices in review —
 * this app shipped as "Vite + React + TS" with the Vite logo for months. These
 * assertions are cheap and they are the reason that cannot happen again.
 */
test.describe("branding", () => {
  // @api: `/` is the landing page only with a backend; the static build
  // redirects it to /home, whose title the next test covers.
  test("the tab is named for the product, not the scaffold @api", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    await page.goto("/");
    await expect(page).toHaveTitle("Model Playground");
  });

  test("the title follows the route", async ({ page, signedIn }) => {
    await signedIn();
    // Task first, so open tabs stay distinguishable; the same string the
    // build writes into the route's own index.html (#64).
    await page.goto("/home");
    await expect(page).toHaveTitle(
      "Model Playground: ML models in your browser, on your GPU",
    );

    await page.goto("/object-detection");
    await expect(page).toHaveTitle(
      "Object Detection in your browser · Model Playground",
    );

    await page.goto("/asr");
    await expect(page).toHaveTitle(
      "Automatic Speech Recognition in your browser · Model Playground",
    );
  });

  test("every declared icon actually resolves", async ({ page, mockApi }) => {
    await mockApi();
    await page.goto("/");

    const hrefs = await page
      .locator('link[rel~="icon"], link[rel="apple-touch-icon"], link[rel="manifest"]')
      .evaluateAll((links) =>
        links.map((l) => (l as HTMLLinkElement).getAttribute("href")!),
      );

    // The scaffold shipped exactly one of these and it pointed at /vite.svg.
    expect(hrefs).toContain("/favicon.svg");
    expect(hrefs).not.toContain("/vite.svg");

    for (const href of hrefs) {
      const response = await page.request.get(href);
      expect(response.status(), `${href} should be served`).toBe(200);
    }
  });

  test("the manifest names the app and its icons resolve", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    await page.goto("/");

    const manifest = await (await page.request.get("/site.webmanifest")).json();
    expect(manifest.name).toBe("Model Playground");
    expect(manifest.icons.length).toBeGreaterThan(0);

    for (const icon of manifest.icons) {
      const response = await page.request.get(icon.src);
      expect(response.status(), `${icon.src} should be served`).toBe(200);
    }
  });

  test("the social card and description are present", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    await page.goto("/");

    await expect(page.locator('meta[name="description"]')).toHaveAttribute(
      "content",
      /browser/i,
    );
    // The card's title follows the page's (#64): on the static build `/` is
    // `/home`, on the dev server it is the landing page.
    await expect(page.locator('meta[property="og:title"]')).toHaveAttribute(
      "content",
      await page.title(),
    );
    // Absolute, as the OG spec requires — against the public origin, so ask
    // this server for the same path rather than the live site.
    const ogImage = await page
      .locator('meta[property="og:image"]')
      .getAttribute("content");
    expect(ogImage).toMatch(/^https:\/\/[^/]+\/og-image\.png$/);
    const local = new URL(ogImage!).pathname;
    expect((await page.request.get(local)).status()).toBe(200);
  });
});
