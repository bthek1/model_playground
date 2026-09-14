import { expect, test } from "../fixtures/base";

/**
 * The tab title and the favicon are the two things nobody notices in review —
 * this app shipped as "Vite + React + TS" with the Vite logo for months. These
 * assertions are cheap and they are the reason that cannot happen again.
 */
test.describe("branding", () => {
  test("the tab is named for the product, not the scaffold", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    await page.goto("/");
    await expect(page).toHaveTitle("Model Playground");
  });

  test("the title follows the route", async ({ page, signedIn }) => {
    await signedIn();
    await page.goto("/home");
    await expect(page).toHaveTitle("Home · Model Playground");

    await page.goto("/object-detection");
    await expect(page).toHaveTitle("Object Detection · Model Playground");

    await page.goto("/asr");
    await expect(page).toHaveTitle(
      "Automatic Speech Recognition · Model Playground",
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
    await expect(page.locator('meta[property="og:title"]')).toHaveAttribute(
      "content",
      "Model Playground",
    );
    const ogImage = await page
      .locator('meta[property="og:image"]')
      .getAttribute("content");
    expect((await page.request.get(ogImage!)).status()).toBe(200);
  });
});
