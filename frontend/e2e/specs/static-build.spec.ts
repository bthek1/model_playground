import { expect, test } from "../fixtures/base";

// The static build (#57): `VITE_BACKEND=off`, deployed to S3 + CloudFront with
// no /api origin. There, a request to /api does not fail — the CloudFront
// function rewrites it to index.html and answers 200 — so "nothing broke" is
// not evidence. The assertion is on the wire: no request to /api at all.
//
// Runs only in the `E2E_STATIC=1` pass, against the built bundle (see
// playwright.config.ts); the dev-server pass is a backend-on build.
test.skip(!process.env.E2E_STATIC, "static-build pass only (E2E_STATIC=1)");

test.describe("static build (VITE_BACKEND=off)", () => {
  test("no page asks /api anything — even with a stale token", async ({
    page,
    signedIn,
  }) => {
    // A token left by a backend-on build is exactly what would wake useMe.
    await signedIn();
    const api: string[] = [];
    page.on("request", (req) => {
      const path = new URL(req.url()).pathname;
      if (path.startsWith("/api/")) api.push(path);
    });

    for (const path of ["/home", "/playground", "/text-classification"]) {
      await page.goto(path);
      await expect(page.getByRole("heading").first()).toBeVisible();
    }
    // Let any query the navbar would have fired get the chance to.
    await page.waitForTimeout(500);
    expect(api).toEqual([]);
  });

  test("/ goes straight to /home", async ({ page }) => {
    await page.goto("/");
    await expect(page).toHaveURL(/\/home$/);
    await expect(
      page.getByRole("heading", { name: "Model Playground", level: 1 }),
    ).toBeVisible();
  });

  for (const path of ["/login", "/signup"]) {
    test(`${path} lands on /home`, async ({ page }) => {
      await page.goto(path);
      await expect(page).toHaveURL(/\/home$/);
    });
  }

  test("has no registry catalogue and no sign-out", async ({ page }) => {
    await page.goto("/home");
    await expect(
      page.getByRole("heading", { name: "Model Playground", level: 1 }),
    ).toBeVisible();
    await expect(page.getByText("Model Catalog")).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Sign out" })).toHaveCount(0);
  });

  test("a deep link survives a reload", async ({ page }) => {
    await page.goto("/asr");
    await page.reload();
    await expect(page).toHaveURL(/\/asr$/);
    await expect(page.getByRole("heading").first()).toBeVisible();
  });
});
