import { expect, test } from "../fixtures/base";
import { ModelPageObject } from "../pages/ModelPage";

// The Natural Language Processing category, mocked: no weights, no Hub. This
// asserts the page's *contract* — the four slots, the idle default, and the
// rule that only GENERATE spends. Whether the model is any good is the `@slow`
// suite's job (`just fe-e2e-text`), because the unit suite mocks the runtime
// away and a mocked E2E run never loads a byte.

test.describe("/text-classification", () => {
  test("renders four slots and downloads nothing on arrival @smoke", async ({
    page,
  }) => {
    const model = new ModelPageObject(page);
    let hubRequests = 0;
    await page.route(
      (url) => url.hostname.endsWith("huggingface.co"),
      (route) => {
        hubRequests += 1;
        return route.abort();
      },
    );

    await page.goto("/text-classification");
    await expect(
      page.getByRole("heading", { name: /text classification/i }),
    ).toBeVisible();

    await expect(model.slots).toHaveCount(4);
    await expect(model.emptyOutput).toBeVisible();
    await expect(model.loadButton).toBeVisible();

    // The guardrail is quoted once, by the picker, before anything is spent.
    await expect(model.sizeNote).toContainText(/MB/);

    // Arriving at a page is not asking for 128 MB of weights.
    expect(hubRequests, "Hub requests before the LOAD click").toBe(0);
  });

  test("typing and picking a sample spend nothing", async ({ page }) => {
    const model = new ModelPageObject(page);
    await model.blockModelDownloads();
    await page.goto("/text-classification");

    const box = page.locator("#tc-text");
    await box.fill("a genuinely wonderful afternoon");
    await page.getByRole("button", { name: /a sarcastic post/i }).click();

    // The sample filled the box and stopped there — no inference, no result.
    await expect(box).not.toHaveValue("a genuinely wonderful afternoon");
    await expect(model.emptyOutput).toBeVisible();
    // And the trigger is still gated on a model, while the box never was.
    await expect(model.button(/^Classify$/)).toBeDisabled();
    await expect(box).toBeEnabled();
  });

  test("offers the head-to-head as a second load, not a toggle", async ({
    page,
  }) => {
    const model = new ModelPageObject(page);
    await model.blockModelDownloads();
    await page.goto("/text-classification");

    await page.getByRole("button", { name: /^FinBERT$/ }).last().click();
    // The cost is on screen before the button that spends it.
    await expect(page.getByTestId("compare-cost")).toContainText(/MB/);
    await expect(page.getByTestId("compare-load")).toBeVisible();
  });

  test("is reachable from the sidebar", async ({ page }) => {
    await page.goto("/");
    await page
      .getByRole("button", { name: /natural language processing/i })
      .click();
    await page.getByRole("link", { name: /^Text Classification$/ }).click();
    await expect(page).toHaveURL(/\/text-classification$/);
  });
});
