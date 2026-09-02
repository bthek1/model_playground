import { mockModels } from "../../src/test/fixtures/models";
import { expect, test } from "../fixtures/base";

// Runs fully mocked — no Django, no Postgres.
test.describe("model catalog", () => {
  test("lists models from the registry", async ({ page, mockApi }) => {
    await mockApi();
    await page.goto("/home");

    const card = page.getByText("Model Catalog").locator("..").locator("..");
    for (const model of mockModels) {
      await expect(card.getByText(model.name)).toBeVisible();
    }
    await expect(card.getByText(mockModels[0].description)).toBeVisible();
  });

  test("shows the empty state when the registry has no models", async ({
    page,
    mockApi,
  }) => {
    await mockApi({ models: [] });
    await page.goto("/home");

    await expect(page.getByText(/No models yet/)).toBeVisible();
  });

  test("shows an error state when the registry call fails", async ({
    page,
    mockApi,
  }) => {
    await mockApi({ modelsError: true });
    await page.goto("/home");

    await expect(page.getByText("Could not load the catalog.")).toBeVisible();
  });
});
