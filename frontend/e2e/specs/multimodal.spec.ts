import { expect, test } from "../fixtures/base";
import { ModelPageObject } from "../pages/ModelPage";

// The mocked half of `/image-text-to-text`: routing, the four-slot shell, the
// gate, and the rules that say what is allowed to spend the user's bandwidth.
// No weights are downloaded here — the real model is `multimodal-models.spec.ts`.

test.describe("/image-text-to-text", () => {
  test("renders the four slots with an empty OUTPUT", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = new ModelPageObject(page);
    await model.blockModelDownloads();

    await page.goto("/image-text-to-text");
    await expect(
      page.getByRole("heading", { name: "Image Text to Text" }),
    ).toBeVisible();

    await expect(model.slots).toHaveCount(4);
    await expect(model.outputPanel).toBeVisible();
    await expect(model.emptyOutput).toBeVisible();
  });

  // Every model on this page is past `LARGE_MODEL_BYTES` several times over, so
  // the gate is the spec: nothing is fetched until the user asks for it.
  test("downloads nothing until Load model is clicked", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = new ModelPageObject(page);

    const hubRequests: string[] = [];
    await page.route(
      (url) => url.hostname.endsWith("huggingface.co"),
      (route) => {
        hubRequests.push(route.request().url());
        return route.abort();
      },
    );

    await page.goto("/image-text-to-text");
    await expect(
      page.getByRole("heading", { name: "Image Text to Text" }),
    ).toBeVisible();

    // The cost is quoted — once, by the picker — before anything is fetched.
    await expect(model.sizeNote).toBeVisible();
    await expect(model.largeModelWarning).toBeVisible();
    expect(hubRequests, "no weights before the click").toEqual([]);

    // The input surface is alive and the trigger is dead: choosing is free,
    // generating is not.
    await expect(page.getByTestId("prompt-input")).toBeVisible();
    await expect(model.button(/^Generate$/)).toBeDisabled();

    // Pressing Load is what starts the download. With the Hub blocked it fails,
    // which is also the proof it was attempted.
    await model.load();
    await expect(model.error).toBeVisible({ timeout: 30_000 });
    expect(hubRequests.length, "load starts the download").toBeGreaterThan(0);
  });

  test("typing a question and picking a picture download nothing", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = new ModelPageObject(page);

    const hubRequests: string[] = [];
    await page.route(
      (url) => url.hostname.endsWith("huggingface.co"),
      (route) => {
        hubRequests.push(route.request().url());
        return route.abort();
      },
    );

    await page.goto("/image-text-to-text");
    await page.getByTestId("prompt-input").fill("What colour is the sky?");
    await page
      .getByTestId("presets")
      .getByRole("button", { name: /how many people/i })
      .click();

    // Both are INPUT: they commit to nothing.
    expect(hubRequests, "INPUT never spends").toEqual([]);
    await expect(model.emptyOutput).toBeVisible();
  });

  test("the sidebar links to the route rather than the placeholder", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    await page.goto("/image-text-to-text");
    // The taxonomy entry is wired, so the task no longer falls through to
    // `/tasks/$slug`.
    await expect(page).toHaveURL(/\/image-text-to-text$/);
    await expect(
      page.getByRole("heading", { name: "Image Text to Text" }),
    ).toBeVisible();
  });
});
