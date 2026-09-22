import { expect, test } from "../fixtures/base";
import { ModelPageObject } from "../pages/ModelPage";

// The mocked half of `/document-question-answering`: routing, the four-slot
// shell, the gate, and the claims the page makes in copy. No weights here — the
// real model is `webgpu/docvqa.spec.ts`… except it is not WebGPU-only, so the
// slow half lives in `docvqa-models.spec.ts` and runs on either backend.

test.describe("/document-question-answering", () => {
  test("renders the four slots with an empty OUTPUT", async ({ page, mockApi }) => {
    await mockApi();
    const model = new ModelPageObject(page);
    await model.blockModelDownloads();

    await page.goto("/document-question-answering");
    await expect(
      page.getByRole("heading", { name: "Document Question Answering" }),
    ).toBeVisible();

    await expect(model.slots).toHaveCount(4);
    await expect(model.emptyOutput).toBeVisible();
  });

  test("downloads nothing until Load model is clicked", async ({ page, mockApi }) => {
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

    await page.goto("/document-question-answering");
    await expect(
      page.getByRole("heading", { name: "Document Question Answering" }),
    ).toBeVisible();

    // The cost is quoted before anything is fetched — 219 MB on the CPU path is
    // well past LARGE_MODEL_BYTES.
    await expect(model.sizeNote).toBeVisible();
    await expect(model.largeModelWarning).toBeVisible();
    expect(hubRequests, "no weights before the click").toEqual([]);

    await expect(page.getByTestId("question-input")).toBeVisible();
    await expect(model.button(/^Generate$/)).toBeDisabled();

    await model.load();
    await expect(model.error).toBeVisible({ timeout: 30_000 });
    expect(hubRequests.length, "load starts the download").toBeGreaterThan(0);
  });

  test("states the privacy argument and the resolution exception", async ({
    page,
    mockApi,
  }) => {
    // Both are correctness requirements rather than decoration, the
    // /video-classification precedent: the first is why the architecture was
    // chosen, the second explains an inconsistency with every sibling route.
    await mockApi();
    const model = new ModelPageObject(page);
    await model.blockModelDownloads();

    await page.goto("/document-question-answering");
    await expect(page.getByTestId("privacy-note")).toContainText(
      /never leaves your device/i,
    );
    await expect(page.getByTestId("resolution-note")).toContainText(
      /pads rather than enlarges/i,
    );
  });

  test("choosing a document and typing a question download nothing", async ({
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

    await page.goto("/document-question-answering");
    await page.getByTestId("question-input").fill("What is the total?");
    await page
      .getByTestId("presets")
      .getByRole("button", { name: /what is the date/i })
      .click();

    expect(hubRequests, "INPUT never spends").toEqual([]);
    await expect(model.emptyOutput).toBeVisible();
  });
});
