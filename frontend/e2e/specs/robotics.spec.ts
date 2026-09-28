import type { Page } from "@playwright/test";

import { ModelPageObject } from "../pages/ModelPage";
import { expect, test } from "../fixtures/base";

// /robotics — grounding an instruction: OWLv2 boxes a phrase, Depth Anything
// says which match is nearest. Two checkpoints the app already ships, loaded as
// one pair in two workers.
//
// Mocked (no weights): the four bands, the limitation note, the gating, and
// zero Hub requests before LOAD. The real load lives in
// `webgpu/robotics.spec.ts` (`just fe-e2e-robotics`), because the pair has no
// working CPU path — see that file's header.

/** A 1x1 PNG — enough for the browser to decode into a real RawImage. */
const PNG = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
  "base64",
);

/** Serve sample images; record (and refuse) every other Hub request. */
async function stubHub(page: Page) {
  const weightRequests: string[] = [];
  await page.route(
    (url) => url.hostname.endsWith("huggingface.co"),
    (route) => {
      const url = route.request().url();
      if (url.includes("/datasets/")) {
        return route.fulfill({ contentType: "image/png", body: PNG });
      }
      weightRequests.push(url);
      return route.abort();
    },
  );
  return weightRequests;
}

test.describe("/robotics", () => {
  test("renders four bands and says it is grounding, not control, before any run", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    await stubHub(page);
    const model = new ModelPageObject(page);
    await page.goto("/robotics");

    await expect(
      page.getByRole("heading", { name: "Robotics", level: 1 }),
    ).toBeVisible();
    await expect(model.slots).toHaveCount(4);
    await expect(model.emptyOutput).toBeVisible();
    await expect(page.getByTestId("grounding-limitation")).toContainText(
      /grounding, not control/i,
    );
  });

  test("quotes the pair's combined size once, with the large-download warning", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    await stubHub(page);
    const model = new ModelPageObject(page);
    await page.goto("/robotics");

    await expect(model.sizeNote).toContainText(/341 MB on WebGPU/);
    await expect(model.largeModelWarning).toBeVisible();
    await expect(model.slot(2)).not.toContainText(/MB/);
  });

  test("gates Locate on the pair while the image sources stay open, and downloads nothing", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const weightRequests = await stubHub(page);
    const model = new ModelPageObject(page);
    await page.goto("/robotics");

    await expect(model.button(/^Locate$/)).toBeDisabled();
    await expect(model.button(/^Upload image$/)).toBeEnabled();

    // Picking an image is free; so is editing a phrase.
    await model.button(/^City street$/).click();
    await expect(page.getByAltText(/selected input: city street/i)).toBeVisible();
    await page.getByLabel(/^Phrases$/).fill("a red block");
    await model.button(/^Add$/).click();

    await expect(model.button(/^Locate$/)).toBeDisabled();
    await expect(model.emptyOutput).toBeVisible();
    await expect(model.button(/^Load model$/)).toBeVisible();
    expect(weightRequests, "/robotics fetched weights before LOAD").toEqual([]);
  });
});
