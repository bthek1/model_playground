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

// #54 — the behaviour-cloning entry. Not @slow: it downloads nothing and trains
// in about half a second a run, so the real training is part of the default
// suite. The assertion is the failure **and** its control, from one page, at
// one seed: the failure alone is not evidence (`just fe-e2e-cloning`).
test.describe("/robotics — behaviour cloning", () => {
  test("selecting it downloads nothing, and the demonstrations draw without training", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const hub = await stubHub(page);
    await page.goto("/robotics");
    await page.getByRole("button", { name: /behaviour cloning/i }).click();
    for (const step of [1, 2, 3, 4]) await expect(page.getByTestId(`slot-${step}`)).toBeVisible();
    await expect(page.getByTestId("model-ready")).toContainText(/nothing to download/i);
    await expect(page.getByTestId("demo-preview")).toBeVisible();
    await expect(page.getByTestId("demo-summary")).toContainText("10 above");
    await page.getByRole("button", { name: /one way round/i }).click();
    await expect(page.getByTestId("demo-summary")).toContainText("20 above");
    await expect(page.getByTestId("output-empty")).toBeVisible();
    await expect(page.getByTestId("rl-progress")).toHaveCount(0);
    expect(hub).toEqual([]);
  });

  test("both ways round collides; one way round, same seed, reaches", async ({ page, mockApi }) => {
    await mockApi();
    await page.goto("/robotics");
    await page.getByRole("button", { name: /behaviour cloning/i }).click();

    const train = async () => {
      await page.getByTestId("train-button").click();
      await expect(page.getByTestId("train-button")).toBeDisabled();
      await expect(page.getByTestId("train-button")).toBeEnabled({ timeout: 60_000 });
    };
    await train(); // both ways round, the default
    await expect(page.getByTestId("cloning-verdict")).toHaveAttribute("data-outcome", "collided");
    await page.getByRole("button", { name: /one way round/i }).click();
    await train();
    await expect(page.getByTestId("cloning-verdict")).toHaveAttribute("data-outcome", "reached");
    // Side by side, from one page.
    await expect(page.getByTestId("verdict-both")).toHaveAttribute("data-outcome", "collided");
    await expect(page.getByTestId("verdict-one")).toHaveAttribute("data-outcome", "reached");
  });
});
