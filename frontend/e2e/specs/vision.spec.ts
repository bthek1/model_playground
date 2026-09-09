import { ModelPageObject } from "../pages/ModelPage";
import { expect, test } from "../fixtures/base";

// /image-classification — the first Computer Vision route, with the network
// mocked. What it covers that jsdom cannot: real image decoding (`RawImage` over
// a real fetch), the object-URL preview, and the promise that navigating here
// downloads no weights.
//
// The four-slot contract itself is asserted for every route in
// model-page.spec.ts; this file is the route's own behaviour.

/** A 1x1 PNG — enough for the browser to decode into a real RawImage. */
const PNG = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
  "base64",
);

/**
 * Serve the sample images, and count every other Hugging Face request. The two
 * halves matter separately: the pictures come from the Hub's datasets origin and
 * must work, while the *weights* live on the same origin and must not be
 * fetched until the user asks.
 */
async function stubHub(page: import("@playwright/test").Page) {
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

test.describe("/image-classification", () => {
  test("shows a picked image before any model exists, and downloads nothing", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const weightRequests = await stubHub(page);

    const model = new ModelPageObject(page);
    await page.goto("/image-classification");

    // The transport is gated on a model, and says why.
    await expect(model.button(/^Classify$/)).toBeDisabled();
    await expect(model.button(/^Load model$/)).toBeVisible();

    // Picking an image is free — it costs a picture, not 88 MB of weights.
    await model.button(/^Tiger$/).click();
    await expect(page.getByAltText(/selected input: tiger/i)).toBeVisible();

    // Still nothing to show, and still nothing downloaded.
    await expect(model.emptyOutput).toBeVisible();
    expect(
      weightRequests,
      "picking an image fetched model weights",
    ).toEqual([]);
  });

  test("quotes the download once, in the SELECT slot", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    await stubHub(page);

    const model = new ModelPageObject(page);
    await page.goto("/image-classification");

    // The size-before-load guardrail: quoted by the picker, and not repeated by
    // the LOAD slot (model-page-pattern.md §4).
    await expect(page.getByTestId("model-size-note")).toContainText(/MB/);
    await expect(model.slot(2)).not.toContainText(/MB/);
  });

  test("restores the selected model across a refresh without downloading", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const weightRequests = await stubHub(page);

    const model = new ModelPageObject(page);
    await page.goto("/image-classification");
    await model.button(/MobileNetV4/).click();

    await page.reload();

    await expect(model.button(/MobileNetV4/)).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    await expect(model.button(/^Load model$/)).toBeVisible();
    expect(weightRequests, "a refresh spent bandwidth").toEqual([]);
  });

  test("keeps the image preview and the results side by side on a desktop", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    await stubHub(page);
    await page.setViewportSize({ width: 1440, height: 900 });

    const model = new ModelPageObject(page);
    await page.goto("/image-classification");
    await model.button(/^Tiger$/).click();
    await expect(page.getByAltText(/selected input: tiger/i)).toBeVisible();

    // A tall image preview is the obvious way for this route to push its own
    // output below the fold. It must not.
    const input = await model.slot(3).boundingBox();
    const output = await model.slot(4).boundingBox();
    expect(input && output, "a workbench band was not laid out").toBeTruthy();
    expect(input!.x + input!.width).toBeLessThanOrEqual(output!.x + 1);
    expect(output!.y).toBeLessThan(900);
  });
});
