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

// The Wave 1 routes. Each one repeats the promise that matters most on that
// page, with the network mocked: the download is the user's to start, and the
// controls that do *not* need the model work before it exists.

test.describe("/depth", () => {
  test("gates the gigabyte model behind an opt-in, and downloads nothing", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const weightRequests = await stubHub(page);

    const model = new ModelPageObject(page);
    await page.goto("/depth");

    // The default entry is quoted but not gated.
    await expect(model.sizeNote).toContainText(/MB/);
    await expect(page.getByTestId("heavy-model-notice")).toBeHidden();

    // Depth Pro is ~1 GB, so it gets a second, blunter statement — and still
    // fetches nothing until Load is pressed.
    await model.button(/Depth Pro/).click();
    await expect(page.getByTestId("heavy-model-notice")).toBeVisible();
    await expect(model.largeModelWarning).toBeVisible();
    expect(
      weightRequests,
      "selecting a gated model started its download",
    ).toEqual([]);
  });

  test("shows a picked image with the run control still gated", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const weightRequests = await stubHub(page);

    const model = new ModelPageObject(page);
    await page.goto("/depth");
    await expect(model.button(/Estimate depth/)).toBeDisabled();

    await model.button(/^Tiger$/).click();
    await expect(page.getByAltText(/selected input: tiger/i)).toBeVisible();
    await expect(model.emptyOutput).toBeVisible();
    expect(weightRequests).toEqual([]);
  });
});

test.describe("/object-detection", () => {
  test("offers the threshold before a model exists, and downloads nothing", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const weightRequests = await stubHub(page);

    const model = new ModelPageObject(page);
    await page.goto("/object-detection");

    // The threshold is a pure derivation over results, so it is a live control
    // rather than something gated on the model.
    const threshold = page.getByLabel(/confidence threshold/i);
    await expect(threshold).toBeVisible();
    await threshold.fill("0.7");
    await expect(page.getByText(/threshold: 0\.70/i)).toBeVisible();

    await expect(model.button(/^Detect$/)).toBeDisabled();
    expect(weightRequests).toEqual([]);
  });

  test("does not touch the camera until it is asked to", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    await stubHub(page);

    // If the route opened a stream on mount, this would reject before the click
    // and the button would never appear in its "off" state.
    await page.addInitScript(() => {
      const media = navigator.mediaDevices as unknown as {
        getUserMedia?: () => Promise<MediaStream>;
      };
      (window as unknown as { __camera: number }).__camera = 0;
      if (media) {
        media.getUserMedia = () => {
          (window as unknown as { __camera: number }).__camera += 1;
          return Promise.reject(new Error("no camera in CI"));
        };
      }
    });

    const model = new ModelPageObject(page);
    await page.goto("/object-detection");
    await expect(model.button(/Use camera/)).toBeVisible();

    const asked = await page.evaluate(
      () => (window as unknown as { __camera: number }).__camera,
    );
    expect(asked, "the page opened the camera on mount").toBe(0);
  });
});

test.describe("/segmentation", () => {
  test("states the class space before anything is downloaded", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const weightRequests = await stubHub(page);

    const model = new ModelPageObject(page);
    await page.goto("/segmentation");

    // A segmenter can only say what its training set contained, so the label
    // space is stated up front rather than discovered from a puzzling result.
    await expect(page.getByTestId("class-space")).toContainText(/semantic/i);
    await expect(page.getByTestId("class-space")).toContainText(/ADE20K/i);

    await model.button(/DETR panoptic/).click();
    await expect(page.getByTestId("class-space")).toContainText(/panoptic/i);

    await expect(model.button(/^Segment$/)).toBeDisabled();
    expect(weightRequests).toEqual([]);
  });
});

test.describe("/zero-shot-image-classification", () => {
  test("lets the labels and the template be written before the model loads", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const weightRequests = await stubHub(page);

    const model = new ModelPageObject(page);
    await page.goto("/zero-shot-image-classification");

    // Writing the labels is the whole input for this task, and it costs nothing.
    // Bare nouns, not phrases: the template supplies the article, so `"cat"`
    // composes into "a photo of a cat" while `"a cat"` would give "a photo of a
    // a cat". `exact`, because the preview line below the chips quotes the first
    // label back inside a longer sentence.
    await expect(page.getByText("cat", { exact: true })).toBeVisible();
    await page.getByLabel(/^Labels$/).fill("bicycle");
    await model.button(/^Add$/).click();
    await expect(page.getByText("bicycle", { exact: true })).toBeVisible();

    await model.button(/Remove dog/).click();
    await expect(page.getByText("dog", { exact: true })).toBeHidden();

    // The template is shown applied to a real label, so the experiment is
    // legible before it is run — and reads as English, which is the point of
    // keeping the labels bare.
    await expect(model.slot(3)).toContainText(/a photo of a cat/i);

    await expect(model.button(/Score labels/)).toBeDisabled();
    expect(weightRequests).toEqual([]);
  });
});
