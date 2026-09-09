import { ModelPageObject } from "../pages/ModelPage";
import { expect, test } from "../fixtures/base";

// @slow — downloads real ONNX weights from Hugging Face and opens a real ONNX
// Runtime session. Excluded from the default run; opt in with
// `just fe-e2e-vision` (E2E_SLOW=1).
//
// The unit suite mocks the network and the runtime away, so it cannot tell a
// working checkpoint from a repo that 404s at load time, and it cannot tell a
// correct preprocessing path from one that silently feeds the model noise. This
// is the spec that can: MobileNetV4 Small is ~4 MB on WASM, so a real load plus
// a real classification costs seconds rather than minutes.

const DOWNLOAD_BUDGET_MS = 5 * 60 * 1000;

test.describe("@slow real vision model loads", () => {
  test.describe.configure({ mode: "serial", timeout: DOWNLOAD_BUDGET_MS });

  test("/image-classification loads MobileNetV4 and labels the tiger sample", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = new ModelPageObject(page);
    await page.goto("/image-classification");

    // The cheapest entry in the catalogue, chosen so this spec stays fast
    // enough to actually be run.
    await model.button(/MobileNetV4/).click();
    await model.button(/^Load model$/).click();
    await expect(page.getByTestId("model-ready")).toBeVisible({
      timeout: DOWNLOAD_BUDGET_MS,
    });

    await model.button(/^Tiger$/).click();

    // A model that loads and returns garbage is still broken, so assert the
    // label — not merely that five rows appeared. `tiger` and `tiger cat` are
    // both acceptable; anything else means the image never reached the model in
    // the form its processor expects.
    await expect(
      model.outputPanel.getByText(/tiger/i).first(),
    ).toBeVisible({ timeout: 60_000 });
  });

  test("/image-classification shows the runner-up, not just the winner", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = new ModelPageObject(page);
    await page.goto("/image-classification");

    await model.button(/MobileNetV4/).click();
    await model.button(/^Load model$/).click();
    await expect(page.getByTestId("model-ready")).toBeVisible({
      timeout: DOWNLOAD_BUDGET_MS,
    });

    await model.button(/^Cats$/).click();
    // Five rows, each with a percentage: the near-tie this page exists to make
    // visible is only visible if all five are there.
    const scores = model.outputPanel.locator("li");
    await expect(scores).toHaveCount(5, { timeout: 60_000 });
    await expect(model.outputPanel.getByText(/top-2 margin/)).toBeVisible();
  });

  // --- Wave 1 -----------------------------------------------------------------
  //
  // Each of these asserts a *known answer on a known input*, not "something
  // appeared". The unit suite mocks the runtime away, so it cannot tell a
  // correct preprocessing path from one that quietly feeds the model noise —
  // and the quantized MobileNetV4 that called a tiger a rattlesnake would have
  // passed a "five rows rendered" assertion without complaint.

  test("/depth loads Depth Anything V2 and separates near from far", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = new ModelPageObject(page);
    await page.goto("/depth");

    await model.button(/^Load model$/).click();
    await expect(page.getByTestId("model-ready")).toBeVisible({
      timeout: DOWNLOAD_BUDGET_MS,
    });

    // The tiger fills the foreground against a distant background, so a working
    // relative-depth model must produce a map with a real spread. A constant
    // map — which is what a broken normalisation or a transposed read produces
    // — would render as a flat colour and fail here.
    await model.button(/^Tiger$/).click();
    await expect(page.getByTestId("depth-map")).toBeVisible({ timeout: 60_000 });

    const spread = await page
      .getByTestId("slot-4")
      .getByText(/\d+\.\d+ … \d+\.\d+/)
      .innerText();
    const [lo, hi] = spread.split("…").map((n) => Number(n.trim()));
    expect(Number.isFinite(lo) && Number.isFinite(hi)).toBe(true);
    expect(hi - lo, "the depth map is flat — nothing was estimated").toBeGreaterThan(1);

    // And the page says what the numbers are not.
    await expect(model.outputPanel).toContainText(/arbitrary scale, not metres/i);
  });

  test("/object-detection loads D-FINE nano and puts boxes in the right places", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = new ModelPageObject(page);
    await page.goto("/object-detection");

    await model.button(/^Load model$/).click();
    await expect(page.getByTestId("model-ready")).toBeVisible({
      timeout: DOWNLOAD_BUDGET_MS,
    });

    // Two cats on a sofa: a working detector finds cats, and finds more than
    // one of them. A "some boxes were returned" assertion would pass even with
    // `percentage` set the wrong way round, which is the bug this route is most
    // likely to grow.
    await model.button(/^Cats$/).click();
    await expect(page.getByTestId("detection-canvas")).toBeVisible({
      timeout: 60_000,
    });
    await expect(model.outputPanel.getByText(/^cat$/i).first()).toBeVisible();

    // Dropping the threshold can only ever reveal more, never fewer — and it
    // must do it without touching the model.
    const shown = () => model.outputPanel.locator("li").count();
    const strict = await shown();
    await page.getByLabel(/confidence threshold/i).fill("0.05");
    expect(await shown()).toBeGreaterThanOrEqual(strict);
  });

  test("/segmentation loads SegFormer-B0 and names plausible ADE classes", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = new ModelPageObject(page);
    await page.goto("/segmentation");

    await model.button(/^Load model$/).click();
    await expect(page.getByTestId("model-ready")).toBeVisible({
      timeout: DOWNLOAD_BUDGET_MS,
    });

    // A street scene through a 150-class ADE model must come back with street
    // furniture, not with kitchenware. This is also what proves the class index
    // to label mapping was not read off by one — an off-by-one produces a
    // perfectly plausible-looking mask set with the wrong names on it.
    await model.button(/^City street$/).click();
    await expect(page.getByTestId("segmentation-canvas")).toBeVisible({
      timeout: 60_000,
    });
    await expect(
      model.outputPanel.getByRole("button", {
        name: /sky|building|road|tree|car|sidewalk|person/i,
      }).first(),
    ).toBeVisible();
  });

  test("/zero-shot-image-classification picks the cat, and the wording moves the score", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = new ModelPageObject(page);
    await page.goto("/zero-shot-image-classification");

    await model.button(/^Load model$/).click();
    await expect(page.getByTestId("model-ready")).toBeVisible({
      timeout: DOWNLOAD_BUDGET_MS,
    });

    await model.button(/^Cats$/).click();
    await expect(page.getByTestId("template-verdict")).toBeVisible({
      timeout: 90_000,
    });

    // Two assertions. The first is that the model works; the second is what the
    // page can honestly claim about prompt templates.
    const row = (n: number) =>
      model.outputPanel.locator("tbody tr").nth(n).locator("td");
    const cat = row(0);
    const dog = row(1);
    const bare = Number(await cat.nth(1).innerText());
    const templated = Number(await cat.nth(2).innerText());

    // 1. The picture is of cats, under *both* wordings. A prompt experiment run
    //    on a model that cannot tell a cat from a dog would be measuring noise.
    expect(templated, "CLIP scored a dog over a cat, templated").toBeGreaterThan(
      Number(await dog.nth(2).innerText()),
    );
    expect(bare, "CLIP scored a dog over a cat, bare").toBeGreaterThan(
      Number(await dog.nth(1).innerText()),
    );

    // 2. **The wording moves the numbers.** That — not "the template always
    //    wins" — is what this page demonstrates, and all it may assert. The
    //    template's advantage is an average over a benchmark and does not hold
    //    image by image: on this very sample CLIP gives the bare label 0.860 and
    //    "a photo of a cat" 0.842. Asserting the templated score is always the
    //    higher one would encode a claim the evidence does not support, and
    //    would fail on a perfectly good build.
    expect(
      Math.abs(templated - bare),
      "the two templates produced identical scores — the experiment showed nothing",
    ).toBeGreaterThan(0.0005);
  });
});
