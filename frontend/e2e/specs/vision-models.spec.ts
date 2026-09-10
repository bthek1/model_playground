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

  // --- Wave 2 -----------------------------------------------------------------

  test("/zero-shot-object-detection finds a phrase it was given, and not one it wasn't", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = new ModelPageObject(page);
    await page.goto("/zero-shot-object-detection");

    await model.button(/^Load model$/).click();
    await expect(page.getByTestId("model-ready")).toBeVisible({
      timeout: DOWNLOAD_BUDGET_MS,
    });

    // Two cats on a sofa. The asymmetry is the assertion: an open-vocabulary
    // detector that boxes *everything* at the default threshold is as broken as
    // one that boxes nothing, and only asking it for something absent can tell
    // the two apart. A "some boxes came back" check passes for both.
    await page.getByLabel(/Remove a person/).click();
    await page.getByLabel(/Remove a car/).click();
    await page.getByLabel(/Remove a traffic light/).click();
    await page.getByLabel(/^Queries$/).fill("a cat");
    await model.button(/^Add$/).click();
    await page.getByLabel(/^Queries$/).fill("a purple giraffe");
    await model.button(/^Add$/).click();

    await model.button(/^Cats$/).click();
    await expect(page.getByTestId("detection-canvas")).toBeVisible({
      timeout: 120_000,
    });

    const groups = page.getByTestId("query-groups").locator("> li");
    await expect(groups.filter({ hasText: "a cat" })).not.toContainText(
      /nothing above the threshold/i,
    );
    await expect(groups.filter({ hasText: "a purple giraffe" })).toContainText(
      /nothing above the threshold/i,
    );
  });

  test("/image-features ranks an animal next to animals, not next to a street", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = new ModelPageObject(page);
    await page.goto("/image-features");

    // DINOv2 small is ~24 MB on WASM, so a real load plus thirteen forward
    // passes is seconds rather than minutes.
    await model.button(/^Load model$/).click();
    await expect(page.getByTestId("model-ready")).toBeVisible({
      timeout: DOWNLOAD_BUDGET_MS,
    });

    await model.button(/Embed the \d+ bundled pictures/).click();
    await expect(page.getByTestId("index-size")).toHaveText(/1[0-9] indexed/, {
      timeout: 180_000,
    });

    // The tiger is deliberately **not** in the gallery, so this is a semantic
    // assertion rather than a lookup: the nearest neighbour of an animal must
    // be an animal. A mis-pooled vector (CLS averaged in with the patches) or
    // an unnormalised one destroys exactly this ordering while still producing
    // a full, plausible-looking list — which is why "five rows appeared" is not
    // an acceptable assertion here.
    await model.button(/^Tiger$/).click();
    await expect(page.getByTestId("neighbours")).toBeVisible({ timeout: 60_000 });

    const first = page.getByTestId("neighbours").locator("li").first();
    await expect(first).toContainText(/animal/);

    // And the vectors really are unit length — the claim the page displays.
    await expect(page.getByTestId("embedding-facts")).toContainText(
      /after\s*1\.000/,
    );

    // Switching the pooling re-ranks without asking the model anything. The two
    // vectors are different, so at least one score must move.
    const scoreOf = async () =>
      Number(
        (await first.locator("span.font-mono").innerText()).trim(),
      );
    const cls = await scoreOf();
    await model.button(/Mean of patches/).click();
    await expect(page.getByTestId("embedding-facts")).toContainText(
      /Mean of patches/,
    );
    const mean = await scoreOf();
    expect(
      Math.abs(cls - mean),
      "CLS and mean pooling produced identical scores — one of them is not being used",
    ).toBeGreaterThan(0.0005);
  });

  test("/mask-generation encodes once, then cuts out a real object per click", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = new ModelPageObject(page);
    await page.goto("/mask-generation");

    // SlimSAM is ~14 MB on WASM, so this is the cheapest @slow spec in the file.
    await model.button(/^Load model$/).click();
    await expect(page.getByTestId("model-ready")).toBeVisible({
      timeout: DOWNLOAD_BUDGET_MS,
    });

    await model.button(/^Beetle \(car\)$/).click();
    // The encode state exists and completes — the half of the wait that is not
    // the download, and which the user has no way to guess is happening.
    await expect(page.getByTestId("encoded")).toBeVisible({ timeout: 120_000 });

    await model.button(/Point at the centre/).click();
    await expect(page.getByTestId("mask-canvas")).toBeVisible({
      timeout: 60_000,
    });

    // **The assertion that matters.** A point at the centre of a car on a plain
    // background must produce a mask covering a real fraction of the frame. A
    // silently wrong point-coordinate space produces exactly one of the two
    // degenerate answers — nothing, or everything — and both look like a
    // plausible mask until the coverage is measured.
    const facts = await page.getByTestId("mask-facts").innerText();
    const covered = Number(/covers\s+([\d.]+)%/.exec(facts)?.[1] ?? NaN);
    expect(
      covered,
      `mask covers ${covered}% of the frame — a degenerate mask means the click never reached the object`,
    ).toBeGreaterThan(3);
    expect(covered).toBeLessThan(95);

    // Three candidates, because a point is ambiguous by construction.
    await expect(
      model.outputPanel.getByRole("button", { name: /^3 · 0\./ }),
    ).toBeVisible();

    // And the claim the page makes about itself: the click is decode-only.
    const decode = await page.getByTestId("decode-ms").innerText();
    const ms = Number(/decode (\d+) ms/.exec(decode)?.[1] ?? NaN);
    expect(Number.isFinite(ms)).toBe(true);
  });
});

// Florence-2 is WebGPU-only in the catalogue, so its spec lives in the `webgpu`
// project rather than the default chromium one — on a machine with no GPU the
// picker correctly refuses to select it, and there is nothing to assert.
test.describe("@slow real vision model loads on WebGPU", () => {
  test.describe.configure({ mode: "serial", timeout: DOWNLOAD_BUDGET_MS });

  test("/image-to-text captions a photo and reads printed text", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = new ModelPageObject(page);
    await page.goto("/image-to-text");

    const florence = model.button(/Florence-2 base/);
    test.skip(
      !(await florence.isEnabled()),
      "no WebGPU adapter — Florence-2 is gated off",
    );

    await model.button(/^Load model$/).click();
    await expect(page.getByTestId("model-ready")).toBeVisible({
      timeout: DOWNLOAD_BUDGET_MS,
    });

    // 1. Captioning. A plausible noun, on a picture whose subject is not in
    //    doubt — "a string came back" would pass on a model generating noise.
    await model.button(/^Tiger$/).click();
    await expect(model.outputPanel).toContainText(/tiger|cat|animal/i, {
      timeout: 180_000,
    });

    // 2. OCR, and **this is the assertion that catches a broken processor
    //    path**. A model handed mis-normalised pixels still produces fluent
    //    text; only text that matches what is actually in the picture proves
    //    the image reached it intact.
    await page.getByTestId("modes").getByRole("button", { name: "OCR" }).click();
    await model.button(/^Advertisement$/).click();
    await expect(model.outputPanel).toContainText(/coca|cola/i, {
      timeout: 180_000,
    });

    // 3. Grounding renders as boxes, not as prose — the two output shapes come
    //    from one task and are routed by the mode, not by sniffing the answer.
    await page
      .getByTestId("modes")
      .getByRole("button", { name: "Grounding" })
      .click();
    await model.button(/^City street$/).click();
    await expect(page.getByTestId("grounding-canvas")).toBeVisible({
      timeout: 180_000,
    });
    await expect(page.getByTestId("grounding-list")).toBeVisible();
  });
});

test.describe("@slow real two-model pose", () => {
  test.describe.configure({ mode: "serial", timeout: DOWNLOAD_BUDGET_MS });

  test("/pose loads both models into one bar and puts the nose above the ankles", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = new ModelPageObject(page);
    await page.goto("/pose");

    await model.button(/^Load model$/).click();
    await expect(page.getByTestId("model-ready")).toBeVisible({
      timeout: DOWNLOAD_BUDGET_MS,
    });

    await model.button(/^Football match$/).click();
    await expect(page.getByTestId("pose-canvas")).toBeVisible({
      timeout: 120_000,
    });

    // **The assertion that matters: a coarse anatomical one.** Any count-based
    // check ("some skeletons appeared") passes with every joint offset by the
    // crop's origin — the skeleton floats beside the person and looks like a
    // mediocre model rather than a bug in our arithmetic. On a photo of people
    // standing up, the nose has to be above the ankles in image coordinates.
    await page.getByText(/Joint confidences for person 1/).click();
    const joints = page.getByTestId("joints");
    await expect(joints).toBeVisible();

    const yOf = async (joint: string) => {
      const text = await joints
        .locator(`li[data-joint="${joint}"]`)
        .innerText();
      const match = /(\d+),(\d+)\s*$/.exec(text.trim());
      expect(match, `no position on the ${joint} row: ${text}`).not.toBeNull();
      return Number(match![2]);
    };

    const nose = await yOf("nose");
    const ankles = [await yOf("left ankle"), await yOf("right ankle")];
    expect(
      nose,
      `nose at y=${nose} is not above the ankles at y=${ankles.join(", ")} — the crop origin was not added back`,
    ).toBeLessThan(Math.min(...ankles));

    // Both models really did run, and the page says what each cost.
    await expect(model.outputPanel).toContainText(/detect \d+ ms/);
    await expect(model.outputPanel).toContainText(/pose \d+ ms/);
  });
});

test.describe("@slow real frame-level video baseline", () => {
  test.describe.configure({ mode: "serial", timeout: DOWNLOAD_BUDGET_MS });

  test("/video-classification scores the bundled clip's true label above a distractor", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = new ModelPageObject(page);
    await page.goto("/video-classification");

    await model.button(/^Load model$/).click();
    await expect(page.getByTestId("model-ready")).toBeVisible({
      timeout: DOWNLOAD_BUDGET_MS,
    });

    // 1 fps keeps a real CLIP pass per frame down to a handful of frames.
    await page.getByLabel(/frames per second/i).fill("1");
    await model.button(/Score the clip/).click();

    await expect(page.getByTestId("pooled-winner")).toBeVisible({
      timeout: 300_000,
    });

    // The interview clip against a football match and a car chase. A pooled
    // verdict is the only honest assertion here — a single frame can go either
    // way, which is the page's whole point.
    await expect(page.getByTestId("pooled-winner")).toContainText(
      /an interview/,
    );

    // Every sampled frame is traceable, and the framing survives a real run.
    await expect(page.getByTestId("filmstrip").locator("li").first()).toBeVisible();
    await expect(page.getByTestId("baseline-note")).toContainText(
      /frame-level baseline/i,
    );

    // Widening the window re-derives the chart and asks the model nothing.
    const requestsBefore: string[] = [];
    page.on("request", (r) => {
      if (r.url().includes("huggingface.co")) requestsBefore.push(r.url());
    });
    await page.getByLabel(/pooling window/i).fill("9");
    await expect(page.getByTestId("pooled-winner")).toContainText(/window 9/);
    expect(
      requestsBefore,
      "moving the pooling window re-scored the clip",
    ).toEqual([]);
  });
});
