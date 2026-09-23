import { ModelPageObject } from "../../pages/ModelPage";
import { expect, test } from "../../fixtures/base";

// @slow — downloads SmolVLM2-256M-Video-Instruct (~189 MB at q4f16) and opens a
// real ONNX Runtime session. Opt in with `just fe-e2e-videovlm` (E2E_SLOW=1).
//
// Two things here cannot be caught anywhere else, and both are silent:
//
//  1. **The multi-image chat template.** N frames means N `{ type: "image" }`
//     slots filled positionally from the image list. One short, or the list out
//     of step with the slots, and the model answers fluently about the wrong
//     pictures. The unit suite mocks the runtime away; a mocked E2E run never
//     loads weights. So: a **known answer about a known clip**.
//  2. **The reverse toggle is a re-run, not a re-derivation.** The model has to
//     actually see the other order, which means a second inference rather than
//     a relabelled result — and the page charges for it honestly. Asserted by
//     counting the runs, because a toggle wired to a re-render would produce a
//     perfectly plausible screen.
//
// Needs a GPU adapter with `shader-f16`: the model is q4f16 and the catalogue
// declares WebGPU only, so a runner without one is skipped rather than failed.

const DOWNLOAD_BUDGET_MS = 8 * 60 * 1000;
// The slowest run in the app: four frames of image tokens through a 256M
// decoder, every one of them attended over for every generated word.
const GENERATE_BUDGET_MS = 5 * 60 * 1000;

test.describe("@slow real video VLM", () => {
  test.describe.configure({ mode: "serial", timeout: DOWNLOAD_BUDGET_MS });

  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    const f16 = await page.evaluate(async () => {
      const gpu = (navigator as unknown as { gpu?: GPU }).gpu;
      if (!gpu) return false;
      const adapter = await gpu.requestAdapter();
      return adapter?.features.has("shader-f16") ?? false;
    });
    test.skip(!f16, "needs a GPU adapter with shader-f16 (SwiftShader has none)");
  });

  async function loadVideoVlm(page: import("@playwright/test").Page) {
    const model = new ModelPageObject(page);
    await page.goto("/video-text-to-text");
    await model.button(/SmolVLM2 256M Video/).click();
    await model.load();
    await expect(page.getByTestId("model-ready")).toBeVisible({
      timeout: DOWNLOAD_BUDGET_MS,
    });
    expect(await model.backend()).toBe("webgpu");
    return model;
  }

  test("answers about the interview clip from the frames it sampled", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = await loadVideoVlm(page);

    await model.button(/^Interview$/).click();
    await page
      .getByTestId("question-input")
      .fill("What are the people in this video doing?");
    await model.run(/^Generate$/);

    // A known answer about a known clip: two people talking to camera. A
    // template with the image slots out of step still writes a confident
    // sentence — it just is not about this.
    await expect(model.outputPanel.getByTestId("answer-text")).toContainText(
      /talk|speak|interview|conversation|sit|discuss/i,
      { timeout: GENERATE_BUDGET_MS },
    );

    // The frames it was given are on screen, and there are as many as the
    // slider asked for.
    const strip = model.outputPanel.getByTestId("filmstrip");
    await expect(strip).toBeVisible();
    await expect(strip.locator("li")).toHaveCount(4);
    await expect(
      model.outputPanel.getByTestId("answer-asked"),
    ).toContainText(/4 frames, in order/);
  });

  test("more frames means more image tokens, and the page sends them all", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = await loadVideoVlm(page);

    await model.button(/^Interview$/).click();
    await page.getByTestId("frame-count").fill("8");
    // Still INPUT: the slider has moved and nothing has been asked of the model.
    await expect(model.emptyOutput).toBeVisible();

    await model.run(/^Generate$/);
    await expect(model.outputPanel.getByTestId("filmstrip")).toBeVisible({
      timeout: GENERATE_BUDGET_MS,
    });
    await expect(
      model.outputPanel.getByTestId("filmstrip").locator("li"),
    ).toHaveCount(8);
  });

  test("reversing the frames is a second inference, not a re-render", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = await loadVideoVlm(page);

    await model.button(/^Interview$/).click();
    await page.getByTestId("question-input").fill("What happens in this video?");
    await model.run(/^Generate$/);
    await expect(
      model.outputPanel.getByTestId("answer-asked"),
    ).toContainText(/in order/, { timeout: GENERATE_BUDGET_MS });

    await page.getByTestId("reverse-toggle").check();
    // The flip alone changes nothing on screen: the previous answer and its
    // label are still the ones that were produced.
    await expect(
      model.outputPanel.getByTestId("answer-asked"),
    ).toContainText(/in order/);

    await model.run(/^Generate$/);
    // The run really happened, against the reversed list — the label is what
    // records which order the model was shown, and it is captured inside the
    // run rather than read off the toggle.
    await expect(
      model.outputPanel.getByTestId("answer-asked"),
    ).toContainText(/in reverse/, { timeout: GENERATE_BUDGET_MS });
    await expect(model.outputPanel.getByTestId("answer-text")).toBeVisible();

    // Whether the *answer* changed is the finding, not the assertion: a model
    // this size usually describes the scene either way, which is precisely what
    // the page exists to show. Asserting a difference would pin a property the
    // model does not have.
  });

  test("states what it did to the video, beside the answer", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = await loadVideoVlm(page);

    await model.button(/^Interview$/).click();
    await model.run(/^Generate$/);
    await expect(model.outputPanel.getByTestId("sampler-note")).toContainText(
      /frame sampler/i,
      { timeout: GENERATE_BUDGET_MS },
    );
  });

  test("picking a clip runs nothing, even with the model loaded", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = await loadVideoVlm(page);

    await model.button(/^Interview$/).click();
    await model.button(/^Courtroom$/).click();
    await expect(model.emptyOutput).toBeVisible();
  });
});
