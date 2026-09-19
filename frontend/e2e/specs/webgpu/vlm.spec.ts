import { ModelPageObject } from "../../pages/ModelPage";
import { expect, test } from "../../fixtures/base";

// @slow — downloads real ONNX weights from Hugging Face and opens a real ONNX
// Runtime session. Excluded from the default run; opt in with
// `just fe-e2e-vlm` (E2E_SLOW=1).
//
// **This is the only test that can catch a broken chat template**, which is this
// page's characteristic failure. The unit suite mocks the worker away, and a
// mocked E2E run never loads weights, so both stay green while the model is
// being prompted with a string it has never seen — and the symptom is not an
// error. It is a fluent, confident sentence that does not answer the question.
// Hence: a **known answer on a known image**, never "some text appeared".
//
// SmolVLM-256M at q4f16 is ~189 MB, so a real load plus a real generation costs
// a couple of minutes rather than tens.
//
// **It needs a GPU with `shader-f16`, and SwiftShader is not one.** That is not a
// performance caveat — the weights load fine (29 s, `ready`, backend `webgpu`) and
// then every run fails on the first operator with "Program Gather requires f16 but
// the device does not support it". This spec is how that was found, and
// `supportsShaderF16()` is what now stops a real user paying 189 MB to discover it.
// On a runner with no `/dev/dri` the sensible outcome is to skip rather than fail:
// there is no f16 to test with.

const DOWNLOAD_BUDGET_MS = 8 * 60 * 1000;
const GENERATE_BUDGET_MS = 3 * 60 * 1000;

test.describe("@slow real VLM loads", () => {
  test.describe.configure({ mode: "serial", timeout: DOWNLOAD_BUDGET_MS });

  // Skip rather than fail where there is no half-precision to test with. A
  // failure here would say "the page is broken" about a machine the page
  // correctly refuses to run on.
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

  async function loadSmallest(page: import("@playwright/test").Page) {
    const model = new ModelPageObject(page);
    await page.goto("/image-text-to-text");
    await model.button(/SmolVLM 256M/).click();
    await model.load();
    await expect(page.getByTestId("model-ready")).toBeVisible({
      timeout: DOWNLOAD_BUDGET_MS,
    });
    return model;
  }

  test("answers a question about the tiger sample with the right animal", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = await loadSmallest(page);

    // Every model here is WebGPU-only: if the runner resolved to WASM the
    // picker would have disabled the row and the load above would never have
    // completed. Assert it rather than assume it.
    expect(await model.backend()).toBe("webgpu");

    await model.button(/^Tiger$/).click();
    await page.getByTestId("prompt-input").fill("What animal is in this photo?");
    await model.run(/^Generate$/);

    // The assertion that pins the chat template. A hand-built prompt, a missing
    // image slot or a missing generation prompt all produce fluent text here —
    // and none of them produce the word "tiger", because the model never
    // actually looked at the picture in the way the question implies.
    await expect(model.outputPanel.getByTestId("answer-text")).toContainText(
      /tiger|cat|feline/i,
      { timeout: GENERATE_BUDGET_MS },
    );
  });

  test("the answer changes with the question, not just with the picture", async ({
    page,
    mockApi,
  }) => {
    // The sharpest available check that the prompt reaches the model at all. If
    // the question were being dropped, both runs would describe the same
    // picture the same way.
    await mockApi();
    const model = await loadSmallest(page);

    await model.button(/^Tiger$/).click();
    await page.getByTestId("prompt-input").fill("What animal is in this photo?");
    await model.run(/^Generate$/);
    await expect(model.outputPanel.getByTestId("answer-text")).toBeVisible({
      timeout: GENERATE_BUDGET_MS,
    });
    const animal = await model.outputPanel.getByTestId("answer-text").innerText();

    await page
      .getByTestId("prompt-input")
      .fill("Answer with a single word: what colour dominates this photo?");
    await model.run(/^Generate$/);
    await expect
      .poll(
        async () =>
          model.outputPanel.getByTestId("answer-text").innerText(),
        { timeout: GENERATE_BUDGET_MS },
      )
      .not.toBe(animal);
  });

  test("labels the encode separately from the generation", async ({
    page,
    mockApi,
  }) => {
    // The pre-token pause is seconds on a real load. The page must name it —
    // an unlabelled pause is indistinguishable from a hang.
    await mockApi();
    const model = await loadSmallest(page);

    await model.button(/^Tiger$/).click();
    await page.getByTestId("prompt-input").fill("Describe this picture.");
    await model.run(/^Generate$/);

    // The encode state appears first. It is transient, so this races the model
    // on a fast GPU — the reported `encodeMs` below is the durable proof.
    await expect(model.outputPanel.getByTestId("answer-text")).toBeVisible({
      timeout: GENERATE_BUDGET_MS,
    });
    await expect(model.outputPanel).toContainText(/s encoding/);
    await expect(model.outputPanel).toContainText(/\d+ tokens/);
  });

  test("picking a sample runs nothing, even with the model loaded", async ({
    page,
    mockApi,
  }) => {
    // The §1.6 rule, asserted against a real loaded model rather than a mock:
    // browsing five samples must not cost five generations.
    await mockApi();
    const model = await loadSmallest(page);

    await model.button(/^Tiger$/).click();
    await model.button(/^Street$/).click();
    await expect(model.emptyOutput).toBeVisible();
  });
});
