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
});
