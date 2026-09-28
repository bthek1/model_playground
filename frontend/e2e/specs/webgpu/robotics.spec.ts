import { ModelPageObject } from "../../pages/ModelPage";
import { expect, test } from "../../fixtures/base";

// @slow — `/robotics` with a real load of both models: OWLv2 base/16 and Depth
// Anything V2 Small, one LOAD, two workers. Opt in with `just fe-e2e-robotics`
// (E2E_SLOW=1). The mocked half of this route is `../robotics.spec.ts`.
//
// **The one assertion no unit test can make: the object that is genuinely
// nearer is reported nearer.** Depth Anything emits *inverse* depth, and an
// inverted convention produces a complete, confident, correctly-formatted
// answer with the ordering backwards — every count-based check passes it. So
// this pins geometry on a known image, the same class of assertion as
// `/mask-generation`'s coverage band and `/pose`'s nose above the ankles.
//
// **It lives under `webgpu/` because the pair has no CPU path.** OWLv2's q8
// export does not open on the ONNX Runtime WASM build bundled with
// Transformers.js 4.2.0: `Could not find an implementation for Cast(13) node
// with name '/class_head/Cast'`. Measured on the `chromium` project, where
// `/zero-shot-object-detection`'s own @slow spec fails the same way. The fp16
// WebGPU path needs `shader-f16`, which SwiftShader lacks, so this skips on a
// runner with no real GPU rather than failing about a machine the page cannot
// serve.

const DOWNLOAD_BUDGET_MS = 8 * 60 * 1000;

test.describe("@slow real grounding pair", () => {
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

  test("/robotics finds the car and reports the near car nearer than the far one", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = new ModelPageObject(page);
    await page.goto("/robotics");

    await model.load();
    // Fail fast, and with the reason, if either half fails to load — a pair
    // has two ways to fail and "model-ready never appeared" names neither.
    const outcome = await Promise.race([
      page.getByTestId("model-ready").waitFor({ timeout: DOWNLOAD_BUDGET_MS }).then(() => "ready"),
      model.slot(2).getByRole("alert").waitFor({ timeout: DOWNLOAD_BUDGET_MS }).then(() => "error"),
    ]);
    if (outcome === "error") {
      await model.slot(2).getByText(/details/i).click().catch(() => {});
      throw new Error(`the pair failed to load: ${await model.slot(2).innerText()}`);
    }

    // The city street, shot from above: a blue hatchback in the right
    // foreground (roughly x 0.70–1.0, y 0.59–0.75 of the frame) and cars far
    // up the road at the top of the picture (y < 0.12). Looking down a street,
    // lower in the frame is nearer, so the geometry gives the ground truth
    // without a second model to trust.
    await page.getByLabel(/Remove a bicycle/).click();
    await model.button(/^City street$/).click();
    // Low enough to keep the small, distant cars.
    await page.getByLabel(/confidence threshold/i).fill("0.05");
    await model.run(/^Locate$/);
    await expect(page.getByTestId("grounding-canvas")).toBeVisible({
      timeout: 180_000,
    });

    const rows = page
      .getByTestId("grounding-row")
      .and(page.locator('[data-label="a car"]'));
    await expect(rows.first()).toBeVisible();
    const cars = await rows.evaluateAll((els) =>
      els.map((el) => ({
        cx: Number(el.getAttribute("data-cx")),
        cy: Number(el.getAttribute("data-cy")),
        rank: Number(el.getAttribute("data-rank") || Infinity),
      })),
    );

    // (a) A known label on a known image: a box centred on the blue car.
    const nearCar = cars.find(
      (c) => c.cx > 0.7 && c.cy > 0.55 && c.cy < 0.8,
    );
    expect(
      nearCar,
      `no "a car" box on the foreground hatchback: ${JSON.stringify(cars)}`,
    ).toBeDefined();

    // (b) The assertion that matters: the far car is reported *further*.
    const farCar = cars.find((c) => c.cy < 0.2);
    expect(
      farCar,
      `no "a car" box up the road to compare against: ${JSON.stringify(cars)}`,
    ).toBeDefined();
    expect(
      nearCar!.rank,
      `the foreground car ranked ${nearCar!.rank}, the distant one ${farCar!.rank} — the depth convention is inverted`,
    ).toBeLessThan(farCar!.rank);

    // And the page says what the numbers are not, and what each model cost.
    await expect(model.outputPanel).toContainText(/not metres/i);
    await expect(model.outputPanel).toContainText(/detect \d+ ms · depth \d+ ms/);
  });
});
