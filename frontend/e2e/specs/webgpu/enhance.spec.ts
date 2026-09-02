import { MIN_SDR_GAIN_DB, measureEnhancement } from "../../utils/enhance";
import { expect, test } from "../../fixtures/base";

// The WebGPU half of the speech-enhancement check. `loadDeepFilterNet` asks ORT
// for `["webgpu", "wasm"]` in that order, so on a GPU machine this is the path
// users actually get — and it has to produce the same audio as the WASM one,
// not merely run. Skips wholesale without a real GPU device, like every spec in
// this folder.
//
// @slow: downloads the real 8.6 MB graph. Run with E2E_SLOW=1.
test.describe("@slow WebGPU speech enhancement", () => {
  test.describe.configure({ timeout: 10 * 60 * 1000 });

  test("enhances a noisy clip on the GPU", async ({
    page,
    mockApi,
    webgpuStatus,
  }) => {
    test.skip(
      webgpuStatus !== "ready",
      `No GPU device in this browser (status: ${webgpuStatus}).`,
    );
    await mockApi();
    await page.goto("/audio-to-audio");

    const { before, after, samples } = await measureEnhancement(page, "webgpu");

    expect(samples).toBeGreaterThan(0);
    expect(before).toBeLessThan(1);
    expect(after - before).toBeGreaterThan(MIN_SDR_GAIN_DB);
  });
});
