import { expect, test } from "../fixtures/base";
import type { ParityReport } from "../fixtures/clipParity";
import { ZERO_SHOT_MODELS, scoringSpec } from "../../src/vision/zeroShot";

// @slow — the spec that justifies `src/vision/zeroshot/`.
//
// `/zero-shot-image-classification` does not use the
// `zero-shot-image-classification` pipeline. It drives the text and vision
// towers separately so the label embeddings can be encoded once and reused
// across frames (docs/roadmaps/vision.md §5). The price is that the last three
// steps of the model — normalise, scale, softmax — are ours, in
// `vision/zeroshot/scoring.ts`.
//
// **That arithmetic fails silently.** A wrong `logit_scale` leaves every score
// in [0, 1], leaves the ranking exactly as it was, and is simply not what the
// model said — the same shape of bug that shipped twice past a green suite in
// `audio/enhance/`. No assertion about ordering can catch it. So this runs the
// real pipeline and our split-tower path over the same checkpoint, the same
// image and the same prompts, in one browser, and compares the numbers.
//
// The harness is `e2e/fixtures/clipParity.ts`, loaded into the page through Vite
// so it imports the **shipped** scoring module. A copy of the arithmetic inside
// `page.evaluate` would only prove that two copies of the same idea agree.
//
// It runs at **fp32**, which costs about 1.2 GB of weights and is the reason
// this spec is opt-in. At q8 the comparison is meaningless: `model.onnx` and
// the two tower files are separately quantized exports, so their embeddings
// differ and a scale-100 softmax magnifies that into a ~0.07 probability gap —
// measured, and it is the quantizer, not the arithmetic. At fp32 the two paths
// agree to six decimal places.

const BUDGET_MS = 8 * 60 * 1000;

test.describe("@slow zero-shot scoring parity", () => {
  test.describe.configure({ mode: "serial", timeout: BUDGET_MS });

  test("the split towers reproduce the full CLIP graph's scores", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    await page.goto("/zero-shot-image-classification");

    // The specifier is a variable, not a literal: it is a URL the Vite dev
    // server resolves at runtime, and TypeScript would otherwise try to resolve
    // it as a module path from this file and fail the build.
    const report = (await page.evaluate(async (harness: string) => {
      const mod = (await import(/* @vite-ignore */ harness)) as {
        compareClipScoring: () => Promise<unknown>;
      };
      return mod.compareClipScoring();
    }, "/e2e/fixtures/clipParity.ts")) as ParityReport;

    expect(
      report.reference.every((s) => s >= 0),
      "the pipeline did not return a score for every prompt",
    ).toBe(true);

    // 1. Every label agrees, not just the winner. A scale error is a systematic
    //    spread across the whole distribution, and the losing rows show it best.
    //    Measured agreement at fp32 is ~5e-7; 0.005 leaves room for ONNX Runtime
    //    version drift without letting a real error through.
    for (let i = 0; i < report.ours.length; i++) {
      expect(
        Math.abs(report.ours[i] - report.reference[i]),
        `"${report.prompts[i]}": split towers ${report.ours[i]} vs pipeline ${report.reference[i]}`,
      ).toBeLessThan(0.005);
    }

    // 2. **The constant itself, derived from the reference rather than trusted.**
    //    A softmax is shift-invariant, so `ln(p_i) - ln(p_j)` equals
    //    `scale · (cos_i - cos_j)`. Feeding the pipeline's own probabilities and
    //    our cosines into that identity recovers the `logit_scale` the full
    //    graph is using — independently of what we wrote in the catalogue. This
    //    is the assertion that catches a scale of 90 where 100 was meant, which
    //    the agreement check above would let through as a small offset.
    const spec = scoringSpec(ZERO_SHOT_MODELS[0]);
    for (let i = 1; i < report.reference.length; i++) {
      const impliedScale =
        (Math.log(report.reference[i]) - Math.log(report.reference[0])) /
        (report.cosines[i] - report.cosines[0]);
      expect(
        Math.abs(impliedScale - spec.scale) / spec.scale,
        `logit_scale implied by the full graph is ${impliedScale}, catalogue says ${spec.scale}`,
      ).toBeLessThan(0.01);
    }

    // 3. …and the distribution is decisive rather than accidentally flat. Three
    //    labels at ~0.33 each would satisfy the checks above while proving
    //    nothing, and is exactly what a scale of 1 would produce.
    expect(
      Math.max(...report.ours),
      "the scores are flat — a wrong scale would look like this",
    ).toBeGreaterThan(0.5);

    // 4. The cache is real: a second image scored against the same text
    //    embeddings needs no second text pass.
    expect(report.reusedTextEmbeddings).toBe(true);
  });
});
