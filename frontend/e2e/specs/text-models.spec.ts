import { expect, test } from "../fixtures/base";
import { ModelPageObject } from "../pages/ModelPage";

// @slow — real Hugging Face downloads and real ONNX Runtime sessions.
// `just fe-e2e-text`.
//
// **These assert a known label on a known sentence**, never "a result
// appeared". The precedent is the vision suite, where a green count-based
// assertion sat happily on top of a quantized MobileNet calling a tiger a
// snake; the same assertion here would pass while a broken tokenizer scored
// noise. The unit suite cannot catch either — it mocks the runtime away — and
// the mocked spec above never loads a byte.

const NEGATIVE_SENTENCE =
  "The film was a triumph of tedium — two hours I will never get back.";

test.describe("@slow text models", () => {
  test.slow();

  test("DistilBERT SST-2 calls a bad review negative", async ({ page }) => {
    test.setTimeout(6 * 60 * 1000);
    const model = new ModelPageObject(page);

    await page.goto("/text-classification");
    await model.load();
    await model.waitForReady();

    await page.locator("#tc-text").fill(NEGATIVE_SENTENCE);
    await model.run(/^Classify$/);

    const scores = page.getByTestId("answer-primary-scores");
    await expect(scores).toBeVisible({ timeout: 120_000 });

    // The label, not the count. A ranked list of the right length is exactly
    // what a model with a broken tokenizer also produces.
    const rows = scores.getByRole("listitem");
    await expect(rows.first()).toContainText(/NEGATIVE/i);

    // And the score is a real distribution, not a placeholder.
    const top = Number(
      (await rows.first().innerText()).match(/([01]\.\d+)/)?.[1] ?? "0",
    );
    expect(top).toBeGreaterThan(0.9);

    // The result is labelled with the sentence that produced it.
    await expect(page.getByTestId("ran-text")).toContainText("tedium");
  });

  test("the head-to-head really runs two models, not one twice", async ({
    page,
  }) => {
    // The bug this page can actually have, and it is invisible: a comparison
    // that renders the primary model's result twice, or runs the same
    // checkpoint under two labels, produces two plausible score lists side by
    // side with nothing failing.
    //
    // The assertion is **structural, not a guessed disagreement**. SST-2's head
    // has exactly two classes and FinBERT's has three, one of them `neutral` —
    // so two genuinely different models cannot produce the same label set.
    // Asserting instead that the two *rankings* differ would pin a property
    // neither model promises, which is the failure mode the video-text-to-text
    // spec was written around.
    test.setTimeout(12 * 60 * 1000);
    const model = new ModelPageObject(page);

    await page.goto("/text-classification");
    await model.load();
    await model.waitForReady();

    // The comparison is a second download, so it has a LOAD of its own.
    await page.getByRole("button", { name: /^FinBERT$/ }).last().click();
    const compare = page.getByTestId("compare-load");
    await compare.getByRole("button", { name: /^Load model$/ }).click();
    await compare.getByTestId("model-ready").waitFor({ timeout: 8 * 60 * 1000 });

    await page.getByRole("button", { name: /a flat statement/i }).click();
    await model.run(/classify with both/i);

    const labelsOf = async (testId: string) => {
      const rows = page.getByTestId(testId).getByRole("listitem");
      await expect(rows.first()).toBeVisible({ timeout: 120_000 });
      const texts = await rows.allInnerTexts();
      return texts.map((t) => t.split("\n")[0].trim().toLowerCase()).sort();
    };

    const primary = await labelsOf("answer-primary-scores");
    const secondary = await labelsOf("answer-compare-scores");

    expect(primary).toEqual(["negative", "positive"]);
    expect(secondary).toContain("neutral");
    expect(secondary).not.toEqual(primary);

    // Both answers are captions on the same captured sentence.
    await expect(page.getByTestId("ran-text")).toContainText("Thursday");
  });
});
