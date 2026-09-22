import { ModelPageObject } from "../pages/ModelPage";
import { expect, test } from "../fixtures/base";

// @slow — downloads real ONNX weights from Hugging Face and opens a real ONNX
// Runtime session. Opt in with `just fe-e2e-docvqa` (E2E_SLOW=1).
//
// Unlike the VLM spec this one does **not** need a GPU: Donut is an encoder plus
// a short extractive decode, the catalogue gates no backend, and 219 MB at q8 on
// WASM is a real path rather than a theoretical one. It runs in the `chromium`
// project for exactly that reason.
//
// **The assertion is a known answer on a known document**, and here that is
// unusually literal: `invoice.png` is the Transformers.js docs' own DocVQA
// example, and its invoice number is `us-001`. "Some text appeared" would pass
// while the processor fed the model a blank pad — which is the failure this page
// can actually have, since resolution handling is its whole correctness surface.

const DOWNLOAD_BUDGET_MS = 8 * 60 * 1000;
const ANSWER_BUDGET_MS = 3 * 60 * 1000;

test.describe("@slow real document-QA loads", () => {
  test.describe.configure({ mode: "serial", timeout: DOWNLOAD_BUDGET_MS });

  async function load(page: import("@playwright/test").Page) {
    const model = new ModelPageObject(page);
    await page.goto("/document-question-answering");
    await model.load();
    await expect(page.getByTestId("model-ready")).toBeVisible({
      timeout: DOWNLOAD_BUDGET_MS,
    });
    return model;
  }

  test("reads the invoice number off the sample invoice", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = await load(page);

    await model.button(/^Invoice$/).click();
    await page.getByTestId("question-input").fill("What is the invoice number?");
    await model.run(/^Generate$/);

    // The known answer. A mis-handled resize, a lost prompt format or a broken
    // extraction regex all produce *something* here; only this pins the page.
    await expect(page.getByTestId("answer-text")).toHaveText(/us-001/i, {
      timeout: ANSWER_BUDGET_MS,
    });
  });

  test("the answer tracks the question, not just the document", async ({
    page,
    mockApi,
  }) => {
    // The sharpest evidence that the question actually reaches the model: the
    // pipeline builds Donut's prompt itself, so a question silently dropped
    // would still return a plausible field from the same page.
    await mockApi();
    const model = await load(page);

    await model.button(/^Invoice$/).click();
    await page.getByTestId("question-input").fill("What is the invoice number?");
    await model.run(/^Generate$/);
    await expect(page.getByTestId("answer-text")).toBeVisible({
      timeout: ANSWER_BUDGET_MS,
    });
    const first = await page.getByTestId("answer-text").innerText();

    await page.getByTestId("question-input").fill("What is the date?");
    await model.run(/^Generate$/);
    await expect
      .poll(async () => page.getByTestId("answer-text").innerText(), {
        timeout: ANSWER_BUDGET_MS,
      })
      .not.toBe(first);
  });

  test("picking a document runs nothing, even with the model loaded", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const model = await load(page);

    await model.button(/^Invoice$/).click();
    await model.button(/^Receipt$/).click();
    await expect(model.emptyOutput).toBeVisible();
  });
});
