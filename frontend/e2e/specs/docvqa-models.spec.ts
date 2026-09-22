import { ModelPageObject } from "../pages/ModelPage";
import { expect, test } from "../fixtures/base";

// @slow — downloads real ONNX weights from Hugging Face and opens a real ONNX
// Runtime session. Opt in with `just fe-e2e-docvqa` (E2E_SLOW=1).
//
// Unlike the VLM spec this one needs **no GPU**: Donut is an encoder plus a short
// extractive decode, the catalogue gates no backend, and the WASM path is real.
// It runs in the `chromium` project for exactly that reason.
//
// **This spec is what found the WASM decoder bug.** A uniform q8 cannot open a
// session at all (`qdq_actions.cc:137 … Missing required scale`), and the unit
// suite, the mocked E2E run, the Hub file check and even a direct
// `onnxruntime-node` load all missed it — the last because the fault is specific
// to the WASM provider in the *browser*. See `dtypes` in
// `multimodal/docvqa/types.ts`.
//
// **It is one test on purpose.** The model is ~597 MB on WASM (an fp32 decoder,
// per that same bug), and Playwright gives each test a fresh context, so a file
// of three tests re-downloads it three times — 8 minutes became 25. Everything
// worth asserting here needs the same loaded model, so it is one arrange and
// three acts rather than three tests.

const DOWNLOAD_BUDGET_MS = 10 * 60 * 1000;
/** One extractive decode on a WASM fp32 decoder. Slow, and legitimately so. */
const ANSWER_BUDGET_MS = 6 * 60 * 1000;

test.describe("@slow real document-QA loads", () => {
  test("loads Donut and reads fields off a real invoice", async ({
    page,
    mockApi,
  }) => {
    test.setTimeout(25 * 60 * 1000);
    await mockApi();
    const model = new ModelPageObject(page);

    await page.goto("/document-question-answering");
    await model.load();
    await expect(page.getByTestId("model-ready")).toBeVisible({
      timeout: DOWNLOAD_BUDGET_MS,
    });

    /** Ask, wait for the run to actually finish, then read the answer. */
    async function ask(question: string): Promise<string> {
      await page.getByTestId("question-input").fill(question);
      await model.run(/^Generate$/);
      // The trigger reads "Reading…" and is disabled for the whole run, so its
      // return is the unambiguous end-of-run signal. `answer-text` is **not**:
      // the previous answer stays on screen until this one replaces it, so
      // polling it for a change cannot tell "still running" from "same answer".
      // The first version of this spec failed on exactly that.
      await expect(model.button(/^Generate$/)).toBeEnabled({
        timeout: ANSWER_BUDGET_MS,
      });
      return (await page.getByTestId("answer-text").innerText()).trim();
    }

    await model.button(/^Invoice$/).click();

    // A known answer on a known document: `invoice.png` is the Transformers.js
    // docs' own DocVQA example and its invoice number is `us-001`. A mis-handled
    // resize, a lost prompt format or a broken extraction regex all produce
    // *something* here — only the specific value pins the page.
    const number = await ask("What is the invoice number?");
    expect(number).toMatch(/us-001/i);

    // The sharpest evidence that the question actually reaches the model: the
    // pipeline builds Donut's prompt itself, so a question silently dropped
    // would still return a plausible field from the same page.
    const date = await ask("What is the date?");
    expect(date).not.toBe(number);

    // §1.6 against a real loaded model: browsing documents must not cost
    // inferences. Asserted on the *asked* label rather than `output-empty`,
    // because by now an answer is legitimately on screen.
    const asked = await page.getByTestId("answer-asked").innerText();
    await model.button(/^Receipt$/).click();
    await expect(model.button(/^Generate$/)).toBeEnabled();
    expect(await page.getByTestId("answer-asked").innerText()).toBe(asked);
  });
});
