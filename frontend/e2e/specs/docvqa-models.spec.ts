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
// **It is one test, and one inference, on purpose.**
//
//   one test        The model is ~597 MB on WASM (an fp32 decoder, per that same
//                   bug), and Playwright gives each test a fresh context, so a
//                   file of three tests re-downloads it three times — 8 minutes
//                   became 25.
//   one inference   The fp32 decoder is *slow*, and measurably so: a second
//                   question on an already-loaded model did not finish inside a
//                   **six-minute** budget. In Node — where the decoder is q8,
//                   because `onnxruntime-node` does not have the bug — the same
//                   three questions answer in 7-8 s each. That gap is the price
//                   of the workaround, not a fault in the page, and the route
//                   says so in its own copy.
//
// So the assertion kept here is the one that cannot be obtained any other way: a
// real browser load, on the real WASM provider, producing a known answer. The
// "does the question reach the model" property is cheap to establish off-browser
// (three questions, three different correct answers) and does not justify another
// six minutes of CI-less wall clock here.

const DOWNLOAD_BUDGET_MS = 10 * 60 * 1000;
/** One extractive decode on a WASM fp32 decoder. Minutes, and legitimately so. */
const ANSWER_BUDGET_MS = 10 * 60 * 1000;

test.describe("@slow real document-QA loads", () => {
  test("loads Donut and reads the invoice number off a real invoice", async ({
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

    // The answer is labelled with the question it was actually asked, captured
    // inside the run rather than read off the box afterwards.
    await expect(page.getByTestId("answer-asked")).toContainText(
      "What is the invoice number?",
    );

    // §1.6 against a real loaded model: browsing documents must not cost
    // inferences. Asserted on the *asked* label rather than `output-empty`,
    // because by now an answer is legitimately on screen. Free — no inference.
    const asked = await page.getByTestId("answer-asked").innerText();
    await model.button(/^Receipt$/).click();
    await expect(model.button(/^Generate$/)).toBeEnabled();
    expect(await page.getByTestId("answer-asked").innerText()).toBe(asked);
  });
});
