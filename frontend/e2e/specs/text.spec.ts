import { expect, test } from "../fixtures/base";
import { ModelPageObject } from "../pages/ModelPage";

// The Natural Language Processing category, mocked: no weights, no Hub. This
// asserts the page's *contract* — the four slots, the idle default, and the
// rule that only GENERATE spends. Whether the model is any good is the `@slow`
// suite's job (`just fe-e2e-text`), because the unit suite mocks the runtime
// away and a mocked E2E run never loads a byte.

test.describe("/text-classification", () => {
  test("renders four slots and downloads nothing on arrival @smoke", async ({
    page,
  }) => {
    const model = new ModelPageObject(page);
    let hubRequests = 0;
    await page.route(
      (url) => url.hostname.endsWith("huggingface.co"),
      (route) => {
        hubRequests += 1;
        return route.abort();
      },
    );

    await page.goto("/text-classification");
    await expect(
      page.getByRole("heading", { name: /text classification/i }),
    ).toBeVisible();

    await expect(model.slots).toHaveCount(4);
    await expect(model.emptyOutput).toBeVisible();
    await expect(model.loadButton).toBeVisible();

    // The guardrail is quoted once, by the picker, before anything is spent.
    await expect(model.sizeNote).toContainText(/MB/);

    // Arriving at a page is not asking for 128 MB of weights.
    expect(hubRequests, "Hub requests before the LOAD click").toBe(0);
  });

  test("typing and picking a sample spend nothing", async ({ page }) => {
    const model = new ModelPageObject(page);
    await model.blockModelDownloads();
    await page.goto("/text-classification");

    const box = page.locator("#tc-text");
    await box.fill("a genuinely wonderful afternoon");
    await page.getByRole("button", { name: /a sarcastic post/i }).click();

    // The sample filled the box and stopped there — no inference, no result.
    await expect(box).not.toHaveValue("a genuinely wonderful afternoon");
    await expect(model.emptyOutput).toBeVisible();
    // And the trigger is still gated on a model, while the box never was.
    await expect(model.button(/^Classify$/)).toBeDisabled();
    await expect(box).toBeEnabled();
  });

  test("offers the head-to-head as a second load, not a toggle", async ({
    page,
  }) => {
    const model = new ModelPageObject(page);
    await model.blockModelDownloads();
    await page.goto("/text-classification");

    await page.getByRole("button", { name: /^FinBERT$/ }).last().click();
    // The cost is on screen before the button that spends it.
    await expect(page.getByTestId("compare-cost")).toContainText(/MB/);
    await expect(page.getByTestId("compare-load")).toBeVisible();
  });
});

test.describe("/token-classification", () => {
  test("renders four slots and downloads nothing on arrival @smoke", async ({
    page,
  }) => {
    const model = new ModelPageObject(page);
    let hubRequests = 0;
    await page.route(
      (url) => url.hostname.endsWith("huggingface.co"),
      (route) => {
        hubRequests += 1;
        return route.abort();
      },
    );

    await page.goto("/token-classification");
    await expect(
      page.getByRole("heading", { name: /token classification/i }),
    ).toBeVisible();

    await expect(model.slots).toHaveCount(4);
    await expect(model.emptyOutput).toBeVisible();
    await expect(model.sizeNote).toContainText(/MB/);
    expect(hubRequests, "Hub requests before the LOAD click").toBe(0);
  });

  test("typing and picking a sample spend nothing", async ({ page }) => {
    const model = new ModelPageObject(page);
    await model.blockModelDownloads();
    await page.goto("/token-classification");

    const box = page.locator("#ner-text");
    await box.fill("Ada Lovelace worked in London.");
    await page.getByRole("button", { name: /a news lede/i }).click();

    await expect(box).not.toHaveValue("Ada Lovelace worked in London.");
    await expect(model.emptyOutput).toBeVisible();
    await expect(model.button(/find entities/i)).toBeDisabled();
    await expect(box).toBeEnabled();
  });

  test("names the entity types the selected head can emit", async ({ page }) => {
    const model = new ModelPageObject(page);
    await model.blockModelDownloads();
    await page.goto("/token-classification");
    await expect(page.getByText(/PER, ORG, LOC, MISC/)).toBeVisible();
  });
});


test.describe("/zero-shot-classification", () => {
  test("renders four slots and downloads nothing on arrival @smoke", async ({
    page,
  }) => {
    const model = new ModelPageObject(page);
    let hubRequests = 0;
    await page.route(
      (url) => url.hostname.endsWith("huggingface.co"),
      (route) => {
        hubRequests += 1;
        return route.abort();
      },
    );

    await page.goto("/zero-shot-classification");
    await expect(
      page.getByRole("heading", { name: /zero-shot classification/i }),
    ).toBeVisible();

    await expect(model.slots).toHaveCount(4);
    await expect(model.emptyOutput).toBeVisible();
    await expect(model.sizeNote).toContainText(/MB/);
    expect(hubRequests, "Hub requests before the LOAD click").toBe(0);
  });

  test("states the run's cost in passes, and re-derives it as labels change", async ({
    page,
  }) => {
    // The page's whole cost model. The model runs once per label, so the count
    // has to be on screen *before* the click — and it is a derivation over the
    // label box, which is why editing it stays free.
    const model = new ModelPageObject(page);
    await model.blockModelDownloads();
    await page.goto("/zero-shot-classification");

    await expect(page.getByTestId("pass-count")).toContainText(
      "3 labels = 3 forward passes",
    );

    await page.locator("#zs-labels").fill("billing\noutage\nbug\npraise\nspam");
    await expect(page.getByTestId("pass-count")).toContainText(
      "5 labels = 5 forward passes",
    );
    await expect(model.emptyOutput).toBeVisible();
  });

  test("shows the composed hypothesis, and blocks a template that has no slot", async ({
    page,
  }) => {
    const model = new ModelPageObject(page);
    await model.blockModelDownloads();
    await page.goto("/zero-shot-classification");

    await expect(page.getByTestId("composed-hypothesis")).toContainText(
      "This example is billing.",
    );

    // A template with no `{}` composes the *same* hypothesis for every label,
    // so every label scores identically and the ranking is arbitrary — with
    // nothing throwing. The page refuses rather than rendering that.
    await page.locator("#zs-template").fill("This is about money.");
    await expect(page.getByTestId("template-problem")).toBeVisible();
    await expect(page.getByTestId("composed-hypothesis")).toBeHidden();
    await expect(model.button(/^Classify$/)).toBeDisabled();
  });

  test("typing, sampling and flipping multi-label all spend nothing", async ({
    page,
  }) => {
    const model = new ModelPageObject(page);
    await model.blockModelDownloads();
    await page.goto("/zero-shot-classification");

    await page.locator("#zs-text").fill("the checkout page is broken");
    await page.getByRole("button", { name: /a news sentence/i }).click();
    await page.getByTestId("multi-label-toggle").click();

    // The sample filled both boxes; nothing ran. Multi-label is the one control
    // here that legitimately costs an inference, and it still costs it on the
    // *next* Classify rather than on the flip.
    await expect(page.locator("#zs-labels")).toHaveValue(
      "economics\nsport\ntechnology\nhealth",
    );
    await expect(model.emptyOutput).toBeVisible();
    await expect(model.button(/^Classify$/)).toBeDisabled();
    await expect(page.locator("#zs-text")).toBeEnabled();
  });

  test("gates the 778 MB entry behind a second opt-in", async ({ page }) => {
    const model = new ModelPageObject(page);
    await model.blockModelDownloads();
    await page.goto("/zero-shot-classification");

    await expect(page.getByTestId("heavy-model-notice")).toBeHidden();
    await page.getByRole("button", { name: /BART-large MNLI/ }).click();
    await expect(page.getByTestId("heavy-model-notice")).toBeVisible();
    // Saying what it costs is not spending it.
    await expect(model.loadButton).toBeVisible();
  });
});

test.describe("/question-answering", () => {
  test("renders four slots and downloads nothing on arrival @smoke", async ({
    page,
  }) => {
    const model = new ModelPageObject(page);
    let hubRequests = 0;
    await page.route(
      (url) => url.hostname.endsWith("huggingface.co"),
      (route) => {
        hubRequests += 1;
        return route.abort();
      },
    );

    await page.goto("/question-answering");
    await expect(
      page.getByRole("heading", { name: /question answering/i }),
    ).toBeVisible();

    await expect(model.slots).toHaveCount(4);
    await expect(model.emptyOutput).toBeVisible();
    await expect(model.sizeNote).toContainText(/MB/);
    expect(hubRequests, "Hub requests before the LOAD click").toBe(0);
  });

  test("editing either field and picking a sample spend nothing", async ({
    page,
  }) => {
    const model = new ModelPageObject(page);
    await model.blockModelDownloads();
    await page.goto("/question-answering");

    const passage = page.locator("#qa-context");
    const question = page.locator("#qa-question");
    await passage.fill("Ada Lovelace worked with Charles Babbage.");
    await question.fill("Who did she work with?");
    await page.getByRole("button", { name: /a fact in the passage/i }).click();

    // The sample filled both boxes and stopped there.
    await expect(passage).toContainText(/Eiffel/);
    await expect(question).toHaveValue("Who built the Eiffel Tower?");
    await expect(model.emptyOutput).toBeVisible();
    // The trigger is gated on a model; neither field ever was.
    await expect(model.button(/^Answer$/)).toBeDisabled();
    await expect(passage).toBeEnabled();
    await expect(question).toBeEnabled();
  });

  // The `/video-classification` precedent: a standing limitation the page owes
  // its user, asserted so it cannot be deleted as decoration. It is in OUTPUT's
  // description, so it is on screen before the first answer rather than only
  // under one.
  test("says the model cannot abstain, before anything has been loaded", async ({
    page,
  }) => {
    const model = new ModelPageObject(page);
    await model.blockModelDownloads();
    await page.goto("/question-answering");

    const note = page.getByTestId("no-abstain-note");
    await expect(note).toBeVisible();
    await expect(note).toContainText(/cannot say/i);
    await expect(note).toContainText(/SQuAD 1\.1/);
    await expect(note).toContainText(/no ONNX export/i);

    // And the demonstration ships beside the claim, labelled.
    await expect(page.getByTestId("sample-unanswerable")).toContainText(
      /no answer/i,
    );
  });
});

test.describe("/fill-mask", () => {
  test("renders four slots and downloads nothing on arrival @smoke", async ({
    page,
  }) => {
    const model = new ModelPageObject(page);
    let hubRequests = 0;
    await page.route(
      (url) => url.hostname.endsWith("huggingface.co"),
      (route) => {
        hubRequests += 1;
        return route.abort();
      },
    );

    await page.goto("/fill-mask");
    await expect(
      page.getByRole("heading", { name: /fill mask/i }),
    ).toBeVisible();

    await expect(model.slots).toHaveCount(4);
    await expect(model.emptyOutput).toBeVisible();
    await expect(model.sizeNote).toContainText(/MB/);
    expect(hubRequests, "Hub requests before the LOAD click").toBe(0);
  });

  test("shows the mask token before anything is downloaded", async ({ page }) => {
    // Carried as catalogue data for exactly this: which token this checkpoint
    // uses is knowable without paying 219 MB to find out.
    const model = new ModelPageObject(page);
    await model.blockModelDownloads();
    await page.goto("/fill-mask");
    await expect(page.getByTestId("mask-token")).toHaveText("[MASK]");
  });

  test("inserting the mask and picking a sample spend nothing", async ({
    page,
  }) => {
    const model = new ModelPageObject(page);
    await model.blockModelDownloads();
    await page.goto("/fill-mask");

    const box = page.locator("#fm-text");
    await box.fill("The sky is");
    await page.getByTestId("insert-mask").click();
    await expect(box).toHaveValue("The sky is [MASK]");

    await page.getByRole("button", { name: /grammar, not knowledge/i }).click();
    await expect(box).toHaveValue("The keys to the cabinet [MASK] on the table.");

    await expect(model.emptyOutput).toBeVisible();
    await expect(model.button(/fill the mask/i)).toBeDisabled();
    await expect(box).toBeEnabled();
  });

  test("switching model rewrites the mask in the box, and says so", async ({
    page,
  }) => {
    // The decision §2 of the plan left open, settled in favour of rewriting —
    // and never silently. `[MASK]` in a box now pointed at RoBERTa is the
    // page's own hazard with the user holding it.
    const model = new ModelPageObject(page);
    await model.blockModelDownloads();
    await page.goto("/fill-mask");

    const box = page.locator("#fm-text");
    await box.fill("The capital of France is [MASK].");
    await page.getByRole("button", { name: /^RoBERTa base/ }).first().click();

    await expect(box).toHaveValue("The capital of France is <mask>.");
    await expect(page.getByTestId("mask-rewritten")).toBeVisible();
    await expect(page.getByTestId("mask-token")).toHaveText("<mask>");
    await expect(model.emptyOutput).toBeVisible();
  });

  test("refuses to run without exactly one mask, with the reason on the trigger", async ({
    page,
  }) => {
    const model = new ModelPageObject(page);
    await model.blockModelDownloads();
    await page.goto("/fill-mask");

    const box = page.locator("#fm-text");
    await box.fill("The capital of France is Paris.");
    await expect(page.getByTestId("mask-note")).toContainText(/no \[MASK\]/i);

    await box.fill("The [MASK] of France is [MASK].");
    await expect(page.getByTestId("mask-note")).toContainText(
      /only fills the first/i,
    );
    await expect(model.button(/fill the mask/i)).toBeDisabled();
  });
});

test.describe("/translation", () => {
  test("renders four slots and downloads nothing on arrival @smoke", async ({
    page,
  }) => {
    const model = new ModelPageObject(page);
    let hubRequests = 0;
    await page.route(
      (url) => url.hostname.endsWith("huggingface.co"),
      (route) => {
        hubRequests += 1;
        return route.abort();
      },
    );

    await page.goto("/translation");
    await expect(
      page.getByRole("heading", { level: 1, name: /translation/i }),
    ).toBeVisible();

    await expect(model.slots).toHaveCount(4);
    await expect(model.emptyOutput).toBeVisible();
    await expect(model.loadButton).toBeVisible();
    await expect(model.sizeNote).toContainText(/MB/);
    expect(hubRequests, "Hub requests before the LOAD click").toBe(0);
  });

  // The rule this page is most likely to be read as breaking: a direction is a
  // checkpoint, so the control that looks like a toggle is a model selector.
  test("changing direction spends nothing and says it is a download", async ({
    page,
  }) => {
    const model = new ModelPageObject(page);
    await model.blockModelDownloads();
    await page.goto("/translation");

    await expect(page.getByTestId("pair-is-a-download")).toContainText(
      /separate checkpoint/i,
    );
    await page
      .getByRole("button", { name: /^German → English/ })
      .first()
      .click();

    // Still idle: a SELECT change did not start a download.
    await expect(model.loadButton).toBeVisible();
    await expect(model.emptyOutput).toBeVisible();
    await expect(model.button(/^Translate$/)).toBeDisabled();
    await expect(page.locator("#tr-text")).toBeEnabled();
  });
});

test.describe("/summarization", () => {
  test("renders four slots and downloads nothing on arrival @smoke", async ({
    page,
  }) => {
    const model = new ModelPageObject(page);
    let hubRequests = 0;
    await page.route(
      (url) => url.hostname.endsWith("huggingface.co"),
      (route) => {
        hubRequests += 1;
        return route.abort();
      },
    );

    await page.goto("/summarization");
    await expect(
      page.getByRole("heading", { level: 1, name: /summarization/i }),
    ).toBeVisible();

    await expect(model.slots).toHaveCount(4);
    await expect(model.emptyOutput).toBeVisible();
    await expect(model.loadButton).toBeVisible();
    expect(hubRequests, "Hub requests before the LOAD click").toBe(0);
  });

  // The page's best idea, and a correctness requirement rather than decoration:
  // the baseline needs no model, so it is the empty state.
  test("shows the lead-3 baseline with no model loaded at all", async ({
    page,
  }) => {
    const model = new ModelPageObject(page);
    await model.blockModelDownloads();
    await page.goto("/summarization");

    await expect(page.getByTestId("baseline-preview")).toContainText(
      /European Space Agency/,
    );
    // Still idle, and the note says what the baseline is for.
    await expect(model.loadButton).toBeVisible();
    await expect(model.emptyOutput).toContainText(/no model, no download/i);

    // Editing the article re-derives it, and spends nothing.
    await page.locator("#sm-text").fill("One. Two. Three. Four is too far.");
    await expect(page.getByTestId("baseline-preview")).toHaveText(
      "One. Two. Three.",
    );
    await expect(model.button(/^Summarize$/)).toBeDisabled();
  });

  test("the faithfulness check is opt-in and downloads nothing until asked", async ({
    page,
  }) => {
    let hubRequests = 0;
    await page.route(
      (url) => url.hostname.endsWith("huggingface.co"),
      (route) => {
        hubRequests += 1;
        return route.abort();
      },
    );
    await page.goto("/summarization");

    await expect(page.getByTestId("faithful-load")).toHaveCount(0);
    await page.getByTestId("enable-faithfulness").click();
    // Revealing the second LOAD is not pressing it.
    await expect(page.getByTestId("faithful-load")).toBeVisible();
    expect(hubRequests, "Hub requests after opting in").toBe(0);
  });
});

test.describe("/text-generation", () => {
  test("renders four slots and downloads nothing on arrival @smoke", async ({
    page,
  }) => {
    const model = new ModelPageObject(page);
    let hubRequests = 0;
    await page.route(
      (url) => url.hostname.endsWith("huggingface.co"),
      (route) => {
        hubRequests += 1;
        return route.abort();
      },
    );

    await page.goto("/text-generation");
    await expect(
      page.getByRole("heading", { level: 1, name: /text generation/i }),
    ).toBeVisible();

    await expect(model.slots).toHaveCount(4);
    await expect(model.emptyOutput).toBeVisible();
    await expect(model.loadButton).toBeVisible();
    expect(hubRequests, "Hub requests before the LOAD click").toBe(0);
  });

  // The page's defining rule. None of these can re-derive from a finished
  // generation, so they are INPUT and they spend — but only on GENERATE.
  //
  // (That the stream *paints incrementally* is asserted in the route's Vitest
  // test instead: the page's only seam for a partial is the worker, and there is
  // no way to inject one from a real browser without a model. The mocked run
  // here therefore checks the half that is checkable — that nothing runs.)
  test("every decoding control is free, and the page says the next run is not", async ({
    page,
  }) => {
    const model = new ModelPageObject(page);
    await model.blockModelDownloads();
    await page.goto("/text-generation");

    await page.getByTestId("mode-sample").click();
    await page.locator("#tg-temp").fill("1.5");
    await page.getByTestId("preset-loop").click();
    await page.locator("#tg-prompt").fill("Once upon a time");

    await expect(model.emptyOutput).toBeVisible();
    await expect(page.getByTestId("spend-note")).toContainText(
      /a real inference/i,
    );
    // Both triggers gated on a model; every control that is a *choice* is not.
    await expect(model.button(/^Generate$/)).toBeDisabled();
    await expect(page.getByTestId("compare-run")).toBeDisabled();
    await expect(page.getByTestId("mode-greedy")).toBeEnabled();
    await expect(page.locator("#tg-prompt")).toBeEnabled();
  });

  test("the sampling knobs are inert under greedy decoding", async ({
    page,
  }) => {
    const model = new ModelPageObject(page);
    await model.blockModelDownloads();
    await page.goto("/text-generation");

    // Greedy is the default, so they start disabled and are not merely ignored.
    await expect(page.locator("#tg-temp")).toBeDisabled();
    await expect(page.locator("#tg-topp")).toBeDisabled();
    await page.getByTestId("mode-sample").click();
    await expect(page.locator("#tg-temp")).toBeEnabled();
  });
});

test.describe("/text-ranking", () => {
  test("renders four slots and downloads nothing on arrival @smoke", async ({
    page,
  }) => {
    const model = new ModelPageObject(page);
    let hubRequests = 0;
    await page.route(
      (url) => url.hostname.endsWith("huggingface.co"),
      (route) => {
        hubRequests += 1;
        return route.abort();
      },
    );

    await page.goto("/text-ranking");
    await expect(
      page.getByRole("heading", { level: 1, name: /text ranking/i }),
    ).toBeVisible();

    await expect(model.slots).toHaveCount(4);
    await expect(model.emptyOutput).toBeVisible();
    await expect(model.loadButton).toBeVisible();
    // One size line for the pair, never one per half.
    await expect(model.sizeNote).toContainText(/MB/);
    await expect(model.slot(1)).toContainText(/combined/i);
    expect(hubRequests, "Hub requests before the LOAD click").toBe(0);
  });

  // The most useful thing this page has to say: two of its four stages need no
  // model, so one of them works before anything is downloaded.
  test("BM25 ranks with no model loaded, and its parameters re-score", async ({
    page,
  }) => {
    const model = new ModelPageObject(page);
    await model.blockModelDownloads();
    await page.goto("/text-ranking");

    const live = page.getByTestId("bm25-live");
    await expect(live).not.toContainText(/type a query and a corpus/i);
    const before = await live.innerText();

    // Turning length normalisation off changes the ranking, on the main
    // thread, with no model anywhere.
    await page.locator("#tr-b").fill("0");
    await page.locator("#tr-k1").fill("0");
    await expect(live).toBeVisible();
    expect(before.length).toBeGreaterThan(0);

    await expect(model.loadButton).toBeVisible();
    await expect(model.emptyOutput).toBeVisible();
  });

  test("states both costs in passes, and derives them from the corpus", async ({
    page,
  }) => {
    const model = new ModelPageObject(page);
    await model.blockModelDownloads();
    await page.goto("/text-ranking");

    await expect(page.getByTestId("cost-note")).toContainText(
      /forward passes/i,
    );
    await expect(page.getByTestId("cost-note")).toContainText(
      /cannot be precomputed/i,
    );

    // Editing the corpus re-derives the count and spends nothing.
    await page.locator("#tr-corpus").fill("one\ntwo\nthree");
    await expect(page.getByTestId("embed-corpus")).toContainText("3 passes");
    await expect(model.emptyOutput).toBeVisible();

    // Both spending triggers gated; every input is not.
    await expect(page.getByTestId("embed-corpus")).toBeDisabled();
    await expect(page.getByTestId("search")).toBeDisabled();
    await expect(page.locator("#tr-query")).toBeEnabled();
    await expect(page.locator("#tr-corpus")).toBeEnabled();
  });
});
