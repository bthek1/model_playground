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

test.describe("@slow NER", () => {
  test.slow();

  // **The offsets, not the count.** A count-based assertion passes while every
  // highlight sits two characters to the left of the word it means — which is
  // exactly what rebuilding the text from the model's tokens produces, and it
  // reads as a styling problem rather than a wrong answer. So this asserts the
  // marked text is *the words themselves*.
  test("marks the right characters, not merely the right number of them", async ({
    page,
  }) => {
    test.setTimeout(8 * 60 * 1000);
    const model = new ModelPageObject(page);

    await page.goto("/token-classification");
    await model.load();
    await model.waitForReady();

    await page
      .locator("#ner-text")
      .fill("Priya Raman flew from Wellington to Berlin to meet Siemens.");
    await model.run(/find entities/i);

    const marks = page.getByTestId("span-mark");
    await expect(marks.first()).toBeVisible({ timeout: 120_000 });

    // Each mark's own text is the entity, exactly — no leading space, no
    // truncated final character, no subword fragment.
    const byLabel = new Map<string, string[]>();
    for (const mark of await marks.all()) {
      const label = (await mark.getAttribute("data-label")) ?? "";
      // Read the text node, not the mark: the type tag is an inline sibling,
      // so the mark's own `innerText` is "Priya Raman PER" and comparing
      // against it would pass regardless of where the span actually starts.
      const text = await mark.getByTestId("span-text").innerText();
      byLabel.set(label, [...(byLabel.get(label) ?? []), text]);
    }

    expect(byLabel.get("PER")).toContain("Priya Raman");
    expect(byLabel.get("LOC")).toEqual(
      expect.arrayContaining(["Wellington", "Berlin"]),
    );
    expect(byLabel.get("ORG")).toContain("Siemens");

    // A subword fragment would show up as a mark that is not a whole word —
    // which is what a missing `aggregation_strategy` produces.
    for (const texts of byLabel.values()) {
      for (const t of texts) expect(t).not.toMatch(/^##/);
    }
  });

  test("redaction removes the names and keeps everything else", async ({
    page,
  }) => {
    test.setTimeout(8 * 60 * 1000);
    const model = new ModelPageObject(page);

    await page.goto("/token-classification");
    await model.load();
    await model.waitForReady();

    await page
      .locator("#ner-text")
      .fill("Priya Raman flew from Wellington to Berlin to meet Siemens.");
    await model.run(/find entities/i);
    await expect(page.getByTestId("span-mark").first()).toBeVisible({
      timeout: 120_000,
    });

    await page.getByTestId("redact-toggle").click();
    const overlay = page.getByTestId("span-overlay");
    await expect(overlay).toContainText("[PER]");

    // The person and the places are gone; the organisation and the ordinary
    // words are untouched — redaction is per type, not all-or-nothing.
    await expect(overlay).not.toContainText("Priya");
    await expect(overlay).not.toContainText("Wellington");
    await expect(overlay).toContainText("Siemens");
    await expect(overlay).toContainText("flew from");
  });
});


test.describe("@slow zero-shot classification", () => {
  test.slow();

  const TICKET =
    "I was charged twice for the same subscription this month and the second payment has not been refunded.";

  /** The score list's rows, top-ranked first. */
  async function ranking(page: import("@playwright/test").Page) {
    const rows = page.getByTestId("score-list").getByRole("listitem");
    await expect(rows.first()).toBeVisible({ timeout: 180_000 });
    return (await rows.allInnerTexts()).map((t) =>
      t.split("\n")[0].trim().toLowerCase(),
    );
  }

  test("puts the right label first on labels the model has never seen", async ({
    page,
  }) => {
    // **A known ranking on a known sentence**, not "three rows appeared" —
    // which is exactly what a run with a broken template also produces, since
    // every label would still come back with a plausible number beside it.
    test.setTimeout(8 * 60 * 1000);
    const model = new ModelPageObject(page);

    await page.goto("/zero-shot-classification");
    await model.load();
    await model.waitForReady();

    await page.locator("#zs-text").fill(TICKET);
    await page.locator("#zs-labels").fill("billing\noutage\nfeature request");
    await model.run(/^Classify$/);

    expect((await ranking(page))[0]).toBe("billing");

    // The answer is captioned with the label set and the template it was
    // actually given — the half of the result that is not the numbers.
    await expect(page.getByTestId("ran-labels")).toContainText("billing");
    await expect(page.getByTestId("ran-template")).toContainText(
      "This example is {}.",
    );
    // Three labels is three forward passes, and the page says so afterwards too.
    await expect(page.getByTestId("ran-cost")).toContainText("3 passes");
  });

  test("the hypothesis template reaches the model", async ({ page }) => {
    // The point of putting the template on screen: if it were ignored — or
    // silently replaced by the pipeline's own default, which is what happens
    // when a page does not pass one — two different templates would produce
    // **identical** scores. Nothing else on the page can catch that.
    //
    // So this asserts the numbers *move*, not that the ranking changes: a good
    // model should still route a billing complaint to `billing` under either
    // frame, and asserting a flip would pin a property the model does not
    // promise.
    test.setTimeout(10 * 60 * 1000);
    const model = new ModelPageObject(page);

    await page.goto("/zero-shot-classification");
    await model.load();
    await model.waitForReady();

    await page.locator("#zs-text").fill(TICKET);
    await page.locator("#zs-labels").fill("billing\noutage\nfeature request");

    const topScore = async () => {
      const rows = page.getByTestId("score-list").getByRole("listitem");
      await expect(rows.first()).toBeVisible({ timeout: 180_000 });
      return Number(
        (await rows.first().innerText()).match(/([01]\.\d+)/)?.[1] ?? "-1",
      );
    };

    await model.run(/^Classify$/);
    const framed = await topScore();
    expect(framed).toBeGreaterThan(0.4);

    // The bare template: the label as the whole hypothesis, no sentence around
    // it. Same premise, same labels, same model — only the frame differs.
    await page.getByRole("button", { name: /^Label only$/ }).click();
    await expect(page.getByTestId("composed-hypothesis")).toContainText(
      "billing",
    );
    await model.run(/^Classify$/);
    await expect(page.getByTestId("ran-template")).toHaveText("“{}”");

    const bare = await topScore();
    expect(bare).toBeGreaterThan(0);
    expect(
      Math.abs(bare - framed),
      "two templates produced identical scores — the template is not reaching the model",
    ).toBeGreaterThan(0.001);
  });

  test("multi-label changes the arithmetic, and only on the next press", async ({
    page,
  }) => {
    // Single-label is one softmax across the labels, so the scores sum to 1;
    // multi-label scores each label against its own contradiction logit, so
    // they do not. A page that "re-derived" the toggle from a finished result
    // could not produce the second set of numbers at all.
    test.setTimeout(10 * 60 * 1000);
    const model = new ModelPageObject(page);

    await page.goto("/zero-shot-classification");
    await model.load();
    await model.waitForReady();

    await page
      .locator("#zs-text")
      .fill("The app crashed during checkout and I was still charged.");
    await page.locator("#zs-labels").fill("billing\nbug report\npraise");

    const scores = async () => {
      const rows = page.getByTestId("score-list").getByRole("listitem");
      await expect(rows.first()).toBeVisible({ timeout: 180_000 });
      return (await rows.allInnerTexts()).map((t) =>
        Number(t.match(/([01]\.\d+)/)?.[1] ?? "0"),
      );
    };

    await model.run(/^Classify$/);
    const single = await scores();
    expect(single.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 1);

    // Flipping it spends nothing — the result on screen is still the old one.
    await page.getByTestId("multi-label-toggle").click();
    await expect(page.getByTestId("scoring-note")).toContainText(
      /single-label/i,
    );

    await model.run(/^Classify$/);
    await expect(page.getByTestId("scoring-note")).toContainText(
      /multi-label/i,
      { timeout: 180_000 },
    );
    const multi = await scores();
    // Two things are true at once and only one of them is about the sum:
    // independent scoring frees the labels from competing, and on a genuinely
    // two-topic sentence that is visible in the numbers.
    expect(multi.reduce((a, b) => a + b, 0)).toBeGreaterThan(1.05);
  });
});

test.describe("@slow question answering", () => {
  test.slow();

  // The passage the first sample ships, repeated here so the assertion's
  // character offsets are checkable against something visible in this file.
  const EIFFEL =
    "The Eiffel Tower was built by Gustave Eiffel for the 1889 World's Fair in Paris. It stood as the world's tallest man-made structure for 41 years, until the Chrysler Building in New York was finished in 1930.";

  test("marks the answer at the right characters, not merely the right words", async ({
    page,
  }) => {
    // **The assertion is a character range**, and that is the point of running
    // it at all. "A span appeared" passes while the alignment is off by a
    // token; "the span reads Gustave Eiffel" passes while it marks the *second*
    // mention of a name; only the offsets pin the thing that can silently go
    // wrong. They are the numbers measured off the real q8 build.
    test.setTimeout(6 * 60 * 1000);
    const model = new ModelPageObject(page);

    await page.goto("/question-answering");
    await model.load();
    await model.waitForReady();

    await page.getByRole("button", { name: /a fact in the passage/i }).click();
    await model.run(/^Answer$/);

    const range = page.getByTestId("answer-range");
    await expect(range).toBeVisible({ timeout: 180_000 });
    await expect(range).toHaveAttribute("data-start", "30");
    await expect(range).toHaveAttribute("data-end", "44");

    // And the marked characters really are those characters.
    expect(EIFFEL.slice(30, 44)).toBe("Gustave Eiffel");
    await expect(page.getByTestId("span-text")).toHaveText("Gustave Eiffel");

    // The score is a real number, not a placeholder.
    const score = Number(await page.getByTestId("answer-score").innerText());
    expect(score).toBeGreaterThan(0.9);
  });

  test("recovers a span whose own decode is not in the passage", async ({
    page,
  }) => {
    // The whole justification for `text/offsets.ts` over the pipeline. The
    // model's answer decodes as `general - purpose compute shaders`, which does
    // not occur in the passage — so the obvious shortcut, searching the passage
    // for the answer string, highlights nothing. Slicing [213, 244) returns the
    // passage's own characters, hyphen intact. A unit test can only assert this
    // against a tokenization we wrote down; this asserts it against the model.
    test.setTimeout(6 * 60 * 1000);
    const model = new ModelPageObject(page);

    await page.goto("/question-answering");
    await model.load();
    await model.waitForReady();

    await page
      .getByRole("button", { name: /an answer the tokenizer re-spaces/i })
      .click();
    await model.run(/^Answer$/);

    const marked = page.getByTestId("span-text");
    await expect(marked).toBeVisible({ timeout: 180_000 });
    await expect(marked).toHaveText("general-purpose compute shaders");
    await expect(page.getByTestId("answer-range")).toHaveAttribute(
      "data-start",
      "213",
    );
  });

  test("answers a question its passage cannot answer, which is the lesson", async ({
    page,
  }) => {
    // The model has no way to decline, so it returns a span — and not a
    // hesitant one. Asserting *that it answers* is the demonstration; asserting
    // a particular wrong answer would pin a property of this checkpoint's
    // quantization rather than of SQuAD 1.1.
    test.setTimeout(6 * 60 * 1000);
    const model = new ModelPageObject(page);

    await page.goto("/question-answering");
    await model.load();
    await model.waitForReady();

    await page.getByTestId("sample-unanswerable").click();
    await model.run(/^Answer$/);

    const marked = page.getByTestId("span-text");
    await expect(marked).toBeVisible({ timeout: 180_000 });
    // Something was highlighted, from a passage that says nothing about it.
    expect((await marked.innerText()).trim().length).toBeGreaterThan(0);
    expect(EIFFEL).toContain((await marked.innerText()).trim());
    // And the standing note is on screen beside it.
    await expect(page.getByTestId("no-abstain-note")).toBeVisible();
  });
});

test.describe("@slow fill-mask", () => {
  test.slow();

  // **The RoBERTa half of this pair is the test.** Both models are asked the
  // same question with the same page, and the only thing that differs is the
  // tokenizer's mask literal — so a page that hard-codes `[MASK]` passes the
  // BERT test and fails this one loudly, which is exactly the shape the plan
  // asked for. It fails loudly rather than quietly because `FillMaskPipeline`
  // looks `mask_token_id` up in the ids and raises "Mask token (<mask>) not
  // found in text.", measured before this spec was written.

  test("BERT puts paris in the capital of France, and the probes come in pairs", async ({
    page,
  }) => {
    test.setTimeout(10 * 60 * 1000);
    const model = new ModelPageObject(page);

    await page.goto("/fill-mask");
    await model.load();
    await model.waitForReady();

    // The user's own capitalisation, so the spliced sentence below proves the
    // page is slicing their string rather than rendering the model's decode —
    // BERT is uncased, and its own `sequence` is "the capital of france is…".
    await page.locator("#fm-text").fill("The CAPITAL of France is [MASK].");
    await model.run(/fill the mask/i);

    const scores = page.getByTestId("fill-scores");
    await expect(scores).toBeVisible({ timeout: 120_000 });

    // The word, not the row count. A ranked list of the right length is what a
    // broken tokenizer also produces.
    const rows = scores.getByRole("listitem");
    await expect(rows.first()).toContainText(/paris/i);

    // The sentence is the user's, with one word replaced.
    await expect(page.getByTestId("span-overlay")).toContainText(
      "The CAPITAL of France is paris.",
    );
    await expect(page.getByTestId("span-mark")).toContainText(/paris/i);

    // The bias probe: one press, six prompts, two columns per pair, and the
    // framing sentence travelling with the result rather than near it.
    await model.run(/run the \d+ bias probes/i);
    const probes = page.getByTestId("probes");
    await expect(probes).toBeVisible({ timeout: 180_000 });
    await expect(probes).toContainText(/evidence about the training data/i);

    const man = page.getByTestId("probe-occupation-man").getByRole("listitem");
    const woman = page
      .getByTestId("probe-occupation-woman")
      .getByRole("listitem");
    await expect(man.first()).toBeVisible();
    await expect(woman.first()).toBeVisible();

    // Structural, not a guessed disagreement: both columns are real ranked
    // lists from the same batch. Asserting *which* words differ would pin a
    // property of one checkpoint's corpus, which is the finding the page is
    // for rather than a promise it makes.
    expect(await man.count()).toBeGreaterThan(1);
    expect(await woman.count()).toBeGreaterThan(1);
  });

  test("RoBERTa answers the same question through its own mask token", async ({
    page,
  }) => {
    test.setTimeout(10 * 60 * 1000);
    const model = new ModelPageObject(page);

    await page.goto("/fill-mask");

    // Switch model *before* loading, which is also the path that rewrites the
    // token already in the box. Both halves matter: the box must end up with
    // `<mask>`, and the run must then succeed.
    await page.getByRole("button", { name: /^RoBERTa base/ }).first().click();
    await expect(page.getByTestId("mask-token")).toHaveText("<mask>");

    await page.locator("#fm-text").fill("The capital of France is ");
    await page.getByTestId("insert-mask").click();
    await expect(page.locator("#fm-text")).toHaveValue(
      "The capital of France is <mask>",
    );

    await model.load();
    await model.waitForReady();
    await model.run(/fill the mask/i);

    const rows = page.getByTestId("fill-scores").getByRole("listitem");
    await expect(rows.first()).toBeVisible({ timeout: 120_000 });
    await expect(rows.first()).toContainText(/paris/i);

    // The leading space a byte-level BPE decode leaves on ` Paris` is gone,
    // rather than rendering as a gap and splicing a double space.
    await expect(page.getByTestId("span-mark")).toHaveText(/^Paris$/);

    // The catalogue and the checkpoint agree, so the drift note stays away.
    await expect(page.getByTestId("mask-drift")).toHaveCount(0);
  });
});

// --- Embeddings (§3.7) -------------------------------------------------------

test.describe("@slow embeddings", () => {
  test.slow();

  /**
   * `just fe-e2e-embed`.
   *
   * **The assertion is a spread, not a threshold**, and that is the whole point
   * of the test. Omitting `pooling`/`normalize` — or pooling a CLS-trained
   * checkpoint by the mean — does not fail: it produces vectors whose cosines
   * all sit in a narrow band near 0.9, so every pair looks alike and the page
   * looks like it works. A single-threshold assertion ("the paraphrase scores
   * above 0.5") passes comfortably on exactly those collapsed embeddings. A
   * *gap* between the paraphrase and the unrelated pair does not.
   */
  test("a paraphrase scores far above an unrelated pair", async ({ page }) => {
    test.setTimeout(6 * 60 * 1000);
    const model = new ModelPageObject(page);

    await page.goto("/sentence-similarity");
    await model.load();
    await model.waitForReady();

    await model.run(/^Score all \d+ sample pairs$/);
    await expect(page.getByTestId("similarity")).toBeVisible({
      timeout: 120_000,
    });

    const scoreOf = async (id: string) =>
      Number((await page.getByTestId(`score-${id}`).innerText()).trim());

    const paraphrase = await scoreOf("paraphrase");
    const unrelated = await scoreOf("unrelated");

    // The gap. Measured on all-MiniLM-L6-v2: ~0.62 against ~0.02.
    expect(paraphrase).toBeGreaterThan(0.45);
    expect(unrelated).toBeLessThan(0.30);
    expect(
      paraphrase - unrelated,
      "the spread between a paraphrase and an unrelated pair — a narrow band " +
        "here means the embeddings collapsed, which a threshold would miss",
    ).toBeGreaterThan(0.25);

    // The negation pair is the page's honest half: these models score a
    // sentence and its negation as very similar. Asserted so the caveat on
    // screen stays true of the model actually shipped.
    const negation = await scoreOf("negation");
    expect(negation).toBeGreaterThan(0.75);

    // Truncation re-derives: no second download, no second inference, and the
    // ranking survives the cut — which is the Matryoshka claim, checked rather
    // than asserted in prose.
    await page.getByTestId("truncate-96").click();
    await expect(page.getByTestId("strip-a-dim")).toContainText("96-d");
    // Still unit length on both sides after the cut. A missing renormalisation
    // would leave these below 1 and scale every cosine above by an arbitrary
    // factor.
    await expect(page.getByTestId("strip-a-norm")).toContainText("1.000");
    await expect(page.getByTestId("strip-b-norm")).toContainText("1.000");

    const cutParaphrase = await scoreOf("paraphrase");
    const cutUnrelated = await scoreOf("unrelated");
    expect(
      cutParaphrase,
      "the ranking must survive a truncation, or the control is misleading",
    ).toBeGreaterThan(cutUnrelated);
    // The model stayed ready and no inference ran — the page is still on the
    // one load it started with.
    await expect(model.readyStatus).toBeVisible();
  });

  test("a vector is drawn, unit length, and its truncation is measured", async ({
    page,
  }) => {
    test.setTimeout(6 * 60 * 1000);
    const model = new ModelPageObject(page);

    await page.goto("/text-features");
    // The pooling and width are catalogue facts, on screen before a byte moves.
    await expect(page.getByTestId("embed-pooling")).toContainText("mean");
    await expect(page.getByTestId("embed-dim")).toContainText("384");

    await model.load();
    await model.waitForReady();
    await model.run(/^Embed$/);

    await expect(page.getByTestId("embedding")).toBeVisible({
      timeout: 120_000,
    });
    // `normalize: true` reaching the model, shown rather than claimed.
    await expect(page.getByTestId("vector-strip-norm")).toContainText("1.000");
    await expect(page.getByTestId("vector-strip-dim")).toContainText("384-d");

    // And the cut's real cost, which is a measurement on a non-MRL checkpoint
    // rather than the "it's free" the Matryoshka framing invites.
    await page.getByTestId("truncate-96").click();
    await expect(page.getByTestId("vector-strip-dim")).toContainText("96-d");
    await expect(page.getByTestId("vector-strip-norm")).toContainText("1.000");
    await expect(page.getByTestId("truncate-kept")).toContainText(
      /first 96 of 384 dimensions hold/i,
    );
  });
});

// --- Translation (§3.5) ------------------------------------------------------

test.describe("@slow translation", () => {
  test.slow();

  /**
   * `just fe-e2e-translate`.
   *
   * **Content words, not an exact string.** A beam search is not pinned to one
   * output, and asserting the whole sentence would fail on a decoder update
   * that is not a regression. Asserting only that "some text appeared" would
   * pass on a model translating in the wrong direction, which is this page's
   * own hazard.
   *
   * The second half is the one that earns the minutes: **the reverse pair, on
   * the same page.** A direction control that changed the label without changing
   * the checkpoint keeps translating en→de, so a German output for a German
   * input is what catches it — and nothing in the unit suite can, because it
   * mocks the worker away.
   */
  test("translates one pair, and the reverse pair really reverses", async ({
    page,
  }) => {
    test.setTimeout(20 * 60 * 1000);
    const model = new ModelPageObject(page);

    await page.goto("/translation");

    // The page's own claim, before a byte moves: a direction is a download.
    await expect(page.getByTestId("pair-is-a-download")).toContainText(
      /separate checkpoint/i,
    );

    await model.load();
    await model.waitForReady();

    await page.locator("#tr-text").fill("The meeting is on Thursday.");
    await model.run(/^Translate$/);

    const out = page.getByTestId("translation-text");
    await expect(out).toBeVisible({ timeout: 180_000 });
    // German content words. "Donnerstag" is the one that cannot appear by
    // accident in an English passthrough.
    await expect(out).toContainText(/Donnerstag/i);
    await expect(page.getByTestId("ran-source")).toContainText("Thursday");

    // Now the reverse direction. It is a second checkpoint and a second
    // download, which is the page's whole point — so it needs its own LOAD.
    await page
      .getByRole("button", { name: /^German → English/ })
      .first()
      .click();
    // A SELECT change spends nothing: the LOAD slot must be back to idle
    // rather than the page having started a download on its own.
    await expect(model.loadButton).toBeVisible();

    await model.load();
    await model.waitForReady();

    await page.locator("#tr-text").fill("Die Besprechung ist am Donnerstag.");
    await model.run(/^Translate$/);

    const back = page.getByTestId("translation-text");
    await expect(back).toContainText(/Thursday/i, { timeout: 180_000 });
    // And the label says which direction produced it, so a reader can tell the
    // two results apart.
    await expect(page.getByTestId("output-panel")).toContainText(
      "German → English",
    );
  });
});

// --- Summarization (§3.6) ----------------------------------------------------

test.describe("@slow summarization", () => {
  test.slow();

  /**
   * `just fe-e2e-summarize`.
   *
   * Two assertions and one **measurement**.
   *
   * The assertions: the summary mentions the article's key entity, and it is
   * shorter than the article. "Some text appeared" would pass while the model
   * echoed its input back — which is a real failure mode for a seq2seq whose
   * `min_length` fights its `max_new_tokens`.
   *
   * The measurement is the plan's Phase 0, re-taken where there is a real GPU:
   * how long one summary takes. The gate was settled on a box with **no** GPU —
   * only SwiftShader, whose 83 s per summary says nothing about hardware — so
   * this is where that number comes from, and where a runtime upgrade that
   * changes it becomes visible rather than silent. It is logged, not asserted:
   * a latency threshold in CI is a flake, and the point is to have the figure.
   */
  test("beats nothing yet, but summarizes a known article and says how long it took", async ({
    page,
  }) => {
    test.setTimeout(12 * 60 * 1000);
    const model = new ModelPageObject(page);

    await page.goto("/summarization");

    // The baseline needs no model, so it is on screen before anything is
    // downloaded — the page's best idea, and a correctness requirement rather
    // than decoration.
    await expect(page.getByTestId("baseline-preview")).toContainText(
      /European Space Agency/,
    );
    await expect(model.loadButton).toBeVisible();

    await model.load();
    await model.waitForReady();

    const started = Date.now();
    await model.run(/^Summarize$/);
    await expect(page.getByTestId("summary-text")).toBeVisible({
      timeout: 300_000,
    });
    const elapsed = Date.now() - started;

    const summary = (await page.getByTestId("summary-text").innerText()).trim();
    const article = await page.locator("#sm-text").inputValue();

    // eslint-disable-next-line no-console
    console.log(
      `[phase 0] one summary on ${await model.backend()}: ${elapsed} ms ` +
        `(${summary.split(/\s+/).length} words)`,
    );

    // The article's subject survived into the summary. T5-small is weak — that
    // is this page's point — but a summarizer that loses the subject entirely
    // is broken rather than weak.
    expect(summary.toLowerCase()).toMatch(/rocket|launch|satellite|agency/);
    // And it actually shortened something.
    expect(summary.length).toBeLessThan(article.length);

    // The baseline is rendered beside it, from the same captured article — a
    // metric owes its null model on screen wherever one exists.
    await expect(page.getByTestId("baseline-text")).toContainText(
      /European Space Agency/,
    );
  });
});
