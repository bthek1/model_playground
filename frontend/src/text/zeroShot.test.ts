import { describe, expect, it } from "vitest";

import {
  BARE_HYPOTHESIS_TEMPLATE,
  composeHypothesis,
  DEFAULT_HYPOTHESIS_TEMPLATE,
  formatLabels,
  MANY_LABELS,
  parseLabels,
  runCost,
  scoredIndependently,
  templateProblem,
} from "./zeroShot";

describe("parseLabels", () => {
  it("takes one label per line", () => {
    expect(parseLabels("billing\noutage\nfeature request")).toEqual([
      "billing",
      "outage",
      "feature request",
    ]);
  });

  it("takes commas too, because people paste both", () => {
    expect(parseLabels("billing, outage,feature request")).toEqual([
      "billing",
      "outage",
      "feature request",
    ]);
  });

  it("drops blank lines and trailing separators rather than sending empty labels", () => {
    // An empty label composes to "This example is ." — a hypothesis the model
    // will happily score, producing a row with no name and a real number
    // beside it.
    expect(parseLabels("billing\n\n  \noutage,")).toEqual([
      "billing",
      "outage",
    ]);
  });

  it("drops a repeated label, case-insensitively", () => {
    // A duplicate is not a second opinion: it is a second forward pass
    // returning the same logits, so it costs a full inference to print the
    // same row twice — which reads as a bug in the model rather than in the
    // label list.
    expect(parseLabels("billing\nBilling\n  billing  ")).toEqual(["billing"]);
  });

  it("keeps the order the user typed", () => {
    expect(parseLabels("zebra\napple")).toEqual(["zebra", "apple"]);
  });

  it("round-trips through the editor's text form", () => {
    const labels = ["billing", "outage", "feature request"];
    expect(parseLabels(formatLabels(labels))).toEqual(labels);
  });
});

describe("composeHypothesis", () => {
  it("puts the label where the placeholder is", () => {
    expect(composeHypothesis(DEFAULT_HYPOTHESIS_TEMPLATE, "urgent")).toBe(
      "This example is urgent.",
    );
  });

  it("supports the bare template, where the label is the whole hypothesis", () => {
    expect(composeHypothesis(BARE_HYPOTHESIS_TEMPLATE, "urgent")).toBe(
      "urgent",
    );
  });

  it("replaces the first placeholder only, as the pipeline does", () => {
    // Pinned against the runtime rather than chosen: the pipeline calls
    // `hypothesis_template.replace('{}', x)`, which is a first-match replace.
    // A template with two placeholders is a user error, but the page's preview
    // must show what will really be sent, not a tidier version of it.
    expect(composeHypothesis("{} and {}", "urgent")).toBe("urgent and {}");
  });
});

describe("templateProblem", () => {
  it("passes the two templates the page offers", () => {
    expect(templateProblem(DEFAULT_HYPOTHESIS_TEMPLATE)).toBeNull();
    expect(templateProblem(BARE_HYPOTHESIS_TEMPLATE)).toBeNull();
  });

  it("rejects a template with no placeholder", () => {
    // The failure this exists for is silent: without `{}` every label produces
    // the *same* hypothesis, so every label gets the same logits and the
    // ranking is whatever order the ties resolve in. Nothing throws and the
    // bar chart looks perfectly ordinary.
    expect(templateProblem("This example is about money.")).toMatch(/\{\}/);
  });

  it("rejects an empty template and says what to type instead", () => {
    expect(templateProblem("   ")).toMatch(/\{\}/);
  });
});

describe("runCost", () => {
  it("is one forward pass per label — the page's whole cost model", () => {
    // Not an approximation. `zero-shot-classification.js` loops over the
    // hypotheses and awaits `this.model(inputs)` inside the loop; there is no
    // batching anywhere in the pipeline.
    expect(runCost(["a", "b", "c"]).passes).toBe(3);
    expect(runCost([]).passes).toBe(0);
  });

  it("flags a label list long enough for the wait to be noticeable", () => {
    expect(runCost(new Array(MANY_LABELS).fill("x").map((_, i) => `l${i}`)).many).toBe(
      false,
    );
    expect(
      runCost(new Array(MANY_LABELS + 1).fill("x").map((_, i) => `l${i}`)).many,
    ).toBe(true);
  });
});

describe("scoredIndependently", () => {
  it("follows the toggle for two or more labels", () => {
    expect(scoredIndependently(["a", "b"], false)).toBe(false);
    expect(scoredIndependently(["a", "b"], true)).toBe(true);
  });

  it("is always true for one label, whatever the toggle says", () => {
    // `softmaxEach = multi_label || candidate_labels.length === 1` in the
    // pipeline: a softmax across one label has nothing to normalise against
    // and would print 1.00 no matter what the model thought. The page says so
    // rather than rendering that as certainty.
    expect(scoredIndependently(["a"], false)).toBe(true);
  });
});
