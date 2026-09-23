// The label list and the hypothesis template — everything `/zero-shot-classification`
// derives from what the user typed, kept pure so it can be pinned by unit tests
// and re-derived on the main thread without touching the model.
//
// Two things make this a module rather than three lines in the route:
//
//   **N labels cost N forward passes.** The pipeline runs the model once per
//   label, as an NLI premise–hypothesis pair (`for (const hypothesis of
//   hypotheses) await this.model(inputs)` — zero-shot-classification.js). Ten
//   labels is ten inferences on one press, and a page that does not say so
//   before the click looks like it has hung. The count is a derivation over the
//   label list, so editing labels still spends nothing.
//
//   **A template without `{}` silently collapses the whole run.** The pipeline
//   composes each hypothesis with `template.replace("{}", label)`; a template
//   missing the placeholder produces the *same* hypothesis for every label, so
//   every label gets the same logits and the ranking is whatever order ties
//   resolve in. Nothing throws, nothing looks wrong, and the bar chart is
//   perfectly plausible. `templateProblem()` is the guard, and the route refuses
//   to run on it.

/** The pipeline's own default, and the reason it must be on screen. */
export const DEFAULT_HYPOTHESIS_TEMPLATE = "This example is {}.";

/**
 * The bare template: the label as its own hypothesis, nothing added.
 *
 * Offered as a preset because it is the cheapest way to see that the template
 * reaches the model at all — two templates producing identical scores would
 * mean it does not. The `@slow` spec runs exactly this comparison.
 */
export const BARE_HYPOTHESIS_TEMPLATE = "{}";

/**
 * Past this many labels the pass count stops being a detail. Not a cap: the
 * page states the cost and the user decides, the same way a heavy download is
 * stated rather than forbidden. It only changes how loudly the note is worded.
 */
export const MANY_LABELS = 8;

/**
 * Parse the label editor's text into candidate labels.
 *
 * Newline- or comma-separated, because people paste both. Labels are kept as
 * **bare nouns** (`billing`, not `a billing issue`) so the template composes —
 * the same rule `/zero-shot-image-classification` follows, and for the same
 * reason: the template supplies the article and the sentence frame, so a label
 * that brings its own produces "This example is a a billing issue."
 *
 * Duplicates are dropped case-insensitively. A repeated label is not a second
 * opinion — it is a second forward pass returning the same logits, and two
 * identical rows in the score list read as a bug in the model.
 */
export function parseLabels(raw: string): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const piece of raw.split(/[\n,]/)) {
    const label = piece.trim();
    if (!label) continue;
    const key = label.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(label);
  }
  return out;
}

/** Render the label list back into the editor's text form. */
export function formatLabels(labels: readonly string[]): string {
  return labels.join("\n");
}

/** The exact hypothesis the model will be given for one label. */
export function composeHypothesis(template: string, label: string): string {
  return template.replace("{}", label);
}

/**
 * Why this template cannot be used, or `null` if it can.
 *
 * Returns a sentence for the page to render rather than a boolean, because the
 * failure it prevents is invisible and the user needs to be told what is wrong
 * with what they typed, not merely that something is.
 */
export function templateProblem(template: string): string | null {
  if (template.trim().length === 0) {
    return "The template is empty. Use {} on its own to send each label as the whole hypothesis.";
  }
  if (!template.includes("{}")) {
    return "The template has no {} placeholder, so every label would produce the same hypothesis and the scores would be meaningless.";
  }
  return null;
}

/**
 * What one press of GENERATE will cost, in the model's own units.
 *
 * `passes` is the headline number and it is exact — one forward pass per label,
 * no batching anywhere in the pipeline.
 */
export interface RunCost {
  labels: number;
  passes: number;
  /** True past `MANY_LABELS`, where the wait stops being instant. */
  many: boolean;
}

export function runCost(labels: readonly string[]): RunCost {
  return {
    labels: labels.length,
    passes: labels.length,
    many: labels.length > MANY_LABELS,
  };
}

/**
 * Is this run in the pipeline's single-label branch?
 *
 * `softmaxEach = multi_label || candidate_labels.length === 1`, so **one label
 * is always scored independently** no matter what the toggle says — the softmax
 * across labels has nothing to normalise against. The page says so rather than
 * showing a lone 1.00 that looks like certainty, and `ScoreList` refuses to
 * render a single row for the same reason.
 */
export function scoredIndependently(
  labels: readonly string[],
  multiLabel: boolean,
): boolean {
  return multiLabel || labels.length === 1;
}
