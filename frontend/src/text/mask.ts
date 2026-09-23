// The mask token, as a value rather than a literal — `/fill-mask`'s whole
// correctness surface, and the NLP category's third shared primitive.
//
// **Never write `[MASK]` in code.** A masked-language-model page ships models
// from several tokenizer families, and the literal differs between them:
// BERT, DistilBERT and ModernBERT use `[MASK]`, RoBERTa uses `<mask>`. Every
// function here takes the token as an argument and every call site reads it
// from the selected catalogue entry — which the engine then reconciles against
// the *loaded tokenizer's* own `mask_token`, so a catalogue entry that has
// drifted cannot turn into a wrong answer.
//
// Two things were measured against the real pipeline before this module was
// written, and both shaped it:
//
//   A **wrong** mask literal throws rather than lying. `FillMaskPipeline`
//   looks up `mask_token_id` in the token ids and raises "Mask token (<mask>)
//   not found in text." — so the failure is loud. That is better than the
//   silent fluent-but-wrong answer a hand-rolled MLM would give, and it is
//   still a failure the user did nothing to deserve, which is why the mask is
//   inserted by a button and rewritten on a model change rather than typed.
//
//   A **second** mask is silently ignored. The pipeline takes `findIndex` over
//   the ids, fills the first mask and drops the rest: "The [MASK] of France is
//   [MASK]." comes back as "the border of france is." — one filling, no error,
//   a sentence that has quietly lost a word. That is the real silent failure on
//   this page, and it is why `countMasks` exists and why the route refuses to
//   run on anything but exactly one.

/** A half-open character range: `[start, end)` into the string it came from. */
export interface MaskSpan {
  start: number;
  end: number;
}

/**
 * Every occurrence of `mask` in `text`, in order.
 *
 * `indexOf` rather than a regex, because the token is data: `[MASK]` is a
 * character class if it ever reaches a pattern by accident.
 */
export function findMasks(text: string, mask: string): MaskSpan[] {
  if (!mask) return [];
  const spans: MaskSpan[] = [];
  let from = 0;
  for (;;) {
    const at = text.indexOf(mask, from);
    if (at === -1) return spans;
    spans.push({ start: at, end: at + mask.length });
    from = at + mask.length;
  }
}

/** How many masks the text holds. Exactly one is the only runnable count. */
export function countMasks(text: string, mask: string): number {
  return findMasks(text, mask).length;
}

/**
 * Insert `mask` over the selection `[start, end)`, with the spacing a person
 * would have typed.
 *
 * The spacing is not cosmetic. `is[MASK].` tokenizes differently from
 * `is [MASK].` on a WordPiece vocabulary, so a button that jams the token
 * against the previous word changes what the model is asked. A trailing space
 * is added only before a word character, so inserting in front of a full stop
 * gives `is [MASK].` rather than `is [MASK] .`.
 *
 * Returns the caret position after the inserted token so the caller can keep
 * the textarea usable.
 */
export function insertMask(
  text: string,
  mask: string,
  start: number = text.length,
  end: number = start,
): { text: string; caret: number } {
  const head = text.slice(0, start);
  const tail = text.slice(end);
  const before = head.length > 0 && !/\s$/.test(head) ? " " : "";
  const after = /^[A-Za-z0-9]/.test(tail) ? " " : "";
  return {
    text: `${head}${before}${mask}${after}${tail}`,
    caret: head.length + before.length + mask.length,
  };
}

/**
 * Rewrite every known mask literal in `text` to `to`.
 *
 * This is what a model change does to text already in the box. The alternative
 * — leaving `[MASK]` in a box now pointed at RoBERTa — is the page's own bug
 * with the user holding it, and doing nothing at all would be silent, which
 * this repo does not do. Rewriting is chosen over refusing because the user's
 * sentence is the part worth keeping; the token is punctuation they did not
 * type in the first place.
 *
 * One pass over an alternation, so a swap cannot convert a token twice
 * (`<mask>` → `[MASK]` → `<mask>`), and `known` is derived from the catalogue
 * rather than listed here — a fifth tokenizer family arrives as a catalogue
 * entry, not as an edit to this file.
 */
export function retargetMasks(
  text: string,
  known: readonly string[],
  to: string,
): string {
  const others = known.filter((t) => t && t !== to);
  if (others.length === 0) return text;
  const pattern = new RegExp(others.map(escapeRegExp).join("|"), "g");
  return text.replace(pattern, to);
}

/**
 * Expand a sample or probe template — `{}` marks where the mask belongs.
 *
 * `split`/`join` rather than `replaceAll`, which this project's `lib` target
 * does not carry.
 */
export function expandTemplate(template: string, mask: string): string {
  return template.split("{}").join(mask);
}

/**
 * Put a filling where the mask was, and say which characters it now occupies.
 *
 * **The original string is spliced, never rebuilt from the model's tokens.**
 * The pipeline also returns a `sequence` per candidate, and it is tempting —
 * but it is `tokenizer.decode(...)` output: an uncased BERT hands back
 * "the capital of france is paris.", so rendering it would silently lowercase
 * the user's sentence and lose whatever whitespace the tokenizer normalised.
 * Same rule as `text/highlight.ts`, which is what draws the returned span.
 */
export function spliceFill(
  text: string,
  span: MaskSpan,
  fill: string,
): { text: string; span: MaskSpan } {
  const head = text.slice(0, span.start);
  return {
    text: `${head}${fill}${text.slice(span.end)}`,
    span: { start: span.start, end: span.start + fill.length },
  };
}

/**
 * Tidy one candidate token for display.
 *
 * Byte-level BPE tokenizers (RoBERTa, ModernBERT) decode a word-initial token
 * with its leading space still attached — ` Paris`, not `Paris` — so an
 * untrimmed candidate renders as a stray gap in the score list and splices a
 * double space into the sentence.
 */
export function cleanFill(token: string): string {
  return token.trim();
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
