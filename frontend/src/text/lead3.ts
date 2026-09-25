// The lead-3 baseline: the first three sentences of the text, unchanged.
//
// It ships as a *toggle on the summarization page* rather than as a footnote,
// because it is the most useful thing on that page. On a news article — the
// genre every one of these checkpoints was fine-tuned on — three sentences of
// `text.split(/(?<=[.!?])\s/).slice(0, 3)` are a genuinely hard baseline, and
// watching a 284 MB neural summarizer fail to beat them is the lesson. A page
// that shows only the model's output invites the opposite conclusion by
// omission.
//
// **The sentence splitter is bad on purpose, and the page says so.** It splits
// on `.`/`!`/`?` followed by whitespace, so "Dr. Smith" and "U.S. Army" split
// wrongly. A real splitter is a model or a rule pile; either would be more code
// than the summarizer's own hook, and neither is this page's subject. What
// matters is the property below.
//
// **The output is a slice of the input, never a rebuild.** Joining split pieces
// back together loses whichever whitespace the split consumed, so a baseline
// assembled that way quietly differs from the article it claims to be quoting —
// the same rule `highlight()` and `spliceFill` follow one page over. The
// implementation returns character *offsets* and slices; the test asserts the
// slice is a substring of the input.

/** Where one sentence sits in the source text. */
export interface Sentence {
  start: number;
  end: number;
  text: string;
}

/**
 * Split on sentence-ending punctuation followed by whitespace.
 *
 * Offsets are into the original string, and every returned `text` is
 * `source.slice(start, end)`. Trailing text with no final punctuation is still
 * a sentence — an article that ends mid-thought is common, and dropping its
 * last line would silently shorten the baseline.
 */
export function splitSentences(text: string): Sentence[] {
  const out: Sentence[] = [];
  let start = 0;
  // Walk rather than `split`, so the offsets are exact and nothing is rebuilt.
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch !== "." && ch !== "!" && ch !== "?") continue;
    // Consume a run of terminators ("…!?" or "?!").
    let end = i + 1;
    while (end < text.length && ".!?".includes(text[end])) end++;
    // A terminator only ends a sentence when whitespace follows it (or the
    // string does). "3.5" and "e.g." survive; "Dr. Smith" does not, which is
    // the documented limitation.
    if (end < text.length && !/\s/.test(text[end])) {
      i = end - 1;
      continue;
    }
    const piece = text.slice(start, end).trim();
    if (piece.length > 0) {
      const lead = text.slice(start, end).length - text.slice(start, end).trimStart().length;
      out.push({ start: start + lead, end, text: piece });
    }
    start = end;
    i = end - 1;
  }

  const tail = text.slice(start).trim();
  if (tail.length > 0) {
    const lead = text.slice(start).length - text.slice(start).trimStart().length;
    out.push({ start: start + lead, end: text.length, text: tail });
  }
  return out;
}

/**
 * The first `n` sentences, as one string.
 *
 * Fewer than `n` sentences returns what there is rather than padding or
 * throwing: a two-sentence input is a legitimate thing to summarize, and the
 * baseline for it is the whole thing — which is itself the answer to "is this
 * worth summarizing".
 */
export function lead(text: string, n = 3): string {
  const sentences = splitSentences(text);
  if (sentences.length === 0) return "";
  const taken = sentences.slice(0, Math.max(0, n));
  if (taken.length === 0) return "";
  // One slice of the original across the whole span, so the whitespace between
  // the sentences is the article's own rather than a space we invented.
  return text.slice(taken[0].start, taken[taken.length - 1].end).trim();
}

/** How many sentences `text` has, for the page's "of N" line. */
export function sentenceCount(text: string): number {
  return splitSentences(text).length;
}

/** Rough word count — the page compares lengths, so it needs both sides' units. */
export function wordCount(text: string): number {
  const trimmed = text.trim();
  return trimmed.length === 0 ? 0 : trimmed.split(/\s+/).length;
}
