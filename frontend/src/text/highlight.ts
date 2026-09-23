// Turning a model's character offsets into renderable slices of the **original
// string** — the NLP category's second shared primitive, used by
// `/token-classification`, `/question-answering` and `/fill-mask`.
//
// **The one rule: slice the input, never rebuild it from tokens.** A tokenizer's
// pieces do not carry the whitespace between them, so concatenating them back
// together produces text that is a character or two out of step with what the
// user typed — and the highlight then lands beside the word it means, which
// reads as a CSS problem rather than as a wrong offset. Every function here
// takes `{ start, end }` in **characters**, which is what a fast tokenizer
// returns, and every slice is a substring of the input.
//
// Everything is pure, so it is cheap to pin: the assertion that matters is that
// the concatenation of every slice equals the input **exactly**, whitespace
// included.

/** One labelled range, in character offsets into the original string. */
export interface EntitySpan {
  /** Inclusive character index. */
  start: number;
  /** Exclusive character index. */
  end: number;
  /** The entity type — `PER`, `LOC`, `ORG`, … */
  label: string;
  /** Model confidence, 0–1. */
  score: number;
}

/** A run of the original string: plain text, or text carrying one span. */
export interface Slice {
  text: string;
  /** Null for the unlabelled runs between spans. */
  span: EntitySpan | null;
}

export interface HighlightResult {
  slices: Slice[];
  /**
   * Spans that could not be rendered: out of range, empty, or overlapping one
   * already placed.
   *
   * They are **reported rather than silently interleaved**. Two spans claiming
   * the same characters cannot both be drawn, and picking one quietly is how a
   * page ends up showing a confident highlight over a range no model proposed.
   * `aggregation_strategy: "simple"` does not produce overlaps, so anything
   * here is a real bug — and the component says so on screen instead of
   * throwing in the middle of a render.
   */
  dropped: EntitySpan[];
}

/**
 * Split `text` into alternating plain and labelled slices.
 *
 * Spans are sorted by `start`; the tail after the last span is always emitted,
 * and so is the head before the first one. A call with no spans returns the
 * whole string as a single plain slice — not an empty list, which would render
 * as a page that lost the user's text.
 */
export function highlight(
  text: string,
  spans: readonly EntitySpan[],
): HighlightResult {
  const slices: Slice[] = [];
  const dropped: EntitySpan[] = [];

  const ordered = [...spans].sort((a, b) => a.start - b.start || a.end - b.end);

  let cursor = 0;
  for (const span of ordered) {
    const invalid =
      !Number.isInteger(span.start) ||
      !Number.isInteger(span.end) ||
      span.start < 0 ||
      span.end > text.length ||
      span.end <= span.start;
    // `span.start < cursor` is the overlap test, and it works precisely because
    // the list is sorted: anything starting before the last span ended cannot
    // be placed without splitting one of them.
    if (invalid || span.start < cursor) {
      dropped.push(span);
      continue;
    }
    if (span.start > cursor) {
      slices.push({ text: text.slice(cursor, span.start), span: null });
    }
    slices.push({ text: text.slice(span.start, span.end), span });
    cursor = span.end;
  }

  // The tail. Also the whole string when there were no usable spans at all.
  if (cursor < text.length) {
    slices.push({ text: text.slice(cursor), span: null });
  }

  return { slices, dropped };
}

/**
 * Replace the selected spans with a placeholder, leaving everything else byte
 * for byte as the user wrote it.
 *
 * A pure derivation over spans already in hand, so toggling redaction — or
 * changing which types are redacted — re-derives on the main thread and costs
 * nothing. Same rule as `/vad`'s threshold and detection's confidence floor:
 * only GENERATE spends.
 */
export function redact(
  text: string,
  spans: readonly EntitySpan[],
  /** Entity types to remove. A type absent from this set is left alone. */
  types: ReadonlySet<string>,
  /** `[PER]`, `[LOC]`, … by default. */
  placeholder: (span: EntitySpan) => string = (s) => `[${s.label}]`,
): string {
  const { slices } = highlight(text, spans);
  return slices
    .map((s) =>
      s.span && types.has(s.span.label) ? placeholder(s.span) : s.text,
    )
    .join("");
}

/**
 * Which of the four `--entity-*` theme slots a label paints with.
 *
 * Keyed on the **entity type**, never on the order the spans came back in:
 * colour follows the entity, so filtering the list must not repaint the
 * survivors.
 *
 * Four slots for five names, because `MISC` and `DATE` belong to different
 * checkpoints and only one model is ever live — they can never appear together,
 * so they share a slot rather than forcing a fifth hue that no validated subset
 * of the palette supports. Anything unrecognised gets `null` and a neutral
 * treatment: a generated hue would be off-palette and unvalidated.
 */
export function entitySlot(label: string): 1 | 2 | 3 | 4 | null {
  switch (label.toUpperCase()) {
    case "PER":
    case "PERSON":
      return 1;
    case "ORG":
    case "ORGANIZATION":
      return 2;
    case "LOC":
    case "LOCATION":
    case "GPE":
      return 3;
    case "MISC":
    case "DATE":
      return 4;
    // The two single-span pages share slot 1, on the same argument that lets
    // MISC and DATE share slot 4: only one model is ever live, so an answer, a
    // filled mask and a person can never appear together.
    //
    // `ANSWER` is `/question-answering`'s extracted span and `FILL` is the word
    // `/fill-mask` put where the mask was — neither is an entity, and neither
    // has a second span to be confused with. Leaving them unrecognised instead
    // would paint the model's own result in the neutral "this is not an entity"
    // grey, which on those pages is the opposite of what is being shown.
    case "ANSWER":
    case "FILL":
      return 1;
    default:
      return null;
  }
}
