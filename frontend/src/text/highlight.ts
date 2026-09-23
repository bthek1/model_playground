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

// --- Recovering the offsets the runtime does not give us ---------------------
//
// **Transformers.js 4.2.0's `token-classification` pipeline returns no
// character offsets.** Each entity carries `entity_group`, `score` and `word`,
// and that is all — the pipeline ships with the work unwritten (`// TODO add
// start and end?`) and declares `start`/`end` as *optional*, so a caller reading
// `result.start` type-checks cleanly and gets `undefined` at runtime. Measured
// against 4.2.0 on `Xenova/bert-base-NER`, not inferred from the types:
//
//   [{ entity_group: "PER", score: 0.998, word: "P" },
//    { entity_group: "PER", score: 0.983, word: "##riya Raman" },
//    { entity_group: "LOC", score: 0.998, word: "Wellington" }, …]
//
// Without offsets every span is dropped as invalid and the overlay renders the
// user's text with nothing marked — a page that looks like a model that found
// nothing. Only a real in-browser run catches it: the unit suite mocks the
// pipeline and hands back the offsets the real one never produces.
//
// So the page has to recover them, and `highlight.ts`'s standing rule says how
// it must end: the result is a genuine `[start, end)` into the string the user
// typed, and the overlay still slices *that* string. Nothing is rebuilt from
// tokens; the pieces are only used to **find** where they came from.
//
// `offsets.ts` solves the adjacent problem for a caller that owns its
// tokenizer — a complete, contiguous WordPiece list. This one is for a caller
// reading the *pipeline's* output, which is already aggregated and skips every
// token the model labelled `O`, so the pieces arrive with arbitrary gaps
// between them and a forward walk is the only thing that can place them.

/** One entity as the pipeline actually returns it: a word, and no offsets. */
export interface LocatableEntity {
  word: string;
  label: string;
  score: number;
}

/** Strip the WordPiece continuation marker; the text itself never carries it. */
function bare(word: string): string {
  return word.startsWith("##") ? word.slice(2) : word;
}

/**
 * Escape a string for use inside a `RegExp`.
 *
 * `RegExp.escape` is ES2025 and past this project's lib, and a hand-rolled
 * character class is exactly the kind of thing that silently mis-handles one
 * metacharacter — an entity containing `(` or `+` is not rare in newswire.
 */
function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Place each entity in `text`, returning real character spans.
 *
 * A **forward walk**: each entity is searched for from where the previous one
 * ended, never from the start. That is what makes a passage naming the same
 * person twice mark both occurrences rather than the first one twice — the
 * failure a bare `indexOf` has, and it looks like a working page.
 *
 * Two matches are tried, in order:
 *
 *   1. the word verbatim, which is the overwhelmingly common case;
 *   2. the same word with whitespace made flexible, because WordPiece decoding
 *      re-spaces punctuation — `Foo-Bar` decodes as `Foo - Bar`, and an exact
 *      search for the decoded form returns -1 on text that plainly contains the
 *      entity.
 *
 * An entity that matches neither is **dropped and reported**, never
 * approximated. A highlight that lands beside the word it means reads as a
 * styling bug rather than a wrong offset, and is indistinguishable from a
 * working page to everyone except someone reading carefully.
 */
export function locateEntities(
  text: string,
  entities: readonly LocatableEntity[],
): { spans: EntitySpan[]; unplaced: LocatableEntity[] } {
  const spans: EntitySpan[] = [];
  const unplaced: LocatableEntity[] = [];
  let cursor = 0;

  for (const entity of entities) {
    const needle = bare(entity.word).trim();
    if (needle.length === 0) {
      unplaced.push(entity);
      continue;
    }

    let start = text.indexOf(needle, cursor);
    let end = start + needle.length;

    if (start < 0) {
      // Re-spaced punctuation: match the same characters with any amount of
      // whitespace (including none) wherever the decoded form had a space.
      const flexible = new RegExp(
        needle.split(/\s+/).map(escapeRe).join("\\s*"),
      );
      const match = flexible.exec(text.slice(cursor));
      if (!match) {
        unplaced.push(entity);
        continue;
      }
      start = cursor + match.index;
      end = start + match[0].length;
    }

    spans.push({ start, end, label: entity.label, score: entity.score });
    cursor = end;
  }

  return { spans: mergeAdjacent(spans), unplaced };
}

/**
 * Join spans of the same label that touch.
 *
 * `aggregation_strategy: "simple"` does not always merge a word it split:
 * "Priya Raman" comes back as `P` + `##riya Raman`, two PER groups, which would
 * paint as two marks across one name. They are contiguous once placed, so the
 * fix belongs here rather than in the renderer — a merged span is also what the
 * redaction step should remove, and two half-names are not.
 *
 * Only *touching* spans merge. Two separate mentions of the same person have
 * text between them and stay separate, which is the whole point of marking them.
 */
function mergeAdjacent(spans: readonly EntitySpan[]): EntitySpan[] {
  const merged: EntitySpan[] = [];
  for (const span of spans) {
    const previous = merged[merged.length - 1];
    if (previous && previous.label === span.label && previous.end === span.start) {
      merged[merged.length - 1] = {
        start: previous.start,
        end: span.end,
        label: span.label,
        // The weaker of the two: a merged span is only as trustworthy as its
        // least confident half, and averaging would flatter it.
        score: Math.min(previous.score, span.score),
      };
      continue;
    }
    merged.push(span);
  }
  return merged;
}
