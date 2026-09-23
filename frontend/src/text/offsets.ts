// Character offsets for a WordPiece tokenization — the missing half of
// `highlight.ts`, and the reason `/question-answering` owns an engine.
//
// **Transformers.js does not produce character offsets.** Its tokenizers return
// `input_ids` and `attention_mask` and nothing else; there is no
// `return_offsets_mapping`, and both pipelines that would carry offsets ship
// with the work unwritten — `question-answering` pushes `{ answer, score }` past
// a literal `// TODO add start and end?`, and `token-classification` has the
// same TODO where its `start`/`end` would be filled in. Both declare the fields
// as *optional* in their types, so a caller reading `result.start` type-checks
// cleanly and receives `undefined` at runtime. That is measured against 4.2.0,
// not inferred from the types.
//
// So a page that wants to mark the model's answer **in the user's own text**
// has to rebuild the alignment, and `highlight.ts`'s standing rule says how it
// must end: slice the original string by character offset, never rebuild the
// text from tokens. This module is what makes that possible — it maps each
// token back to the range of the source it came from, so the span handed to
// `highlight()` is a genuine `[start, end)` into the string the user typed.
//
// **Why a substring search is not an acceptable shortcut.** The obvious cheat
// is `context.indexOf(answer)`, and it fails two ways that both look like
// working pages. It finds the *first* occurrence, which on a passage that says
// a name twice highlights the wrong one; and the decoded answer is frequently
// not in the passage at all, because WordPiece decoding re-spaces punctuation.
// Measured on this page's own sample: the model answers "general-purpose
// compute shaders", the tokenizer decodes it as `general - purpose compute
// shaders`, and `indexOf` returns -1. Slicing [213, 244) returns the passage's
// own characters, hyphen intact.
//
// Pure, so it is cheap to pin: the assertion that matters is that every emitted
// range slices back to exactly the piece it was derived from.

/** One token's half-open character range into the source string. */
export interface CharOffset {
  /** Inclusive character index. */
  start: number;
  /** Exclusive character index. */
  end: number;
}

/** The WordPiece continuation marker — a piece that joins the previous word. */
const CONTINUATION = "##";

/**
 * Map a WordPiece tokenization back onto the string it came from.
 *
 * `pieces` are the tokenizer's own token strings in order — `Tower`, `E`,
 * `##iff`, `##el` — covering `text` and nothing else, so the caller slices the
 * special tokens and the question half off first. Returns one range per piece.
 *
 * **It returns `null` rather than guessing.** The walk is exact: skip
 * whitespace, then require the next characters of `text` to equal the piece
 * verbatim. Anything that breaks that — an `[UNK]`, a lowercasing tokenizer, an
 * accent-stripping normaliser, a zero-width character the normaliser dropped —
 * fails the comparison and ends the whole alignment. A partial or approximate
 * answer here is the exact failure this module exists to prevent: a highlight
 * that lands beside the word it means reads as a styling bug rather than as a
 * wrong offset, and it is indistinguishable from a working page to everyone
 * except the person reading the passage carefully. The caller's job is to say
 * "the offsets are unavailable" instead.
 *
 * This is safe for the cased WordPiece checkpoints the NLP catalogue uses
 * (`do_lower_case: false`, `strip_accents: null`), which is why the page that
 * depends on it pins its checkpoint's tokenizer config rather than assuming it.
 */
export function wordPieceOffsets(
  text: string,
  pieces: readonly string[],
): CharOffset[] | null {
  const offsets: CharOffset[] = [];
  let cursor = 0;

  for (const piece of pieces) {
    const word = piece.startsWith(CONTINUATION)
      ? piece.slice(CONTINUATION.length)
      : piece;
    // An empty piece consumes nothing and would make the walk ambiguous — a
    // real tokenization has none, so treat it as a broken alignment.
    if (word.length === 0) return null;

    while (cursor < text.length && isSpace(text[cursor])) cursor += 1;
    if (text.slice(cursor, cursor + word.length) !== word) return null;

    offsets.push({ start: cursor, end: cursor + word.length });
    cursor += word.length;
  }

  return offsets;
}

/**
 * Whether the alignment stopped short of the end of the passage — i.e. the
 * tokenizer truncated it and the model never read the tail.
 *
 * Derived from the offsets already in hand rather than from a second
 * tokenization, and it matters because truncation is silent: the model answers
 * confidently from the half it was given, and a question about the missing half
 * gets the same confident span as any other unanswerable question.
 */
export function truncatedAfter(
  text: string,
  offsets: readonly CharOffset[],
): boolean {
  const last = offsets[offsets.length - 1];
  if (!last) return text.trim().length > 0;
  return text.slice(last.end).trim().length > 0;
}

/** Whitespace as the BERT normaliser treats it: split on it, never emit it. */
function isSpace(ch: string): boolean {
  return /\s/.test(ch);
}
