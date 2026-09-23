import { describe, expect, it } from "vitest";

import { truncatedAfter, wordPieceOffsets } from "./offsets";

/**
 * The tokenizations below are real — taken from
 * `Xenova/distilbert-base-cased-distilled-squad`'s own tokenizer, decoded one
 * id at a time — rather than invented. An alignment test written against
 * plausible-looking pieces pins our idea of WordPiece instead of the one the
 * page actually runs.
 */
const EIFFEL =
  "The Eiffel Tower was built by Gustave Eiffel for the 1889 World's Fair in Paris.";
const EIFFEL_PIECES = [
  "The", "E", "##iff", "##el", "Tower", "was", "built", "by", "Gustav", "##e",
  "E", "##iff", "##el", "for", "the", "1889", "World", "'", "s", "Fair", "in",
  "Paris", ".",
];

describe("wordPieceOffsets", () => {
  it("maps every piece back to the characters it came from", () => {
    const offsets = wordPieceOffsets(EIFFEL, EIFFEL_PIECES);
    expect(offsets).not.toBeNull();
    // The property that matters, asserted for every piece rather than spot
    // checked: the range slices back to exactly the piece it was derived from.
    offsets!.forEach((o, i) => {
      const piece = EIFFEL_PIECES[i].replace(/^##/, "");
      expect(EIFFEL.slice(o.start, o.end)).toBe(piece);
    });
  });

  it("rejoins a split word into one contiguous range", () => {
    const offsets = wordPieceOffsets(EIFFEL, EIFFEL_PIECES)!;
    // `E` `##iff` `##el` — three pieces, one word, no gap between them.
    const [e, iff, el] = offsets.slice(1, 4);
    expect(e.end).toBe(iff.start);
    expect(iff.end).toBe(el.start);
    expect(EIFFEL.slice(e.start, el.end)).toBe("Eiffel");
  });

  // The whole reason this module exists rather than an `indexOf` of the
  // pipeline's answer string. WordPiece decoding re-spaces punctuation, so the
  // decoded answer is not a substring of the passage at all.
  it("recovers a hyphenated span the tokenizer's own decode would not", () => {
    const text = "Unlike WebGL, it supports general-purpose compute shaders.";
    const pieces = [
      "Un", "##lik", "##e", "Web", "##GL", ",", "it", "supports", "general",
      "-", "purpose", "compute", "shade", "##rs", ".",
    ];
    const offsets = wordPieceOffsets(text, pieces)!;
    expect(offsets).not.toBeNull();
    const answer = text.slice(offsets[8].start, offsets[13].end);
    expect(answer).toBe("general-purpose compute shaders");
    // The tokenizer would decode those same ids as `general - purpose …`, which
    // does not appear in the passage. A substring search returns -1.
    expect(text.includes("general - purpose compute shaders")).toBe(false);
  });

  it("handles accented text, which the cased normaliser leaves alone", () => {
    const text = "Café owners in Zürich pay 8.1% VAT.";
    const pieces = [
      "Café", "owners", "in", "Z", "##ür", "##ich", "pay", "8", ".", "1", "%",
      "VA", "##T", ".",
    ];
    const offsets = wordPieceOffsets(text, pieces)!;
    expect(text.slice(offsets[0].start, offsets[0].end)).toBe("Café");
    expect(text.slice(offsets[3].start, offsets[5].end)).toBe("Zürich");
  });

  it("collapses runs of whitespace without emitting them", () => {
    const text = "The\n  Eiffel\tTower";
    const offsets = wordPieceOffsets(text, ["The", "E", "##iff", "##el", "Tower"])!;
    expect(offsets).not.toBeNull();
    expect(text.slice(offsets[4].start, offsets[4].end)).toBe("Tower");
    // No range ever covers a whitespace character.
    for (const o of offsets) expect(text.slice(o.start, o.end).trim().length).toBeGreaterThan(0);
  });

  // The honesty requirement: an alignment that cannot be made exactly is
  // reported as unavailable, never approximated. A near-miss highlight reads as
  // a styling bug and is invisible to everyone but a careful reader.
  it("returns null rather than guessing when a piece does not match", () => {
    expect(wordPieceOffsets("The Eiffel Tower", ["The", "[UNK]", "Tower"])).toBeNull();
    // A lowercasing tokenizer against a cased passage — the classic mismatch.
    expect(wordPieceOffsets("The Eiffel Tower", ["the", "eiffel", "tower"])).toBeNull();
    // Pieces that run past the end of the text.
    expect(wordPieceOffsets("The Tower", ["The", "Tower", "of", "London"])).toBeNull();
    // An empty piece consumes nothing and would make the walk ambiguous.
    expect(wordPieceOffsets("The Tower", ["The", "", "Tower"])).toBeNull();
  });

  it("returns an empty list for an empty tokenization", () => {
    expect(wordPieceOffsets("anything", [])).toEqual([]);
  });
});

describe("truncatedAfter", () => {
  it("is false when the alignment reached the end of the passage", () => {
    const offsets = wordPieceOffsets(EIFFEL, EIFFEL_PIECES)!;
    expect(truncatedAfter(EIFFEL, offsets)).toBe(false);
  });

  it("is false when only trailing whitespace is left over", () => {
    const text = `${EIFFEL}\n\n  `;
    expect(truncatedAfter(text, wordPieceOffsets(text, EIFFEL_PIECES)!)).toBe(false);
  });

  // Truncation is silent: the model answers confidently from the half it was
  // given, and a question about the missing half gets a span like any other.
  it("is true when the tokenizer stopped part way through", () => {
    const offsets = wordPieceOffsets(EIFFEL, EIFFEL_PIECES.slice(0, 5))!;
    expect(truncatedAfter(EIFFEL, offsets)).toBe(true);
  });

  it("is true when nothing aligned at all but the passage has text", () => {
    expect(truncatedAfter(EIFFEL, [])).toBe(true);
    expect(truncatedAfter("   ", [])).toBe(false);
  });
});
