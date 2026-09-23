import { describe, expect, it } from "vitest";

import { entitySlot, highlight, redact, type EntitySpan } from "./highlight";

const span = (start: number, end: number, label = "PER", score = 0.99):
  EntitySpan => ({ start, end, label, score });

/** The assertion that catches the token-rebuild bug, wherever it is applied. */
function expectLossless(text: string, spans: EntitySpan[]) {
  const { slices } = highlight(text, spans);
  expect(slices.map((s) => s.text).join("")).toBe(text);
}

describe("highlight", () => {
  const TEXT = "Priya flew to Berlin on Tuesday.";

  it("returns the whole string as one plain slice when there are no spans", () => {
    const { slices, dropped } = highlight(TEXT, []);
    // Not an empty list — that would render as a page that lost the user's text.
    expect(slices).toEqual([{ text: TEXT, span: null }]);
    expect(dropped).toEqual([]);
  });

  it("marks a span at index 0 without emitting an empty head", () => {
    const { slices } = highlight(TEXT, [span(0, 5)]);
    expect(slices[0].text).toBe("Priya");
    expect(slices[0].span).not.toBeNull();
    expect(slices).toHaveLength(2);
  });

  it("marks a span that runs to the very end without dropping the tail", () => {
    const text = "Written by Priya";
    const { slices } = highlight(text, [span(11, 16)]);
    expect(slices.map((s) => s.text)).toEqual(["Written by ", "Priya"]);
  });

  it("keeps the gap between two spans, whitespace and all", () => {
    const { slices } = highlight(TEXT, [span(0, 5), span(14, 20, "LOC")]);
    expect(slices.map((s) => s.text)).toEqual([
      "Priya",
      " flew to ",
      "Berlin",
      " on Tuesday.",
    ]);
  });

  it("handles two adjacent spans with nothing between them", () => {
    const text = "PriyaRaman";
    const { slices } = highlight(text, [span(0, 5), span(5, 10)]);
    expect(slices).toHaveLength(2);
    expect(slices.every((s) => s.span != null)).toBe(true);
  });

  it("sorts spans that arrive out of order", () => {
    const { slices } = highlight(TEXT, [span(14, 20, "LOC"), span(0, 5)]);
    expect(slices[0].text).toBe("Priya");
    expect(slices[2].text).toBe("Berlin");
  });

  // The single assertion that catches the token-rebuild bug: concatenated
  // subwords lose the original whitespace, and the highlight then lands beside
  // the word it means.
  it("reproduces the input exactly, in every arrangement", () => {
    expectLossless(TEXT, []);
    expectLossless(TEXT, [span(0, 5)]);
    expectLossless(TEXT, [span(0, 5), span(14, 20, "LOC")]);
    expectLossless(TEXT, [span(24, 31, "DATE")]);
    expectLossless("  leading and trailing  ", [span(2, 9, "MISC")]);
    expectLossless("tabs\tand\nnewlines", [span(0, 4, "MISC")]);
  });

  it("reports overlapping spans instead of interleaving them", () => {
    const { slices, dropped } = highlight(TEXT, [
      span(0, 10),
      span(5, 12, "ORG"),
    ]);
    expect(dropped).toHaveLength(1);
    expect(dropped[0].label).toBe("ORG");
    // And the text still survives intact.
    expect(slices.map((s) => s.text).join("")).toBe(TEXT);
  });

  it("drops a span that falls outside the text rather than slicing past the end", () => {
    const { slices, dropped } = highlight("short", [span(2, 99)]);
    expect(dropped).toHaveLength(1);
    expect(slices.map((s) => s.text).join("")).toBe("short");
  });

  it("drops an empty or inverted span", () => {
    expect(highlight(TEXT, [span(5, 5)]).dropped).toHaveLength(1);
    expect(highlight(TEXT, [span(9, 3)]).dropped).toHaveLength(1);
  });
});

describe("redact", () => {
  const TEXT = "Priya flew to Berlin for Siemens.";
  const SPANS = [span(0, 5, "PER"), span(14, 20, "LOC"), span(25, 32, "ORG")];

  it("replaces only the selected types, leaving the rest byte for byte", () => {
    expect(redact(TEXT, SPANS, new Set(["PER", "LOC"]))).toBe(
      "[PER] flew to [LOC] for Siemens.",
    );
  });

  it("returns the input unchanged when nothing is selected", () => {
    expect(redact(TEXT, SPANS, new Set())).toBe(TEXT);
  });

  it("removes every entity when every type is selected", () => {
    expect(redact(TEXT, SPANS, new Set(["PER", "LOC", "ORG"]))).toBe(
      "[PER] flew to [LOC] for [ORG].",
    );
  });

  it("takes a custom placeholder", () => {
    expect(
      redact(TEXT, SPANS, new Set(["PER"]), () => "███"),
    ).toBe("███ flew to Berlin for Siemens.");
  });

  it("is unaffected by spans it could not place", () => {
    const withOverlap = [...SPANS, span(2, 8, "ORG")];
    expect(redact(TEXT, withOverlap, new Set(["PER"]))).toBe(
      "[PER] flew to Berlin for Siemens.",
    );
  });
});

describe("entitySlot", () => {
  it("keys on the entity type, never on the order spans arrived in", () => {
    // Colour follows the entity: filtering the list must not repaint the
    // survivors, which is only true if the slot is a function of the label.
    expect(entitySlot("PER")).toBe(entitySlot("PER"));
    expect(entitySlot("LOC")).not.toBe(entitySlot("PER"));
  });

  it("gives MISC and DATE the same slot — they cannot co-occur", () => {
    // Different checkpoints, and only one model is live at a time, so they
    // share a hue rather than forcing a fifth that no validated subset of the
    // palette supports.
    expect(entitySlot("MISC")).toBe(entitySlot("DATE"));
  });

  it("is case-insensitive and accepts the common long forms", () => {
    expect(entitySlot("per")).toBe(entitySlot("PERSON"));
    expect(entitySlot("org")).toBe(entitySlot("ORGANIZATION"));
  });

  it("returns null for an unrecognised type rather than inventing a hue", () => {
    expect(entitySlot("PRODUCT")).toBeNull();
  });

  it("only ever returns one of the four theme slots", () => {
    for (const label of ["PER", "ORG", "LOC", "MISC", "DATE"]) {
      expect([1, 2, 3, 4]).toContain(entitySlot(label));
    }
  });
});
