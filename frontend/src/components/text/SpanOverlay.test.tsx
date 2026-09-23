import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { highlight, type EntitySpan } from "@/text/highlight";

import { SpanOverlay } from "./SpanOverlay";

const TEXT = "Priya flew to Berlin for Siemens.";
const SPANS: EntitySpan[] = [
  { start: 0, end: 5, label: "PER", score: 0.99 },
  { start: 14, end: 20, label: "LOC", score: 0.97 },
  { start: 25, end: 32, label: "ORG", score: 0.94 },
];

const marked = (spans = SPANS, text = TEXT) => highlight(text, spans);

describe("SpanOverlay", () => {
  it("renders one mark per span", () => {
    render(<SpanOverlay result={marked()} />);
    expect(screen.getAllByTestId("span-mark")).toHaveLength(3);
  });

  it("renders the user's text intact, whitespace and all", () => {
    const { container } = render(<SpanOverlay result={marked()} />);
    // The type tags are extra text nodes, so compare against the marked-up
    // string with them stripped — what matters is that no *original* character
    // was lost or moved.
    const rendered = container.textContent ?? "";
    for (const chunk of ["Priya", " flew to ", "Berlin", " for ", "Siemens", "."]) {
      expect(rendered).toContain(chunk);
    }
  });

  // Colour alone would leave the four entity types indistinguishable for a
  // colour-blind reader — the validated palette's worst pair sits in the band
  // that is only legal *with* a secondary encoding. The type is that encoding,
  // so it is on screen rather than behind a hover.
  it("writes each entity's type out as text, not only as colour", () => {
    render(<SpanOverlay result={marked()} />);
    const marks = screen.getAllByTestId("span-mark");
    expect(marks[0]).toHaveTextContent("PER");
    expect(marks[1]).toHaveTextContent("LOC");
    expect(marks[2]).toHaveTextContent("ORG");
  });

  it("gives each type a different theme slot, and never a raw hex", () => {
    render(<SpanOverlay result={marked()} />);
    const classes = screen
      .getAllByTestId("span-mark")
      .map((m) => m.className);
    expect(classes[0]).toContain("entity-1");
    expect(classes[1]).toContain("entity-3");
    expect(classes[2]).toContain("entity-2");
    for (const c of classes) expect(c).not.toMatch(/#[0-9a-f]{3,6}/i);
  });

  it("falls back to a neutral treatment for an unknown type", () => {
    render(
      <SpanOverlay
        result={marked([{ start: 0, end: 5, label: "PRODUCT", score: 0.5 }])}
      />,
    );
    // Not a generated hue: off-palette colour is unvalidated colour.
    expect(screen.getByTestId("span-mark").className).toContain("muted");
  });

  // The type tag is an inline sibling of the entity text, so anything reading
  // the mark's own text content gets "Priya PER" — an offset assertion against
  // that is vacuous. The text node carries its own hook for exactly this.
  it("exposes the entity text separately from its type tag", () => {
    render(<SpanOverlay result={marked()} />);
    const texts = screen.getAllByTestId("span-text").map((n) => n.textContent);
    expect(texts).toEqual(["Priya", "Berlin", "Siemens"]);
  });

  it("makes the score reachable without a pointer", () => {
    render(<SpanOverlay result={marked()} />);
    const mark = screen.getAllByTestId("span-mark")[0];
    expect(mark).toHaveAttribute("tabindex", "0");
    expect(mark).toHaveAttribute("title", expect.stringContaining("0.99"));
  });

  it("renders the selected types as placeholders when redacting", () => {
    render(<SpanOverlay result={marked()} redacted={new Set(["PER", "LOC"])} />);
    const hidden = screen.getAllByTestId("span-redacted");
    expect(hidden).toHaveLength(2);
    expect(hidden[0]).toHaveTextContent("[PER]");
    // The unselected type is still shown normally.
    expect(screen.getAllByTestId("span-mark")).toHaveLength(1);
  });

  it("says so when a span could not be drawn, rather than hiding it", () => {
    const result = highlight(TEXT, [
      { start: 0, end: 10, label: "PER", score: 0.9 },
      { start: 5, end: 12, label: "ORG", score: 0.8 },
    ]);
    render(<SpanOverlay result={result} />);
    expect(screen.getByTestId("span-dropped")).toHaveTextContent(
      /could not be drawn/i,
    );
  });

  it("renders plain text with no marks at all when nothing was found", () => {
    render(<SpanOverlay result={marked([])} />);
    expect(screen.queryByTestId("span-mark")).not.toBeInTheDocument();
    expect(screen.getByTestId("span-overlay")).toHaveTextContent("Priya flew");
  });
});
