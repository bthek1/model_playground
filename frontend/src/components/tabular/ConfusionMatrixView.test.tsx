import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ConfusionMatrixView } from "./ConfusionMatrixView";

const matrix = {
  // rows are actual, columns predicted
  counts: Int32Array.from([5, 2, 0, 1, 7, 0, 0, 3, 4]),
  labels: ["Adelie", "Gentoo", "Chinstrap"],
};

describe("ConfusionMatrixView", () => {
  it("renders a real table with row and column headers", () => {
    // A grid of coloured divs is unreadable by a screen reader and its counts
    // are not selectable text. The shading is a secondary encoding only.
    render(<ConfusionMatrixView matrix={matrix} />);
    const table = screen.getByRole("table");
    expect(within(table).getAllByRole("columnheader")).toHaveLength(4);
    expect(within(table).getAllByRole("rowheader")).toHaveLength(3);
  });

  it("puts every count in the cell its indices name", () => {
    // `counts[actual * k + predicted]` — transposing this is the bug available
    // here, and it produces a plausible-looking matrix.
    render(<ConfusionMatrixView matrix={matrix} />);
    const rows = screen.getAllByRole("row").slice(1);
    const cells = (i: number) =>
      within(rows[i]).getAllByRole("cell").map((c) => c.textContent);
    expect(cells(0)).toEqual(["5", "2", "0"]);
    expect(cells(1)).toEqual(["1", "7", "0"]);
    expect(cells(2)).toEqual(["0", "3", "4"]);
  });

  it("says which axis is which, so the diagonal can be read", () => {
    render(<ConfusionMatrixView matrix={matrix} />);
    expect(screen.getByRole("table")).toHaveAccessibleName(/rows are the true class/i);
  });

  it("takes a caption when the page has better words for it", () => {
    render(<ConfusionMatrixView matrix={matrix} caption="At a threshold of 0.20." />);
    expect(screen.getByRole("table")).toHaveAccessibleName("At a threshold of 0.20.");
  });

  it("leaves a zero cell unshaded, and never hides a count behind colour alone", () => {
    render(<ConfusionMatrixView matrix={matrix} />);
    const rows = screen.getAllByRole("row").slice(1);
    const zero = within(rows[0]).getAllByRole("cell")[2];
    expect(zero).toHaveTextContent("0");
    expect(zero.getAttribute("style") ?? "").not.toContain("background-color");
  });

  it("survives a two-class matrix, which is the common case", () => {
    render(
      <ConfusionMatrixView
        matrix={{ counts: Int32Array.from([3, 1, 2, 4]), labels: ["no", "yes"] }}
      />,
    );
    expect(screen.getAllByRole("row")).toHaveLength(3);
    expect(screen.getByTestId("confusion-matrix")).toHaveTextContent("4");
  });
});
