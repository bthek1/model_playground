import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { CLASS_COLORS, GraphCanvas, layoutToPixels, PAD } from "./GraphCanvas";

// happy-dom gives a canvas no 2D context and has no ResizeObserver, so both
// draw effects take their guarded early-return paths. These tests cover the two
// things that are assertable without pixels: the pure coordinate mapping, and
// the structural/accessibility contract. What the canvas actually paints is
// asserted in e2e/specs/webgpu/graph.spec.ts, against a real browser.

const unit = (values: number[]) => Float32Array.from(values);

describe("layoutToPixels", () => {
  it("centres the drawing, with equal padding on the constrained axis", () => {
    // A 400x300 box: the square is 300 - 2*PAD on a side, centred.
    const { px, py, span } = layoutToPixels(unit([0, 1]), unit([0, 1]), 2, 400, 300);

    expect(span).toBe(300 - 2 * PAD);
    // Vertical is the constrained axis, so both gaps are exactly PAD.
    expect(py[0]).toBeCloseTo(PAD, 5);
    expect(300 - py[1]).toBeCloseTo(PAD, 5);
    // Horizontal has slack, and it is split evenly.
    expect(px[0]).toBeCloseTo((400 - span) / 2, 5);
    expect(400 - px[1]).toBeCloseTo((400 - span) / 2, 5);
  });

  it("preserves the layout's aspect ratio in a wide box", () => {
    // A node at (0.5, 0.5) is the centre of the unit square and must land in
    // the centre of the box, not be stretched across it.
    const { px, py } = layoutToPixels(unit([0.5]), unit([0.5]), 1, 800, 200);
    expect(px[0]).toBeCloseTo(400, 5);
    expect(py[0]).toBeCloseTo(100, 5);
  });

  it("keeps every node inside the box", () => {
    const n = 5;
    const coords = unit([0, 0.25, 0.5, 0.75, 1]);
    for (const [w, h] of [
      [400, 300],
      [300, 400],
      [250, 250],
    ]) {
      const { px, py } = layoutToPixels(coords, coords, n, w, h);
      for (let i = 0; i < n; i++) {
        expect(px[i]).toBeGreaterThanOrEqual(0);
        expect(px[i]).toBeLessThanOrEqual(w);
        expect(py[i]).toBeGreaterThanOrEqual(0);
        expect(py[i]).toBeLessThanOrEqual(h);
      }
    }
  });

  it("degrades to a zero span in a box smaller than its own padding", () => {
    // Never negative: a negative span would mirror the graph.
    const { span, px } = layoutToPixels(unit([0, 1]), unit([0, 1]), 2, 4, 4);
    expect(span).toBe(0);
    expect(px[0]).toBe(px[1]);
  });

  it("returns empty arrays before the host has been measured", () => {
    const { px, py, span } = layoutToPixels(unit([]), unit([]), 0, 0, 0);
    expect(px).toHaveLength(0);
    expect(py).toHaveLength(0);
    expect(span).toBe(0);
  });
});

describe("CLASS_COLORS", () => {
  it("has one distinct colour per Cora topic", () => {
    expect(CLASS_COLORS).toHaveLength(7);
    expect(new Set(CLASS_COLORS).size).toBe(7);
  });
});

describe("GraphCanvas", () => {
  const props = {
    nNodes: 3,
    rowPtr: Uint32Array.from([0, 1, 3, 4]),
    colIdx: Uint32Array.from([1, 0, 2, 1]),
    x: unit([0, 0.5, 1]),
    y: unit([0, 0.5, 1]),
    labels: Uint8Array.from([0, 1, 2]),
    predictions: Uint8Array.from([0, 0, 2]),
    trainMask: Uint8Array.from([1, 0, 0]),
    colorBy: "predicted" as const,
  };

  it("renders without a 2D context or a ResizeObserver", () => {
    // The guarded early-returns are the contract: a canvas that threw here
    // would take the whole route down on any browser that refuses a context.
    expect(() => render(<GraphCanvas {...props} />)).not.toThrow();
    expect(screen.getByTestId("graph-canvas")).toBeInTheDocument();
  });

  it("describes what the colours mean, so the picture is not the only clue", () => {
    render(<GraphCanvas {...props} />);
    const canvas = screen.getByRole("img");
    expect(canvas).toHaveAccessibleName(/3 papers/);
    expect(canvas).toHaveAccessibleName(/predicted topic/);
    expect(canvas).toHaveAccessibleName(/ringed/i);
  });

  it("says when it is showing the true labels instead", () => {
    render(<GraphCanvas {...props} colorBy="true" />);
    expect(screen.getByRole("img")).toHaveAccessibleName(/true topic/);
  });
});
