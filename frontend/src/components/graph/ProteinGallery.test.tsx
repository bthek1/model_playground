import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { GraphLayoutPayload } from "@/webgpu/proteinSession";

import { ProteinGallery, ProteinTile, TILE_PAD, tilePixels } from "./ProteinGallery";

// happy-dom gives a canvas no 2D context, so the painting cannot be asserted
// here — the geometry is pulled out as a pure function for exactly that reason,
// the same lesson GraphCanvas.layoutToPixels records. The degenerate cases are
// the ones that matter: PROTEINS really does contain graphs of one node, and a
// four-residue chain occupies a sliver of the unit square it was normalised in.

const unit = (values: number[]) => Float32Array.from(values);

describe("tilePixels", () => {
  it("fits the graph's own extent to the tile, not the unit square", () => {
    // A graph laid out into the left-hand tenth of the unit square must still
    // fill the tile — otherwise a small protein draws as a dot in one corner.
    const { px } = tilePixels(unit([0, 0.05, 0.1]), unit([0, 0, 0]), 100);
    expect(px[0]).toBeCloseTo(TILE_PAD, 5);
    expect(px[2]).toBeCloseTo(100 - TILE_PAD, 5);
  });

  it("keeps every node inside the tile", () => {
    const coords = unit([0, 0.25, 0.5, 0.75, 1]);
    for (const size of [40, 84, 200]) {
      const { px, py } = tilePixels(coords, coords, size);
      for (let i = 0; i < coords.length; i++) {
        expect(px[i]).toBeGreaterThanOrEqual(0);
        expect(px[i]).toBeLessThanOrEqual(size);
        expect(py[i]).toBeGreaterThanOrEqual(0);
        expect(py[i]).toBeLessThanOrEqual(size);
      }
    }
  });

  it("preserves the aspect ratio of a wide graph", () => {
    // A flat chain must stay flat, not be stretched to fill the height.
    const { px, py } = tilePixels(unit([0, 0.5, 1]), unit([0, 0, 0]), 100);
    expect(py[0]).toBeCloseTo(py[2], 5);
    expect(px[2] - px[0]).toBeCloseTo(100 - 2 * TILE_PAD, 5);
  });

  it("centres a single-node graph instead of dividing by zero", () => {
    // PROTEINS contains graphs of four nodes and fewer; a collapsed extent must
    // not produce NaN coordinates, which draw nothing at all.
    const { px, py } = tilePixels(unit([0.5]), unit([0.5]), 100);
    expect(Number.isFinite(px[0])).toBe(true);
    expect(px[0]).toBeCloseTo(50, 5);
    expect(py[0]).toBeCloseTo(50, 5);
  });

  it("handles an empty graph without throwing", () => {
    const { px, py } = tilePixels(unit([]), unit([]), 100);
    expect(px).toHaveLength(0);
    expect(py).toHaveLength(0);
  });
});

function layout(index: number, nNodes = 3): GraphLayoutPayload {
  return {
    index,
    nNodes,
    rowPtr: Uint32Array.from([0, 1, 2, 2]).slice(0, nNodes + 1),
    colIdx: Uint32Array.from([1, 0]),
    x: unit(Array.from({ length: nNodes }, (_, i) => i / Math.max(1, nNodes - 1))),
    y: unit(Array.from({ length: nNodes }, () => 0.5)),
  };
}

describe("ProteinTile", () => {
  it("renders without a 2D context", () => {
    expect(() =>
      render(
        <ProteinTile layout={layout(0)} correct selected={false} size={84} />,
      ),
    ).not.toThrow();
  });

  it("renders a placeholder before its layout has arrived", () => {
    expect(() =>
      render(
        <ProteinTile layout={undefined} correct={false} selected size={84} />,
      ),
    ).not.toThrow();
  });
});

describe("ProteinGallery", () => {
  const labels = Uint8Array.from([0, 1, 0]);
  const predicted = Uint8Array.from([0, 0, 0]);

  it("asks for the layouts it does not have, once each", () => {
    const onNeedLayout = vi.fn();
    render(
      <ProteinGallery
        graphs={[0, 1, 2]}
        labels={labels}
        predicted={predicted}
        layouts={new Map([[1, layout(1)]])}
        onNeedLayout={onNeedLayout}
        selected={null}
        onSelect={vi.fn()}
        classNames={["No", "Yes"]}
      />,
    );
    expect(onNeedLayout).toHaveBeenCalledTimes(2);
    expect(onNeedLayout).toHaveBeenCalledWith(0);
    expect(onNeedLayout).toHaveBeenCalledWith(2);
    expect(onNeedLayout).not.toHaveBeenCalledWith(1);
  });

  it("says what each protein is and what was predicted, without colour", () => {
    // The tile's border carries right-or-wrong, and a border is not available to
    // a screen reader — so the same fact is in the accessible name.
    render(
      <ProteinGallery
        graphs={[1]}
        labels={labels}
        predicted={predicted}
        layouts={new Map([[1, layout(1)]])}
        onNeedLayout={vi.fn()}
        selected={null}
        onSelect={vi.fn()}
        classNames={["Not an enzyme", "Enzyme"]}
      />,
    );
    const tile = screen.getByRole("button");
    expect(tile).toHaveAccessibleName(/really Enzyme/i);
    expect(tile).toHaveAccessibleName(/predicted Not an enzyme/i);
  });

  it("reports which tile is selected", () => {
    render(
      <ProteinGallery
        graphs={[0, 1]}
        labels={labels}
        predicted={predicted}
        layouts={new Map()}
        onNeedLayout={vi.fn()}
        selected={1}
        onSelect={vi.fn()}
        classNames={["No", "Yes"]}
      />,
    );
    const tiles = screen.getAllByRole("button");
    expect(tiles[0]).toHaveAttribute("aria-pressed", "false");
    expect(tiles[1]).toHaveAttribute("aria-pressed", "true");
  });
});
