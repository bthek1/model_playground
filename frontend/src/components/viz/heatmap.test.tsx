import { render, renderHook, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { DivergingLegend, HeatmapTile, useMaxAbs } from "./heatmap";

// happy-dom canvases expose no 2D context, so HeatmapTile takes its guarded
// early-return path; these tests cover the structural contract (labels, sizing,
// normalisation) rather than pixel output.
describe("useMaxAbs", () => {
  it("returns the largest absolute value", () => {
    const { result } = renderHook(() =>
      useMaxAbs(new Float32Array([1, -4, 2, 3])),
    );
    expect(result.current).toBe(4);
  });

  it("falls back to 1 for an all-zero tensor (safe divisor)", () => {
    const { result } = renderHook(() => useMaxAbs(new Float32Array(9)));
    expect(result.current).toBe(1);
  });
});

describe("HeatmapTile", () => {
  it("sizes the canvas from rows/cols and forwards aria-label", () => {
    render(
      <HeatmapTile
        values={new Float32Array([1, 2, 3, 4, 5, 6])}
        rows={2}
        cols={3}
        maxAbs={6}
        aria-label="result heatmap"
      />,
    );
    const canvas = screen.getByLabelText("result heatmap") as HTMLCanvasElement;
    expect(canvas.width).toBe(3);
    expect(canvas.height).toBe(2);
    expect(canvas).toHaveClass("[image-rendering:pixelated]");
  });
});

describe("DivergingLegend", () => {
  it("labels the ±maxAbs endpoints and both directions", () => {
    render(
      <DivergingLegend
        maxAbs={2.5}
        posLabel="positive"
        negLabel="negative"
        note="after epoch 3"
      />,
    );
    expect(screen.getByText("+2.50")).toBeInTheDocument();
    expect(screen.getByText("−2.50")).toBeInTheDocument();
    expect(screen.getByText(/positive/)).toBeInTheDocument();
    expect(screen.getByText(/negative/)).toBeInTheDocument();
    expect(screen.getByText("after epoch 3")).toBeInTheDocument();
  });
});
