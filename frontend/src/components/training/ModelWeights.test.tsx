import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ModelWeights } from "./ModelWeights";

// happy-dom canvases have no 2D context, so ClassTemplate takes its guarded
// early-return path; these tests assert structure, labels, and the tileSize
// plumbing rather than pixel output.
describe("ModelWeights", () => {
  it("renders one labelled template per class", () => {
    render(<ModelWeights weights={new Float32Array(784 * 10)} />);
    expect(
      screen.getAllByLabelText(/weight template for digit/i),
    ).toHaveLength(10);
    expect(
      screen.getByLabelText(/weight template for digit 0/i),
    ).toBeInTheDocument();
    expect(
      screen.getByLabelText(/weight template for digit 9/i),
    ).toBeInTheDocument();
  });

  it("defaults each tile to size-16", () => {
    render(<ModelWeights weights={new Float32Array(784 * 10)} />);
    expect(screen.getByLabelText(/weight template for digit 0/i)).toHaveClass(
      "size-16",
    );
  });

  it("applies a custom tileSize", () => {
    render(
      <ModelWeights weights={new Float32Array(784 * 10)} tileSize="size-32" />,
    );
    const tile = screen.getByLabelText(/weight template for digit 0/i);
    expect(tile).toHaveClass("size-32");
    expect(tile).not.toHaveClass("size-16");
  });

  it("shows per-class bias values when provided", () => {
    const bias = new Float32Array(10);
    bias[2] = -0.4;
    render(<ModelWeights weights={new Float32Array(784 * 10)} bias={bias} />);
    // Bias chip renders as "b -0.40" for the negative class.
    expect(screen.getByText(/b\s*-0\.40/)).toBeInTheDocument();
  });

  it("renders the colour legend and epoch marker", () => {
    const weights = new Float32Array(784 * 10);
    weights[0] = 2; // sets maxAbs = 2.00
    render(<ModelWeights weights={weights} epoch={4} />);
    expect(screen.getByText("+2.00")).toBeInTheDocument();
    expect(screen.getByText(/against/i)).toBeInTheDocument();
    expect(screen.getByText(/for/i)).toBeInTheDocument();
    expect(screen.getByText(/after epoch 5/i)).toBeInTheDocument();
  });

  it("falls back to a unit scale for all-zero weights", () => {
    render(<ModelWeights weights={new Float32Array(784 * 10)} />);
    // maxAbs = 0 → `|| 1`, so the legend endpoints are ±1.00.
    expect(screen.getByText("+1.00")).toBeInTheDocument();
  });
});
