import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ClassificationMetrics } from "@/tabular/types";

import { ThresholdControl } from "./ThresholdControl";

const metrics: ClassificationMetrics = {
  accuracy: 0.8,
  precision: 0.712,
  recall: 0.934,
  f1: 0.808,
  confusion: { counts: Int32Array.from([4, 1, 1, 4]), labels: ["no", "yes"] },
  baselineAccuracy: 0.55,
  baselineLabel: "no",
};

function setup(threshold = 0.5) {
  const onChange = vi.fn();
  render(
    <ThresholdControl
      threshold={threshold}
      onChange={onChange}
      positiveLabel="defaulted"
      metrics={metrics}
    />,
  );
  return { onChange };
}

describe("ThresholdControl", () => {
  it("names the class the threshold applies to", () => {
    // "Threshold 0.5" is meaningless without saying 0.5 *of what*, and on a
    // two-class problem the answer is not symmetric.
    setup();
    expect(screen.getByLabelText(/decision threshold for “defaulted”/i)).toBeInTheDocument();
  });

  it("shows the current value and reports a drag", () => {
    const { onChange } = setup(0.35);
    const slider = screen.getByLabelText(/decision threshold/i);
    expect(slider).toHaveValue("0.35");
    expect(screen.getByTestId("threshold-control")).toHaveTextContent("0.35");
    fireEvent.change(slider, { target: { value: "0.8" } });
    expect(onChange).toHaveBeenCalledWith(0.8);
  });

  it("keeps the slider inside (0, 1), where a decision boundary can live", () => {
    // 0 predicts every row positive and 1 predicts none; both are degenerate and
    // neither is a threshold anyone is choosing.
    setup();
    const slider = screen.getByLabelText(/decision threshold/i);
    expect(slider).toHaveAttribute("min", "0.01");
    expect(slider).toHaveAttribute("max", "0.99");
  });

  it("renders the three metrics that actually move with it", () => {
    // Accuracy is deliberately absent: it is the number that hides the trade-off
    // the slider exists to expose.
    setup();
    const panel = screen.getByTestId("threshold-control");
    expect(panel).toHaveTextContent("precision 71.2%");
    expect(panel).toHaveTextContent("recall 93.4%");
    expect(panel).toHaveTextContent("F1 80.8%");
  });

  it("says 0.5 is a convention, and that moving it costs nothing", () => {
    // Both halves are load-bearing: the first is why the control exists, the
    // second is the promise that it re-reads rather than refits.
    setup();
    const panel = screen.getByTestId("threshold-control");
    expect(panel).toHaveTextContent(/convention, not a decision/i);
    expect(panel).toHaveTextContent(/does not refit/i);
  });
});
