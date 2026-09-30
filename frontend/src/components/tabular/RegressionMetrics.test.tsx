import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { RegressionMetrics } from "@/tabular/types";

import { RegressionMetricBlock } from "./RegressionMetrics";

function metrics(over: Partial<RegressionMetrics> = {}): RegressionMetrics {
  return {
    rmse: 312.5,
    mae: 244.1,
    r2: 0.845,
    baselineRmse: 801.2,
    baselineMae: 640.7,
    units: "body_mass_g",
    ...over,
  };
}

describe("RegressionMetricBlock", () => {
  it("labels every score with the units it is in", () => {
    // The whole point of transform.ts owning the units: a number whose units are
    // implied can be read as a number in different units.
    render(<RegressionMetricBlock metrics={metrics()} />);
    const block = screen.getByTestId("regression-metrics");
    expect(block).toHaveTextContent("RMSE");
    expect(block).toHaveTextContent("312.50");
    expect(block.textContent?.match(/body_mass_g/g)?.length).toBeGreaterThanOrEqual(2);
    // R² is the one number here that is not in the target's units, and says so.
    expect(block).toHaveTextContent("unitless");
    expect(block).toHaveTextContent("0.845");
  });

  it("states the train-mean baseline in the same units as the score", () => {
    // R² is already a comparison but it is unitless and is taken against the
    // *test* mean; this is the null model in the reader's own units.
    render(<RegressionMetricBlock metrics={metrics()} />);
    const note = screen.getByTestId("regression-baseline");
    expect(note).toHaveTextContent(/predicting the training mean/i);
    expect(note).toHaveTextContent("801.20 body_mass_g");
    expect(note).toHaveTextContent(/488.70 better/);
  });

  it("says plainly when the model has not beaten the mean", () => {
    render(<RegressionMetricBlock metrics={metrics({ rmse: 900, baselineRmse: 801.2 })} />);
    expect(screen.getByTestId("regression-baseline")).toHaveTextContent(
      /learned the average and nothing else/i,
    );
  });

  it("renders no log-space block when the fit was on the raw target", () => {
    render(<RegressionMetricBlock metrics={metrics()} />);
    expect(screen.queryByTestId("log-space-metrics")).toBeNull();
  });

  it("keeps the log-space numbers in a separate block, with their own units", () => {
    // Rendering these inline beside the numbers above *is* the mistake the
    // toggle exists to demonstrate, so the separation is the assertion.
    render(
      <RegressionMetricBlock
        metrics={metrics()}
        logSpace={metrics({ rmse: 0.081, mae: 0.062, r2: 0.83, units: "log(1 + body_mass_g)" })}
      />,
    );
    const log = screen.getByTestId("log-space-metrics");
    expect(log).toHaveTextContent("log(1 + body_mass_g)");
    expect(log).toHaveTextContent("0.0810");
    expect(log).toHaveTextContent(/not comparable with the ones above/i);
    // And it says which of the two the reader should act on.
    expect(log).toHaveTextContent(/comparable numbers are the ones at the top/i);
  });

  it("renders a dash rather than NaN when a score could not be computed", () => {
    render(<RegressionMetricBlock metrics={metrics({ rmse: NaN, mae: Infinity })} />);
    expect(screen.getByTestId("regression-metrics").textContent).not.toMatch(/NaN|Infinity/);
  });
});
