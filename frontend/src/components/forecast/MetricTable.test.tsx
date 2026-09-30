import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { ForecastMetrics } from "@/forecast/metrics";

import { MetricTable, type MethodRow } from "./MetricTable";

function m(over: Partial<ForecastMetrics> = {}): ForecastMetrics {
  return { mae: 47.83, rmse: 55.2, mape: 0.112, mapeSkipped: 0, mase: 1.57, ...over };
}

const rows: MethodRow[] = [
  { id: "naive", label: "Naive", metrics: m({ mae: 76, mase: 2.5 }) },
  { id: "seasonal-naive", label: "Seasonal naive", metrics: m() },
  { id: "drift", label: "Drift", metrics: m({ mae: 66.31, mase: 2.18 }) },
  { id: "mean", label: "Historical mean", metrics: m({ mae: 213.67, mase: 7.02 }) },
];

describe("MetricTable", () => {
  it("scores every method on the same window, in one table", () => {
    // There is no model to put the baselines beside — the baselines *are* the
    // null models, so the comparison on screen is between them.
    render(<MetricTable rows={rows} horizon={12} />);
    const table = within(screen.getByTestId("metric-table")).getByRole("table");
    expect(within(table).getAllByRole("row")).toHaveLength(5);
    expect(table).toHaveTextContent("Seasonal naive");
    expect(table).toHaveTextContent("1.57");
  });

  it("marks the best method by MAE, so the reader need not scan the column", () => {
    render(<MetricTable rows={rows} horizon={12} />);
    const best = screen
      .getAllByRole("row")
      .find((r) => r.textContent?.includes("best"));
    expect(best).toHaveTextContent("Seasonal naive");
  });

  it("highlights the selected method without filtering the others out", () => {
    // The selection drives the backtest below; the table stays a comparison.
    render(<MetricTable rows={rows} selected="drift" horizon={12} />);
    const selected = screen.getAllByRole("row").filter((r) => r.dataset.selected === "true");
    expect(selected).toHaveLength(1);
    expect(selected[0]).toHaveTextContent("Drift");
  });

  it("renders MAPE as a percentage when every actual was non-zero", () => {
    render(<MetricTable rows={rows} horizon={12} />);
    expect(screen.getByTestId("metric-table")).toHaveTextContent("11.2%");
  });

  it("blanks MAPE rather than averaging over the rows that happened to be non-zero", () => {
    // That average describes a different test set from every other number in
    // the row, so a dash plus an explanation beats a plausible figure.
    render(
      <MetricTable
        rows={[{ id: "naive", label: "Naive", metrics: m({ mape: null, mapeSkipped: 2 }) }]}
        horizon={12}
      />,
    );
    const panel = screen.getByTestId("metric-table");
    expect(within(panel).getByRole("table")).toHaveTextContent("—");
    expect(panel).toHaveTextContent(/contains a zero/i);
    expect(panel).toHaveTextContent(/different test set/i);
  });

  it("omits the MAPE explanation when nothing was skipped", () => {
    render(<MetricTable rows={rows} horizon={12} />);
    expect(screen.getByTestId("metric-table")).not.toHaveTextContent(/contains a zero/i);
  });

  it("dashes a MASE that could not be scaled, rather than rendering Infinity", () => {
    // A perfectly flat history has a zero denominator.
    render(
      <MetricTable
        rows={[{ id: "naive", label: "Naive", metrics: m({ mase: null }) }]}
        horizon={12}
      />,
    );
    expect(screen.getByTestId("metric-table").textContent).not.toMatch(/Infinity|NaN/);
  });

  it("says MASE is the column to read, and that it is a ratio not a quantity", () => {
    render(<MetricTable rows={rows} horizon={12} />);
    const panel = screen.getByTestId("metric-table");
    expect(panel).toHaveTextContent(/MASE is the column to read/i);
    expect(panel).toHaveTextContent(/in-sample one-step/i);
  });

  it("claims 1.0 is break-even only at a horizon of one", () => {
    // The denominator is the in-sample *one-step* error by definition, so at
    // horizon 1 the benchmark is the same estimator and 1.0 is the break-even
    // point.
    render(<MetricTable rows={rows} horizon={1} />);
    const note = screen.getByTestId("mase-horizon-note");
    expect(note).toHaveTextContent(/no better than repeating the last value/i);
    expect(note).not.toHaveTextContent(/not the break-even point/i);
  });

  it("warns that 1.0 is NOT break-even beyond one step, and says what to compare instead", () => {
    // Measured on the bundled random walk: a naive forecast scores ~1.0 one step
    // out and ~3.0 fourteen steps out, with nothing wrong. Reading 2.6 as "worse
    // than naive" is the mistake this note exists to prevent.
    render(<MetricTable rows={rows} horizon={14} />);
    const note = screen.getByTestId("mase-horizon-note");
    expect(note).toHaveTextContent("14 steps ahead");
    expect(note).toHaveTextContent(/1\.0 is not the break-even point here/i);
    expect(note).toHaveTextContent(/against each other/i);
  });
});
