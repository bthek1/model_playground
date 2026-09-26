import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { EChartsOption } from "echarts";

const captured: EChartsOption[] = [];
vi.mock("@/components/charts/EChart", () => ({
  default: ({ option }: { option: EChartsOption }) => {
    captured.push(option);
    return <div data-testid="echart" />;
  },
}));
vi.mock("@/lib/theme", () => ({ getCSSVar: (t: string) => `var(--${t})` }));

const { BacktestStrip } = await import("./BacktestStrip");
const { backtest } = await import("@/forecast/backtest");

/** A series with a level shift, so the windows genuinely disagree. */
function shifted(n = 200): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = 100 + (i % 7) * 3 + (i > 150 ? 60 : 0);
  return out;
}

const options = {
  horizon: 4,
  windows: 12,
  stride: 5,
  mode: "expanding" as const,
  windowLength: 30,
  season: 1,
};

async function setup(metric: "mase" | "mae" | "rmse" = "mae", values = shifted()) {
  captured.length = 0;
  const result = backtest(values, "naive", options);
  render(<BacktestStrip result={result} metric={metric} />);
  await waitFor(() => expect(screen.getByTestId("backtest-strip")).toBeInTheDocument());
  return result;
}

const series = () => {
  const s = captured[0].series;
  return (Array.isArray(s) ? s[0] : s) as {
    data: number[];
    markLine?: { data: { yAxis: number }[] };
  };
};

describe("BacktestStrip", () => {
  it("draws one bar per window", async () => {
    const result = await setup();
    expect(series().data).toHaveLength(result.windows.length);
    expect((captured[0].xAxis as { data: string[] }).data).toHaveLength(result.windows.length);
  });

  it("draws the single-split number across the bars as a line", async () => {
    // Both numbers on screen at once is the whole demonstration. A page showing
    // only the spread would be correct and would not make the point.
    const result = await setup();
    expect(series().markLine?.data[0].yAxis).toBeCloseTo(result.single.mae, 6);
  });

  it("reports the single split, the range and the median in text too", async () => {
    // The chart carries the shape; the line carries the numbers, so the finding
    // survives a screen reader and a copy-paste.
    const result = await setup();
    const line = screen.getByTestId("backtest-spread");
    expect(line).toHaveTextContent(`one split ${result.single.mae.toFixed(3)}`);
    expect(line).toHaveTextContent(/windows \d/);
    expect(line).toHaveTextContent(/median \d/);
  });

  it("says how much worse the worst window is when the spread is wide", async () => {
    // "One split lies" is the claim; the ratio is what makes it a number.
    await setup();
    const panel = screen.getByTestId("backtest-strip");
    expect(panel).toHaveTextContent(/× the best/);
    expect(panel).toHaveTextContent(/one split lies/i);
  });

  it("stays quiet about the ratio when the windows agree", async () => {
    // On a series with no regime change there is no finding to announce, and
    // announcing one anyway would make the sentence meaningless.
    const flat = new Float32Array(200).fill(100);
    await setup("mae", flat);
    expect(screen.getByTestId("backtest-strip")).not.toHaveTextContent(/one split lies/i);
  });

  it("switches the metric it charts, and names it on the axis", async () => {
    await setup("mase");
    expect((captured[0].yAxis as { name: string }).name).toBe("MASE");
    // Unmounted between the two renders, or the second `getByTestId` matches
    // both strips and fails for a reason that has nothing to do with the axis.
    cleanup();
    await setup("rmse");
    expect((captured[0].yAxis as { name: string }).name).toBe("RMSE");
  });

  it("charts a null MASE as zero rather than breaking the axis", async () => {
    // A flat history has no scale to divide by, so MASE is null by design.
    const flat = new Float32Array(200).fill(100);
    await setup("mase", flat);
    expect(series().data.every((v) => Number.isFinite(v))).toBe(true);
    expect(screen.getByTestId("backtest-spread")).toHaveTextContent("one split —");
  });

  it("says the bars are the same method with the cut moved", async () => {
    await setup();
    expect(screen.getByTestId("backtest-strip")).toHaveTextContent(
      /same method, the same series and the same horizon/i,
    );
  });
});
