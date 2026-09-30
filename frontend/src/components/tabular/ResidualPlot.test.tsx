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

const { PredictedVsActual, ResidualPlot } = await import("./ResidualPlot");

type Series = { type: string; data: unknown[]; name?: string };
const series = (i = 0) => (captured[i].series as Series[]) ?? [];

const actual = Float32Array.from([10, 20, 30, 40]);
const predicted = Float32Array.from([9, 21, 29, 41]);

describe("PredictedVsActual", () => {
  it("plots one point per held-out row against an identity line", async () => {
    // The *shape* of the departure from that line is the thing a single R²
    // cannot report, which is why the line is drawn rather than implied.
    captured.length = 0;
    render(<PredictedVsActual predicted={predicted} actual={actual} units="price" />);
    await waitFor(() => expect(screen.getByTestId("predicted-vs-actual")).toBeInTheDocument());

    const [points, identity] = series();
    expect(points.type).toBe("scatter");
    expect(points.data).toHaveLength(4);
    expect(points.data[0]).toEqual([10, 9]);
    expect(identity.type).toBe("line");
    // Spans the full range of both axes, so it is a reference rather than a fit.
    expect(identity.data).toEqual([
      [9, 9],
      [41, 41],
    ]);
  });

  it("names the axis with the target's units", async () => {
    captured.length = 0;
    render(<PredictedVsActual predicted={predicted} actual={actual} units="body_mass_g" />);
    await waitFor(() => expect(screen.getByTestId("predicted-vs-actual")).toBeInTheDocument());
    expect((captured[0].xAxis as { name: string }).name).toBe("actual (body_mass_g)");
  });

  it("adds the quantile band's edges only when there is a band", async () => {
    captured.length = 0;
    render(<PredictedVsActual predicted={predicted} actual={actual} units="price" />);
    await waitFor(() => expect(screen.getByTestId("predicted-vs-actual")).toBeInTheDocument());
    expect(series()).toHaveLength(2);

    cleanup();
    captured.length = 0;
    // three quantiles x four rows, one contiguous row per quantile
    const band = Float32Array.from([5, 15, 25, 35, 9, 21, 29, 41, 14, 24, 34, 44]);
    render(
      <PredictedVsActual
        predicted={predicted}
        actual={actual}
        units="price"
        band={band}
        quantiles={[0.1, 0.5, 0.9]}
      />,
    );
    await waitFor(() => expect(screen.getByTestId("predicted-vs-actual")).toBeInTheDocument());
    const withBand = series(0);
    expect(withBand).toHaveLength(3);
    // The outer edges only — two points per row, low and high.
    expect(withBand[2].data).toHaveLength(8);
    expect(withBand[2].data[0]).toEqual([10, 5]);
    expect(withBand[2].data[1]).toEqual([10, 14]);
  });

  it("thins a large cloud rather than handing ECharts every row", async () => {
    // Past a couple of thousand points the scatter is a blob, and the cost is
    // paid on every re-render of a page whose controls re-derive constantly.
    captured.length = 0;
    const n = 5000;
    const big = Float32Array.from({ length: n }, (_, i) => i);
    render(<PredictedVsActual predicted={big} actual={big} units="y" />);
    await waitFor(() => expect(screen.getByTestId("predicted-vs-actual")).toBeInTheDocument());
    expect(series()[0].data).toHaveLength(2000);
  });

  it("says what the identity line is for", async () => {
    captured.length = 0;
    render(<PredictedVsActual predicted={predicted} actual={actual} units="price" />);
    await waitFor(() => expect(screen.getByTestId("predicted-vs-actual")).toBeInTheDocument());
    expect(screen.getByTestId("predicted-vs-actual")).toHaveTextContent(
      /systematically wrong there/i,
    );
  });

  it("survives an empty held-out set without producing a NaN axis", async () => {
    captured.length = 0;
    render(
      <PredictedVsActual
        predicted={new Float32Array(0)}
        actual={new Float32Array(0)}
        units="y"
      />,
    );
    await waitFor(() => expect(screen.getByTestId("predicted-vs-actual")).toBeInTheDocument());
    expect(JSON.stringify(series()[1].data)).not.toMatch(/null|Infinity/);
  });
});

describe("ResidualPlot", () => {
  it("plots the residual against the prediction, with a zero line", async () => {
    captured.length = 0;
    render(<ResidualPlot predicted={predicted} actual={actual} units="price" />);
    await waitFor(() => expect(screen.getByTestId("residual-plot")).toBeInTheDocument());
    const [points, zero] = series();
    expect(points.data[0]).toEqual([9, 1]);
    expect(points.data[1]).toEqual([21, -1]);
    expect(zero.data).toEqual([
      [9, 0],
      [41, 0],
    ]);
    expect((captured[0].yAxis as { name: string }).name).toBe("residual");
  });

  it("explains the funnel, which is the failure a good R² hides", async () => {
    // A model with a respectable R² and a funnel-shaped residual plot is
    // systematically worse at one end, and no scalar on the page says so.
    captured.length = 0;
    render(<ResidualPlot predicted={predicted} actual={actual} units="price" />);
    await waitFor(() => expect(screen.getByTestId("residual-plot")).toBeInTheDocument());
    const panel = screen.getByTestId("residual-plot");
    expect(panel).toHaveTextContent(/shapeless band around zero/i);
    expect(panel).toHaveTextContent(/funnel/i);
    expect(panel).toHaveTextContent(/worse than the RMSE above suggests/i);
  });
});
