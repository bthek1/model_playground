import { render, screen, waitFor } from "@testing-library/react";
import type { EChartsOption } from "echarts";
import { beforeEach, describe, expect, it, vi } from "vitest";

// Assert the option object, never the pixels (CLAUDE.md): capture what the
// lazy wrapper is handed, and echo token names so colours are comparable.
const captured: EChartsOption[] = [];
vi.mock("@/components/charts/EChart", () => ({
  default: ({ option }: { option: EChartsOption }) => {
    captured.push(option);
    return <div data-testid="echart" />;
  },
}));
vi.mock("@/lib/theme", () => ({ getCSSVar: (t: string) => `var(--${t})` }));

const { ReturnChart } = await import("./ReturnChart");
const { MAX_POINTS } = await import("./series");

type Series = { name: string; data: [number, number][]; lineStyle: { color: string; type?: string } };

async function draw(props: Parameters<typeof ReturnChart>[0]): Promise<Series[]> {
  captured.length = 0;
  render(<ReturnChart {...props} />);
  await waitFor(() => expect(screen.getByTestId("echart")).toBeInTheDocument());
  return captured[captured.length - 1].series as Series[];
}

beforeEach(() => {
  captured.length = 0;
});

describe("ReturnChart", () => {
  it("draws the raw trace, unsmoothed, and its running mean over it", async () => {
    const series = await draw({ returns: [0, 1, 0, 1], window: 2, yLabel: "success", label: "Q" });
    expect(series.map((s) => s.name)).toEqual(["Q (each episode)", "Q (mean of last 2)"]);
    // The raw series is the returns themselves — the variance is the render.
    expect(series[0].data).toEqual([[1, 0], [2, 1], [3, 0], [4, 1]]);
    expect(series[1].data.map(([, y]) => y)).toEqual([0, 0.5, 0.5, 0.5]);
    // Episodes are numbered from 1, not 0.
    expect(series[0].data[0][0]).toBe(1);
  });

  it("draws an earlier run as a dashed mean, underneath the current one", async () => {
    const series = await draw({
      returns: [5, 6],
      window: 2,
      yLabel: "steps",
      comparisons: [{ label: "seed 2", returns: [1, 2, 3] }],
    });
    // Comparisons first, so the current run is drawn on top.
    expect(series[0].name).toBe("seed 2");
    expect(series[0].lineStyle.type).toBe("dashed");
    expect(series[0].lineStyle.color).not.toBe(series[2].lineStyle.color);
    expect(series).toHaveLength(3);
  });

  it("draws the head-to-head's other run raw as well, so neither is flattered", async () => {
    const series = await draw({
      returns: [5, 6],
      window: 2,
      yLabel: "steps",
      comparisons: [{ label: "REINFORCE, seed 1", returns: [1, 9, 2], raw: true }],
    });
    expect(series.map((s) => s.name)).toEqual([
      "REINFORCE, seed 1 (each episode)",
      "REINFORCE, seed 1",
      "this run (each episode)",
      "this run (mean of last 2)",
    ]);
    expect(series[0].data.map(([, y]) => y)).toEqual([1, 9, 2]);
  });

  it("thins a long run to a bounded number of points, keeping the last episode", async () => {
    const returns = Array.from({ length: 20_000 }, (_, i) => i % 2);
    const series = await draw({ returns, window: 100, yLabel: "success" });
    expect(series[0].data).toHaveLength(MAX_POINTS);
    expect(series[0].data[series[0].data.length - 1][0]).toBe(20_000);
  });
});
