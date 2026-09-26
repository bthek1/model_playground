import { render, screen, waitFor } from "@testing-library/react";
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

const { SeriesChart } = await import("./SeriesChart");
const { parseSeries } = await import("@/forecast/series");

type Series = {
  name?: string;
  data: (number | null)[];
  markArea?: { data: { xAxis: string }[][] };
};

const dated = parseSeries(
  Array.from({ length: 10 }, (_, i) => `2024-0${i < 9 ? i + 1 : 9}-01,${10 + i}`).join("\n"),
  "ten",
);
const bare = parseSeries(Array.from({ length: 10 }, (_, i) => String(10 + i)).join("\n"));

const lines = [
  { label: "Naive", values: Float32Array.from([17, 17, 17]) },
  { label: "Drift", values: Float32Array.from([18, 19, 20]) },
];

async function setup(series = dated, testStart = 7) {
  captured.length = 0;
  render(<SeriesChart series={series} testStart={testStart} lines={lines} />);
  await waitFor(() => expect(screen.getByTestId("series-chart")).toBeInTheDocument());
  return (captured[0].series as Series[]) ?? [];
}

describe("SeriesChart", () => {
  it("draws the whole actual series first", async () => {
    const s = await setup();
    expect(s[0].name).toBe("actual");
    expect(s[0].data).toHaveLength(10);
    expect(s[0].data[0]).toBe(10);
  });

  it("marks the held-out region, so a forecast cannot read as a fit", async () => {
    // A forecast chart that does not show where the history stopped is a picture
    // of a model being tested on its training data, as far as the reader knows.
    const s = await setup();
    const area = s[0].markArea?.data[0];
    expect(area?.[0].xAxis).toBe("2024-08");
    expect(area?.[1].xAxis).toBe("2024-09");
  });

  it("draws each forecast only across the held-out region", async () => {
    // Plotting from index 0 would draw a line over the history the method was
    // fitted on — the same mistake the markArea exists to prevent.
    const s = await setup();
    expect(s).toHaveLength(3);
    for (const line of s.slice(1)) {
      expect(line.data.slice(0, 6).every((v) => v === null)).toBe(true);
      expect(line.data.slice(7)).toHaveLength(3);
    }
    expect(s[1].data.slice(7)).toEqual([17, 17, 17]);
    expect(s[2].data.slice(7)).toEqual([18, 19, 20]);
  });

  it("anchors each forecast to the last actual point, so it visibly leaves the history", async () => {
    const s = await setup();
    expect(s[1].data[6]).toBe(16);
    expect(s[2].data[6]).toBe(16);
  });

  it("labels the axis by date when the series has one", async () => {
    await setup();
    expect((captured[0].xAxis as { data: string[] }).data[0]).toBe("2024-01");
  });

  it("labels the axis by position when it does not", async () => {
    // A pasted column of numbers has no time axis, and inventing one would imply
    // a frequency the series never declared.
    await setup(bare);
    expect((captured[0].xAxis as { data: string[] }).data[0]).toBe("1");
  });

  it("says the shaded region is held out", async () => {
    await setup();
    expect(screen.getByTestId("series-chart")).toHaveTextContent(/shaded region is held out/i);
    expect(screen.getByTestId("series-chart")).toHaveTextContent(/a picture of a fit/i);
  });

  it("does not run off the end when the forecast is longer than the held-out region", async () => {
    // Reachable while the horizon slider is being dragged down: the forecast in
    // hand is still the old, longer one for a render.
    const s = await setup(dated, 9);
    expect(s[1].data).toHaveLength(10);
    expect(s[1].data.filter((v) => v != null)).toHaveLength(2);
  });
});
