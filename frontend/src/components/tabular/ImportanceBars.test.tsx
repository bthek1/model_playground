import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { EChartsOption } from "echarts";

// The chart itself needs a layout engine happy-dom does not have, so the option
// object is captured and asserted instead — which is where this component's own
// decisions live (ordering, colour by sign).
let captured: EChartsOption | null = null;
vi.mock("@/components/charts/EChart", () => ({
  default: ({ option }: { option: EChartsOption }) => {
    captured = option;
    return <div data-testid="echart" />;
  },
}));

// happy-dom resolves no CSS custom properties, so `getCSSVar` returns "" for
// every token and two different tokens become indistinguishable. Echoing the
// token name back is what makes "positive and negative use different colours"
// assertable at all.
vi.mock("@/lib/theme", () => ({
  getCSSVar: (token: string) => `var(--${token})`,
}));

const { ImportanceBars } = await import("./ImportanceBars");

const importance = [
  { name: "income", drop: 0.161 },
  { name: "debt_ratio", drop: 0.113 },
  { name: "has_mortgage", drop: 0.021 },
  { name: "postcode_noise", drop: -0.004 },
];

async function setup(rows = importance, unit = "accuracy lost") {
  captured = null;
  render(<ImportanceBars importance={rows} unit={unit} />);
  await waitFor(() => expect(screen.getByTestId("echart")).toBeInTheDocument());
}

function series() {
  const s = captured?.series;
  return (Array.isArray(s) ? s[0] : s) as { data: { value: number; itemStyle: { color: string } }[] };
}

describe("ImportanceBars", () => {
  it("orders the bars ascending, so the most important lands at the top", async () => {
    // ECharts draws a horizontal category axis bottom-up, so ascending data puts
    // the largest bar at the top of the chart.
    await setup();
    const categories = (captured?.yAxis as { data: string[] }).data;
    expect(categories).toEqual(["postcode_noise", "has_mortgage", "debt_ratio", "income"]);
    expect(series().data.map((d) => d.value)).toEqual([-0.004, 0.021, 0.113, 0.161]);
  });

  it("keeps a negative bar rather than clipping it, and colours it differently", async () => {
    // A negative drop is real: shuffling the column *helped*, so on this split the
    // model was being misled by it. Clamping it to zero hides a finding.
    await setup();
    const colours = series().data.map((d) => d.itemStyle.color);
    // The one negative bar is a different token from the three positive ones,
    // which are all the same.
    expect(colours[0]).not.toBe(colours[3]);
    expect(new Set(colours.slice(1)).size).toBe(1);
  });

  it("names the axis with the quantity the bars are in", async () => {
    // "accuracy lost" on a classifier and "R² lost" on a regressor — an unlabelled
    // bar length is not a measurement.
    await setup(importance, "R² lost");
    expect((captured?.xAxis as { name: string }).name).toBe("R² lost");
    expect(screen.getByTestId("importance-bars")).toHaveTextContent("R² lost");
  });

  it("explains the method, including what a near-zero and a negative bar mean", async () => {
    await setup();
    const panel = screen.getByTestId("importance-bars");
    expect(panel).toHaveTextContent(/shuffled on the held-out rows/i);
    expect(panel).toHaveTextContent(/did not use that column/i);
    expect(panel).toHaveTextContent(/negative.*left as it is rather than clipped/i);
  });

  it("states the causal caveat, which is the one the chart invites breaking", async () => {
    // "Which of my columns matter" is the question this chart answers, and the
    // answer is about the model rather than about the world.
    await setup();
    expect(screen.getByTestId("importance-bars")).toHaveTextContent(
      /not the same as what causes the outcome/i,
    );
    expect(screen.getByTestId("importance-bars")).toHaveTextContent(/share the credit/i);
  });

  it("renders an empty ranking without throwing", async () => {
    // Reachable: a fit stopped during the permutation pass returns no rows.
    await setup([]);
    expect((captured?.yAxis as { data: string[] }).data).toEqual([]);
  });
});
