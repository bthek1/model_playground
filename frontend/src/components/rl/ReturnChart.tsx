// The return curve: every episode's return, **unsmoothed**, with a running
// mean over it — plus the running mean of each earlier run kept in the
// history, so a second seed is drawn against the first rather than replacing it.
//
// The raw trace is not decoration. On FrozenLake a return is 0 or 1, so the
// raw series is a barcode and the running mean is the success rate; on the
// policy-gradient pages the raw trace's violence *is* the argument. A chart
// that smoothed it away would remove what the page is for.

import type { EChartsOption } from "echarts";
import { lazy, Suspense, useMemo } from "react";

import { useTheme } from "@/hooks/useTheme";
import { getCSSVar } from "@/lib/theme";

import { runningMean, strideIndices } from "./series";

const EChart = lazy(() => import("@/components/charts/EChart"));

export interface ComparisonLine {
  label: string;
  returns: readonly number[];
  /** Draw its raw trace too, not only its running mean. */
  raw?: boolean;
}

export function ReturnChart({
  returns,
  window,
  yLabel,
  label = "this run",
  comparisons = [],
}: {
  returns: readonly number[];
  window: number;
  yLabel: string;
  /** The current run's legend name. */
  label?: string;
  /** Earlier runs, drawn as running means (and raw, if asked). */
  comparisons?: readonly ComparisonLine[];
}) {
  const { theme } = useTheme();
  const option = useMemo<EChartsOption>(() => {
    const axis = getCSSVar("mutedForeground");
    const raw = getCSSVar("chart1");
    const mean = getCSSVar("chart3");
    const others = [getCSSVar("chart2"), getCSSVar("chart4"), getCSSVar("chart5")];

    const toPairs = (values: readonly number[]) =>
      strideIndices(values.length).map((i) => [i + 1, values[i]] as [number, number]);

    const series: NonNullable<EChartsOption["series"]> = comparisons.flatMap((c, i) => {
      const color = others[i % others.length];
      const mean = {
        name: c.label,
        type: "line" as const,
        showSymbol: false,
        lineStyle: { color, width: 1.6, type: "dashed" as const },
        itemStyle: { color },
        data: toPairs(runningMean(c.returns, window)),
      };
      // The head-to-head draws the other run's raw trace too: comparing one
      // run's noise against the other's smoothed mean would flatter the mean.
      if (!c.raw) return [mean];
      return [
        {
          name: `${c.label} (each episode)`,
          type: "line" as const,
          showSymbol: false,
          lineStyle: { color, width: 0.7, opacity: 0.4 },
          itemStyle: { color },
          data: toPairs(c.returns),
        },
        mean,
      ];
    });
    series.push(
      {
        name: `${label} (each episode)`,
        type: "line",
        showSymbol: false,
        lineStyle: { color: raw, width: 0.8, opacity: 0.55 },
        itemStyle: { color: raw },
        data: toPairs(returns),
      },
      {
        name: `${label} (mean of last ${window})`,
        type: "line",
        showSymbol: false,
        lineStyle: { color: mean, width: 2 },
        itemStyle: { color: mean },
        data: toPairs(runningMean(returns, window)),
      },
    );

    return {
      grid: { left: 8, right: 16, top: 30, bottom: 24, containLabel: true },
      legend: { top: 0, textStyle: { color: axis }, type: "scroll" },
      xAxis: {
        type: "value",
        name: "episode",
        nameLocation: "middle",
        nameGap: 22,
        min: 1,
        axisLabel: { color: axis },
        nameTextStyle: { color: axis },
      },
      yAxis: {
        type: "value",
        scale: true,
        axisLabel: { color: axis },
      },
      tooltip: { trigger: "axis" },
      animation: false,
      series,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [returns, window, yLabel, label, comparisons, theme]);

  return (
    <div className="h-56 w-full" data-testid="return-chart" aria-label={`${yLabel}, per episode`}>
      <Suspense fallback={<p className="text-xs text-muted-foreground">Loading chart…</p>}>
        <EChart option={option} />
      </Suspense>
    </div>
  );
}
