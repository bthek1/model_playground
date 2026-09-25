// Predicted against actual, and the residual against the prediction.
//
// These replace the confusion matrix, and for the same reason it exists: **the
// aggregate number hides the failure.** A model with a respectable R² and a
// funnel-shaped residual plot is the case worth seeing — the error grows with
// the prediction, so every large estimate is worse than the single number
// suggests, and no scalar on the page says so. That is why the plot is a main
// panel here rather than an extra.
//
// Both charts go through the lazy `EChart` wrapper — `echarts` is 1.1 MB and one
// static import anywhere undoes the code-split that `npm run check:bundle`
// checks.

import type { EChartsOption } from "echarts";
import { lazy, Suspense, useMemo } from "react";

import { useTheme } from "@/hooks/useTheme";
import { getCSSVar } from "@/lib/theme";

const EChart = lazy(() => import("@/components/charts/EChart"));

/** At most this many points are drawn; beyond it the scatter is a blob anyway. */
const MAX_POINTS = 2000;

function thin(n: number): number[] {
  if (n <= MAX_POINTS) return Array.from({ length: n }, (_, i) => i);
  return Array.from({ length: MAX_POINTS }, (_, i) => Math.floor((i * n) / MAX_POINTS));
}

export function PredictedVsActual({
  predicted,
  actual,
  units,
  /** `quantiles.length × rows`, when the fitted family produced a band. */
  band,
  quantiles,
}: {
  predicted: Float32Array;
  actual: Float32Array;
  units: string;
  band?: Float32Array;
  quantiles?: number[];
}) {
  const { theme } = useTheme();
  const option = useMemo<EChartsOption>(() => {
    const axis = getCSSVar("mutedForeground");
    const point = getCSSVar("chart2");
    const guide = getCSSVar("chart4");
    const rows = actual.length;
    const idx = thin(rows);

    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < rows; i++) {
      min = Math.min(min, actual[i], predicted[i]);
      max = Math.max(max, actual[i], predicted[i]);
    }
    if (!Number.isFinite(min)) {
      min = 0;
      max = 1;
    }

    const series: NonNullable<EChartsOption["series"]> = [
      {
        type: "scatter",
        symbolSize: 5,
        itemStyle: { color: point, opacity: 0.55 },
        data: idx.map((i) => [actual[i], predicted[i]]),
        name: "rows",
      },
      {
        // The identity line. A perfect model lies on it, and the *shape* of the
        // departure from it is the thing a single R² cannot tell you.
        type: "line",
        showSymbol: false,
        lineStyle: { color: guide, type: "dashed", width: 1 },
        data: [
          [min, min],
          [max, max],
        ],
        name: "perfect",
      },
    ];

    if (band && quantiles && quantiles.length >= 2) {
      const lo = 0;
      const hi = quantiles.length - 1;
      series.push({
        type: "scatter",
        symbolSize: 3,
        itemStyle: { color: guide, opacity: 0.35 },
        data: idx.flatMap((i) => [
          [actual[i], band[lo * rows + i]],
          [actual[i], band[hi * rows + i]],
        ]),
        name: `q${quantiles[lo]} / q${quantiles[hi]}`,
      });
    }

    return {
      grid: { left: 8, right: 16, top: 8, bottom: 32, containLabel: true },
      xAxis: {
        type: "value",
        name: `actual (${units})`,
        scale: true,
        axisLabel: { color: axis },
        nameTextStyle: { color: axis },
      },
      yAxis: {
        type: "value",
        name: "predicted",
        scale: true,
        axisLabel: { color: axis },
        nameTextStyle: { color: axis },
      },
      tooltip: { trigger: "item" },
      animation: false,
      series,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [predicted, actual, units, band, quantiles, theme]);

  return (
    <div className="space-y-1" data-testid="predicted-vs-actual">
      <div className="h-56 w-full">
        <Suspense fallback={<p className="text-xs text-muted-foreground">Loading chart…</p>}>
          <EChart option={option} />
        </Suspense>
      </div>
      <p className="text-xs leading-snug text-muted-foreground">
        Every held-out row, plotted against the dashed identity line a perfect
        model would lie on. A cloud that bends away from the line at one end is a
        model that is systematically wrong there — which no single score reports.
      </p>
    </div>
  );
}

export function ResidualPlot({
  predicted,
  actual,
  units,
}: {
  predicted: Float32Array;
  actual: Float32Array;
  units: string;
}) {
  const { theme } = useTheme();
  const option = useMemo<EChartsOption>(() => {
    const axis = getCSSVar("mutedForeground");
    const point = getCSSVar("chart1");
    const guide = getCSSVar("chart4");
    const rows = actual.length;
    const idx = thin(rows);
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < rows; i++) {
      min = Math.min(min, predicted[i]);
      max = Math.max(max, predicted[i]);
    }
    if (!Number.isFinite(min)) {
      min = 0;
      max = 1;
    }
    return {
      grid: { left: 8, right: 16, top: 8, bottom: 32, containLabel: true },
      xAxis: {
        type: "value",
        name: `predicted (${units})`,
        scale: true,
        axisLabel: { color: axis },
        nameTextStyle: { color: axis },
      },
      yAxis: {
        type: "value",
        name: "residual",
        scale: true,
        axisLabel: { color: axis },
        nameTextStyle: { color: axis },
      },
      tooltip: { trigger: "item" },
      animation: false,
      series: [
        {
          type: "scatter",
          symbolSize: 5,
          itemStyle: { color: point, opacity: 0.55 },
          data: idx.map((i) => [predicted[i], actual[i] - predicted[i]]),
        },
        {
          type: "line",
          showSymbol: false,
          lineStyle: { color: guide, type: "dashed", width: 1 },
          data: [
            [min, 0],
            [max, 0],
          ],
        },
      ],
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [predicted, actual, units, theme]);

  return (
    <div className="space-y-1" data-testid="residual-plot">
      <div className="h-48 w-full">
        <Suspense fallback={<p className="text-xs text-muted-foreground">Loading chart…</p>}>
          <EChart option={option} />
        </Suspense>
      </div>
      <p className="text-xs leading-snug text-muted-foreground">
        Residuals should be a shapeless band around zero. A <em>funnel</em> — wider
        at one end — means the error grows with the prediction, so the large
        estimates are worse than the RMSE above suggests. That is the failure a
        good-looking R² hides, and it is why this plot is here rather than in an
        appendix.
      </p>
    </div>
  );
}
