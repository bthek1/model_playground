// The series, the forecast, and the held-out region marked.
//
// The marking is the part that matters. A forecast chart that does not show
// where the history stopped is a picture of a model being tested on data it was
// fitted on, as far as the reader can tell — and on a time series that is the
// failure mode, not a pedantic distinction.

import type { EChartsOption } from "echarts";
import { lazy, Suspense, useMemo } from "react";

import { useTheme } from "@/hooks/useTheme";
import { getCSSVar } from "@/lib/theme";
import type { Series } from "@/forecast/series";
import { pointLabel } from "@/forecast/series";

const EChart = lazy(() => import("@/components/charts/EChart"));

export interface ForecastLine {
  label: string;
  /** One value per held-out point, aligned with `testStart`. */
  values: Float32Array;
}

export function SeriesChart({
  series,
  testStart,
  lines,
}: {
  series: Series;
  /** Index where the held-out region begins. */
  testStart: number;
  lines: ForecastLine[];
}) {
  const { theme } = useTheme();
  const option = useMemo<EChartsOption>(() => {
    const axis = getCSSVar("mutedForeground");
    const history = getCSSVar("chart1");
    const palette = [
      getCSSVar("chart2"),
      getCSSVar("chart3"),
      getCSSVar("chart4"),
      getCSSVar("chart5"),
    ];
    const n = series.values.length;
    const labels = Array.from({ length: n }, (_, i) => pointLabel(series, i));

    const series0: NonNullable<EChartsOption["series"]> = [
      {
        name: "actual",
        type: "line",
        showSymbol: false,
        lineStyle: { color: history, width: 1.5 },
        itemStyle: { color: history },
        data: Array.from(series.values),
        markArea: {
          silent: true,
          itemStyle: { color: axis, opacity: 0.08 },
          data: [[{ xAxis: labels[testStart] }, { xAxis: labels[n - 1] }]],
        },
      },
    ];

    lines.forEach((line, i) => {
      // Each forecast is drawn only over the held-out region, with a `null`
      // prefix — plotting it from index 0 would draw a line across the history
      // the model was fitted on and read as a fit rather than a forecast.
      const data: (number | null)[] = new Array(n).fill(null);
      for (let k = 0; k < line.values.length && testStart + k < n; k++) {
        data[testStart + k] = line.values[k];
      }
      // One point of overlap so the forecast visibly leaves the history.
      if (testStart > 0) data[testStart - 1] = series.values[testStart - 1];
      series0.push({
        name: line.label,
        type: "line",
        showSymbol: false,
        lineStyle: { color: palette[i % palette.length], width: 1.5, type: "dashed" },
        itemStyle: { color: palette[i % palette.length] },
        data,
      });
    });

    return {
      grid: { left: 8, right: 16, top: 28, bottom: 28, containLabel: true },
      legend: { textStyle: { color: axis }, top: 0 },
      xAxis: {
        type: "category",
        data: labels,
        axisLabel: { color: axis, showMaxLabel: true },
      },
      yAxis: { type: "value", scale: true, axisLabel: { color: axis } },
      tooltip: { trigger: "axis" },
      animation: false,
      series: series0,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [series, testStart, lines, theme]);

  return (
    <div className="space-y-1" data-testid="series-chart">
      <div className="h-64 w-full">
        <Suspense fallback={<p className="text-xs text-muted-foreground">Loading chart…</p>}>
          <EChart option={option} />
        </Suspense>
      </div>
      <p className="text-xs text-muted-foreground">
        The shaded region is held out — every forecast is drawn only across it,
        because a line drawn over the history is a picture of a fit, not a
        forecast.
      </p>
    </div>
  );
}
