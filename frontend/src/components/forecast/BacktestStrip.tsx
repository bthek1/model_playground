// One point per rolling-origin window, with the single-split number drawn
// across it as a line.
//
// **This chart is the page.** Everything else — the baselines, the horizon, the
// season control — exists so that there is something to backtest. The finding is
// the gap between the one number a notebook would report and the distribution of
// numbers the same method produces when the cut moves a few steps, and it is
// only a finding if both are on screen at once. A page that showed only the
// spread would be correct and would not make the point; a page that showed only
// the single number would be every other forecasting tutorial.

import type { EChartsOption } from "echarts";
import { lazy, Suspense, useMemo } from "react";

import { useTheme } from "@/hooks/useTheme";
import { getCSSVar } from "@/lib/theme";
import { spread } from "@/forecast/metrics";
import type { BacktestResult } from "@/forecast/backtest";

const EChart = lazy(() => import("@/components/charts/EChart"));

export type MetricKey = "mae" | "rmse" | "mase";

const METRIC_LABEL: Record<MetricKey, string> = {
  mae: "MAE",
  rmse: "RMSE",
  mase: "MASE",
};

export function BacktestStrip({
  result,
  metric,
}: {
  result: BacktestResult;
  metric: MetricKey;
}) {
  const { theme } = useTheme();
  const values = useMemo(
    () =>
      result.windows
        .map((w) => (metric === "mase" ? w.metrics.mase : w.metrics[metric]))
        .filter((v): v is number => v != null && Number.isFinite(v)),
    [result, metric],
  );
  const single =
    metric === "mase" ? result.single.mase : result.single[metric];
  const stats = useMemo(() => spread(values), [values]);

  const option = useMemo<EChartsOption>(() => {
    const axis = getCSSVar("mutedForeground");
    const point = getCSSVar("chart2");
    const mark = getCSSVar("chart5");
    return {
      grid: { left: 8, right: 16, top: 16, bottom: 30, containLabel: true },
      xAxis: {
        type: "category",
        name: "window",
        data: result.windows.map((_, i) => String(i + 1)),
        axisLabel: { color: axis },
        nameTextStyle: { color: axis },
      },
      yAxis: {
        type: "value",
        name: METRIC_LABEL[metric],
        scale: true,
        axisLabel: { color: axis },
        nameTextStyle: { color: axis },
      },
      tooltip: { trigger: "axis" },
      animation: false,
      series: [
        {
          type: "bar",
          itemStyle: { color: point },
          data: result.windows.map((w) =>
            metric === "mase" ? (w.metrics.mase ?? 0) : w.metrics[metric],
          ),
          markLine:
            single == null
              ? undefined
              : {
                  silent: true,
                  symbol: "none",
                  lineStyle: { color: mark, type: "solid", width: 2 },
                  label: { formatter: "one split", color: mark },
                  data: [{ yAxis: single }],
                },
        },
      ],
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result, metric, single, theme]);

  const ratio = stats.min > 0 ? stats.max / stats.min : null;

  return (
    <div className="space-y-1" data-testid="backtest-strip">
      <div className="h-48 w-full">
        <Suspense fallback={<p className="text-xs text-muted-foreground">Loading chart…</p>}>
          <EChart option={option} />
        </Suspense>
      </div>
      <p className="font-mono text-xs tabular-nums" data-testid="backtest-spread">
        one split {single?.toFixed(3) ?? "—"} · windows {stats.min.toFixed(3)}–
        {stats.max.toFixed(3)} · median {stats.median.toFixed(3)}
      </p>
      <p className="text-xs leading-snug text-muted-foreground">
        The solid line is the number a single train/test split reports — the one
        a notebook prints. The bars are the same method, the same series and the
        same horizon, with the cut moved a few steps each time.
        {ratio != null && ratio > 1.3 && (
          <>
            {" "}
            Here the worst window is{" "}
            <strong>{ratio.toFixed(1)}× the best</strong>, which is usually
            larger than the difference between the methods being compared. That
            is what "one split lies" means.
          </>
        )}
      </p>
    </div>
  );
}
