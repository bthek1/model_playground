// Permutation importance, as a bar per column.
//
// This is the output non-technical users actually read: "which of my columns
// mattered" is the question a spreadsheet cannot answer and a fitted model can.
// It is also the one that needs its caveat *next to it* rather than in a doc —
// importance measured by shuffling says what the model used, which is not the
// same as what causes the outcome, and the distinction is the difference
// between a useful chart and a wrong decision.
//
// Bars are grouped by the column the user chose, not by design column: a
// one-hot city with eleven levels is one thing to them, and eleven bars reading
// "city = Wellington" would bury every genuine feature underneath it.
//
// `echarts` is reached only through the lazy `EChart` wrapper — one static
// import anywhere undoes the code-split, and `npm run check:bundle` is what
// fails then.

import type { EChartsOption } from "echarts";
import { lazy, Suspense, useMemo } from "react";

import { getCSSVar } from "@/lib/theme";
import { useTheme } from "@/hooks/useTheme";
import type { FeatureImportance } from "@/tabular/types";

const EChart = lazy(() => import("@/components/charts/EChart"));

export function ImportanceBars({
  importance,
  /** What a bar's length means — "accuracy lost" or "R² lost". */
  unit,
}: {
  importance: FeatureImportance[];
  unit: string;
}) {
  const { theme } = useTheme();
  const option = useMemo<EChartsOption>(() => {
    const axis = getCSSVar("mutedForeground");
    const positive = getCSSVar("chart2");
    const negative = getCSSVar("chart5");
    // Ascending, because ECharts draws a horizontal bar axis bottom-up and the
    // most important column belongs at the top.
    const rows = [...importance].sort((a, b) => a.drop - b.drop);
    return {
      grid: { left: 8, right: 24, top: 8, bottom: 28, containLabel: true },
      xAxis: {
        type: "value",
        name: unit,
        axisLabel: { color: axis },
        nameTextStyle: { color: axis },
      },
      yAxis: {
        type: "category",
        data: rows.map((r) => r.name),
        axisLabel: { color: axis },
      },
      tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
      animation: false,
      series: [
        {
          type: "bar",
          data: rows.map((r) => ({
            value: Number(r.drop.toFixed(4)),
            itemStyle: { color: r.drop >= 0 ? positive : negative },
          })),
        },
      ],
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [importance, unit, theme]);

  return (
    <div className="space-y-1" data-testid="importance-bars">
      <div className="h-56 w-full">
        <Suspense
          fallback={
            <p className="text-xs text-muted-foreground">Loading chart…</p>
          }
        >
          <EChart option={option} />
        </Suspense>
      </div>
      <p className="text-xs leading-snug text-muted-foreground">
        Each column was shuffled on the held-out rows and the model re-scored;
        the bar is how much {unit} that cost. A bar near zero means the model did
        not use that column. A <em>negative</em> bar is left as it is rather than
        clipped — it means shuffling the column helped, so on this split the
        model was being misled by it.
      </p>
      <p className="text-xs leading-snug text-muted-foreground">
        This is what the model used, which is not the same as what causes the
        outcome. Two columns carrying the same information will share the credit
        and both look unimportant.
      </p>
    </div>
  );
}
