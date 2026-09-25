// Regression scores, each with the null model beside it — and each labelled
// with the units it is in.
//
// The units are the point. A fit on `log1p(target)` produces a smaller RMSE for
// the same reason a logarithm is smaller, and reading that next to a raw fit's
// RMSE is the mistake `/tabular-regression` exists to demonstrate. So this
// component renders whatever `transform.ts` put in `metrics.units` and never
// derives it: a route that could compose the label itself could compose it
// wrongly.
//
// The null model for regression is predicting the **training mean**. R² is that
// comparison already, but it is unitless and it is computed against the *test*
// mean — so the baseline's own RMSE and MAE are shown in the target's units
// beside the model's, out of the same call, so the two cannot come from
// different splits.

import { Minus, TrendingDown, TrendingUp } from "lucide-react";

import type { RegressionMetrics as Metrics } from "@/tabular/types";

/** Four significant figures, which is more than a held-out estimate supports. */
function num(x: number): string {
  if (!Number.isFinite(x)) return "—";
  const abs = Math.abs(x);
  if (abs >= 1000) return x.toFixed(0);
  if (abs >= 1) return x.toFixed(2);
  return x.toPrecision(3);
}

export function RegressionMetricBlock({
  metrics,
  logSpace,
}: {
  metrics: Metrics;
  /**
   * The same fit scored in the space it was actually fitted in, when that
   * differs. Rendered as a clearly separate block, never inline beside the
   * numbers above — that juxtaposition is the error being demonstrated.
   */
  logSpace?: Metrics;
}) {
  const better = metrics.baselineRmse - metrics.rmse;
  const Icon = better > 0 ? TrendingUp : better < 0 ? TrendingDown : Minus;
  const tone =
    better > 0
      ? "text-emerald-600 dark:text-emerald-500"
      : "text-red-600 dark:text-red-500";

  return (
    <div className="space-y-2" data-testid="regression-metrics">
      <div className="grid grid-cols-3 gap-2">
        <Stat label="RMSE" value={num(metrics.rmse)} unit={metrics.units} primary />
        <Stat label="MAE" value={num(metrics.mae)} unit={metrics.units} />
        <Stat label="R²" value={metrics.r2.toFixed(3)} unit="unitless" />
      </div>
      <p className={`flex items-start gap-1.5 text-xs ${tone}`}>
        <Icon className="mt-0.5 size-3.5 shrink-0" />
        <span data-testid="regression-baseline">
          Predicting the training mean — the null model — scores{" "}
          <strong>
            {num(metrics.baselineRmse)} {metrics.units}
          </strong>{" "}
          RMSE on these same held-out rows.{" "}
          {better > 0
            ? `This model is ${num(better)} better.`
            : "This model has not beaten it, so it has learned the average and nothing else."}
        </span>
      </p>

      {logSpace && (
        <div
          className="space-y-1 rounded-md border border-amber-600/40 bg-amber-600/5 p-2"
          data-testid="log-space-metrics"
        >
          <p className="text-xs font-medium text-amber-700 dark:text-amber-500">
            The same fit, scored in the space it was fitted in
          </p>
          <p className="font-mono text-xs tabular-nums">
            RMSE {num(logSpace.rmse)} · MAE {num(logSpace.mae)} · R²{" "}
            {logSpace.r2.toFixed(3)}
            <span className="ml-1 font-sans text-muted-foreground">
              — in {logSpace.units}
            </span>
          </p>
          <p className="text-xs leading-snug text-muted-foreground">
            <strong>These numbers are not comparable with the ones above.</strong>{" "}
            They are errors in log units, and a log-space RMSE is smaller for the
            same reason a logarithm is smaller. Comparing them with a raw fit's
            RMSE and concluding the transform helped is the mistake this toggle
            exists to show. The comparable numbers are the ones at the top: the
            predictions back-transformed and re-scored in {metrics.units}.
          </p>
        </div>
      )}
    </div>
  );
}

function Stat({
  label,
  value,
  unit,
  primary = false,
}: {
  label: string;
  value: string;
  unit: string;
  primary?: boolean;
}) {
  return (
    <div className="rounded-md border p-2">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p
        className={
          primary
            ? "text-xl font-semibold tabular-nums"
            : "text-base font-medium tabular-nums"
        }
      >
        {value}
      </p>
      <p className="truncate text-xs text-muted-foreground" title={unit}>
        {unit}
      </p>
    </div>
  );
}
