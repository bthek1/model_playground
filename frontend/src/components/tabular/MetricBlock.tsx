// Scores, each next to the null model that makes it readable.
//
// The baseline is rendered **beside every accuracy**, not under a "details"
// toggle, and it arrives inside the same metrics object so the two cannot come
// from different splits. `/graph-classification` is the precedent and the
// reason: PROTEINS is 663/450, so a classifier that ignores the molecule
// entirely scores 0.598 — a number that looks like a result. A class prior is
// the easiest thing in any dataset to learn, which is exactly what a broken
// readout produces.
//
// The verdict line is deliberately in words. "0.731 against 0.556" is a
// comparison the reader still has to make; "27 points above" is the finding.

import { Minus, TrendingDown, TrendingUp } from "lucide-react";

import type { ClassificationMetrics } from "@/tabular/types";

function pct(x: number): string {
  return `${(x * 100).toFixed(1)}%`;
}

export function MetricBlock({ metrics }: { metrics: ClassificationMetrics }) {
  const lift = metrics.accuracy - metrics.baselineAccuracy;
  const points = Math.abs(lift * 100);
  const Icon = lift > 0.01 ? TrendingUp : lift < -0.01 ? TrendingDown : Minus;
  const tone =
    lift > 0.01
      ? "text-emerald-600 dark:text-emerald-500"
      : lift < -0.01
        ? "text-red-600 dark:text-red-500"
        : "text-amber-600 dark:text-amber-500";

  return (
    <div className="space-y-2" data-testid="metric-block">
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="Accuracy" value={pct(metrics.accuracy)} primary />
        <Stat label="Precision" value={pct(metrics.precision)} hint="macro" />
        <Stat label="Recall" value={pct(metrics.recall)} hint="macro" />
        <Stat label="F1" value={pct(metrics.f1)} hint="macro" />
      </div>
      <p className={`flex items-start gap-1.5 text-xs ${tone}`}>
        <Icon className="mt-0.5 size-3.5 shrink-0" />
        <span data-testid="baseline-note">
          Always answering <strong>{metrics.baselineLabel}</strong> — the
          training half's most common class — scores{" "}
          <strong>{pct(metrics.baselineAccuracy)}</strong> on these same held-out
          rows.{" "}
          {lift > 0.01
            ? `This model is ${points.toFixed(1)} points above it.`
            : lift < -0.01
              ? `This model is ${points.toFixed(1)} points below it, so it is worse than guessing the majority.`
              : "This model has not beaten it, so it has learned the class prior and nothing else."}
        </span>
      </p>
    </div>
  );
}

function Stat({
  label,
  value,
  hint,
  primary = false,
}: {
  label: string;
  value: string;
  hint?: string;
  primary?: boolean;
}) {
  return (
    <div className="rounded-md border p-2">
      <p className="text-xs text-muted-foreground">
        {label}
        {hint && <span className="ml-1 opacity-70">({hint})</span>}
      </p>
      <p
        className={
          primary
            ? "text-xl font-semibold tabular-nums"
            : "text-base font-medium tabular-nums"
        }
      >
        {value}
      </p>
    </div>
  );
}
