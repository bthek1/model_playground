// The decision threshold — the single most useful control in this category, and
// the clearest example of a control that **re-derives and never re-fits**.
//
// 0.5 is a convention, not a decision. Whether a 0.5 cut is right depends on
// what a false positive costs relative to a false negative, which is a question
// about the user's problem and not about the model — so the page hands it over,
// and moving it recomputes precision, recall, F1 and the whole confusion matrix
// from the held-out probabilities already in hand, on the main thread, with no
// worker message at all. Same trick `/vad`'s threshold and detection's score
// floor use.
//
// Binary only. At three or more classes the prediction is an argmax and there is
// no single number to move; a "threshold" there would be a different control
// (one per class, or a cost matrix) pretending to be this one.

import type { ClassificationMetrics } from "@/tabular/types";

export function ThresholdControl({
  threshold,
  onChange,
  positiveLabel,
  metrics,
}: {
  threshold: number;
  onChange: (t: number) => void;
  positiveLabel: string;
  metrics: ClassificationMetrics;
}) {
  return (
    <div className="space-y-1.5" data-testid="threshold-control">
      <label
        htmlFor="decision-threshold"
        className="flex items-baseline justify-between text-xs font-medium"
      >
        Decision threshold for “{positiveLabel}”
        <span className="font-mono text-xs text-muted-foreground tabular-nums">
          {threshold.toFixed(2)}
        </span>
      </label>
      <input
        id="decision-threshold"
        type="range"
        min={0.01}
        max={0.99}
        step={0.01}
        value={threshold}
        onChange={(e) => onChange(Number(e.target.value))}
        className="block w-full"
      />
      <p className="font-mono text-xs text-muted-foreground tabular-nums">
        precision {(metrics.precision * 100).toFixed(1)}% · recall{" "}
        {(metrics.recall * 100).toFixed(1)}% · F1 {(metrics.f1 * 100).toFixed(1)}%
      </p>
      <p className="text-xs leading-snug text-muted-foreground">
        0.5 is a convention, not a decision — it is only right when a false
        positive and a false negative cost the same. Moving this re-reads the
        held-out probabilities the fit already produced; it does not refit, and
        it costs nothing.
      </p>
    </div>
  );
}
