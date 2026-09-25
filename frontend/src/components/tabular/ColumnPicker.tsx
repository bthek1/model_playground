// Target and features: which column is the answer, and which are allowed to
// predict it.
//
// Two rules are enforced here rather than left to the engine, because both are
// silent failures if they slip. The target is **never** offered as a feature —
// a model handed its own answer scores 1.00 and teaches nothing — and only
// categorical columns are offered as a classification target, because a numeric
// column with 300 distinct values is a regression problem wearing the wrong hat
// and would produce 300 classes of one row each.

import { Button } from "@/components/ui/button";
import type { Dataset, Objective } from "@/tabular/types";

export function ColumnPicker({
  dataset,
  objective,
  target,
  onTarget,
  features,
  onFeatures,
  disabled = false,
}: {
  dataset: Dataset;
  objective: Objective;
  target: number;
  onTarget: (index: number) => void;
  features: number[];
  onFeatures: (indices: number[]) => void;
  disabled?: boolean;
}) {
  const eligible = dataset.columns
    .map((c, i) => ({ c, i }))
    .filter(({ c }) =>
      objective === "classification" ? c.kind === "categorical" : c.kind === "numeric",
    );

  return (
    <div className="space-y-3" data-testid="column-picker">
      <div className="space-y-1.5">
        <label htmlFor="target-column" className="text-xs font-medium">
          Target — the column to predict
        </label>
        <select
          id="target-column"
          value={target}
          disabled={disabled}
          onChange={(e) => onTarget(Number(e.target.value))}
          className="h-8 w-full rounded-md border bg-background px-2 text-xs"
        >
          {eligible.map(({ c, i }) => (
            <option key={c.name} value={i}>
              {c.name}
              {c.kind === "categorical" && c.levels
                ? ` (${c.levels.length} values)`
                : ""}
            </option>
          ))}
        </select>
        {eligible.length === 0 && (
          <p className="text-xs text-amber-600 dark:text-amber-500">
            {objective === "classification"
              ? "No text column in this file to classify. A classification target needs a small set of distinct values."
              : "No numeric column in this file to predict."}
          </p>
        )}
      </div>

      <div className="space-y-1.5">
        <div className="flex items-baseline justify-between">
          <p className="text-xs font-medium">
            Features — {features.length} of {dataset.columns.length - 1}
          </p>
          <div className="flex gap-1">
            <Button
              size="sm"
              variant="ghost"
              disabled={disabled}
              onClick={() =>
                onFeatures(
                  dataset.columns.map((_, i) => i).filter((i) => i !== target),
                )
              }
            >
              All
            </Button>
            <Button
              size="sm"
              variant="ghost"
              disabled={disabled}
              onClick={() => onFeatures([])}
            >
              None
            </Button>
          </div>
        </div>
        <div className="flex flex-wrap gap-1">
          {dataset.columns.map((c, i) =>
            i === target ? null : (
              <Button
                key={c.name}
                size="sm"
                variant={features.includes(i) ? "secondary" : "outline"}
                aria-pressed={features.includes(i)}
                disabled={disabled}
                onClick={() =>
                  onFeatures(
                    features.includes(i)
                      ? features.filter((f) => f !== i)
                      : [...features, i].sort((a, b) => a - b),
                  )
                }
                className="h-7 px-2 text-xs"
              >
                {c.name}
                {c.missingCount > 0 && (
                  <span className="ml-1 text-muted-foreground">
                    ·{c.missingCount} missing
                  </span>
                )}
              </Button>
            ),
          )}
        </div>
      </div>
    </div>
  );
}
