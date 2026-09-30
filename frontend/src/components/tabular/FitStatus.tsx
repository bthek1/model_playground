// The FIT band — slot 2, which on every other page in the app is LOAD.
//
// The substitution is the route's, not the machine's: Machine A is untouched
// (`idle → loading → ready | error`), and what this band renders is the *run*
// that follows it, because on this page the load costs nothing and the fit is
// what spends. `docs/standards/model-page-pattern.md` §7 records the swap.
//
// The bar is **determinate**, and that is the one place this category departs
// from `model/progress.ts`. That module is written around a byte count, and its
// indeterminate mode is the obvious reach for a page with no bytes — but a
// fit's total is not unknown: epochs, trees and rows are hyperparameters the
// user set a moment ago. An indeterminate bar in front of a number the page
// already has is worse than no bar. So the counter is route-owned, it reports
// `{ done, total }` over the shared envelope's `partial` arm, and
// `progress.ts` is neither touched nor reused.

import { CheckCircle2, Cpu, Loader2, PlayCircle, Sparkles, X } from "lucide-react";

import { ErrorNote } from "@/components/model/ErrorNote";
import { Button } from "@/components/ui/button";
import type { FitPartial, FitResult } from "@/tabular/types";

export function FitStatus({
  canFit,
  running,
  partial,
  result,
  error,
  onFit,
  onStop,
  blockedReason,
}: {
  canFit: boolean;
  running: boolean;
  partial: FitPartial | null;
  result: FitResult | null;
  error: string | null;
  onFit: () => void;
  onStop: () => void;
  /** Why FIT is unavailable, in the user's terms. Null when it is available. */
  blockedReason: string | null;
}) {
  const percent =
    partial && partial.total > 0
      ? Math.min(100, Math.round((partial.done / partial.total) * 100))
      : null;

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          size="sm"
          onClick={onFit}
          disabled={!canFit || running}
          data-testid="fit-button"
          className="xl:w-full"
        >
          {running ? (
            <Loader2 className="size-4 animate-spin" />
          ) : (
            <PlayCircle className="size-4" />
          )}
          {running ? "Fitting…" : result ? "Fit again" : "Fit model"}
        </Button>
        {running && (
          <Button size="sm" variant="outline" onClick={onStop} data-testid="fit-stop">
            <X className="size-4" /> Stop
          </Button>
        )}
      </div>

      {blockedReason && !running && (
        <p className="text-xs text-muted-foreground">{blockedReason}</p>
      )}

      {running && partial && (
        <div className="space-y-1" data-testid="fit-progress">
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
            <div
              className="h-full bg-primary transition-[width] duration-200"
              style={{ width: `${percent ?? 0}%` }}
            />
          </div>
          <p className="font-mono text-xs text-muted-foreground tabular-nums">
            {partial.phase} · {partial.done}/{partial.total}
            {partial.loss != null && ` · loss ${partial.loss.toFixed(4)}`}
          </p>
        </div>
      )}

      {!running && result && (
        <p
          data-testid="model-ready"
          className="flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground"
        >
          <CheckCircle2 className="size-3.5 text-emerald-600 dark:text-emerald-500" />
          Fitted in {(result.fitMs / 1000).toFixed(1)}s on{" "}
          {result.trainRows.toLocaleString()} rows ·
          <span className="inline-flex items-center gap-1 font-medium">
            {result.compute === "gpu" ? (
              <Sparkles className="size-3.5" />
            ) : (
              <Cpu className="size-3.5" />
            )}
            {result.compute === "gpu" ? "GPU" : "CPU"}
          </span>
        </p>
      )}

      <ErrorNote message={error} />
    </div>
  );
}
