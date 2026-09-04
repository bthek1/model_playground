// One rendering for every error on a model page. The pattern requires an error
// to appear *in the slot that produced it* (model-page-pattern.md §4), so this
// is deliberately a small inline block rather than a page-level banner: it can
// sit inside LOAD, RUN, or OUTPUT without changing the page's shape.

import { AlertTriangle } from "lucide-react";
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

export function ErrorNote({
  message,
  hint,
  detail,
  action,
  className,
}: {
  /** Nothing renders when null — callers can pass an optional error straight in. */
  message: string | null | undefined;
  /** What to try next, when that isn't obvious from the message. */
  hint?: ReactNode;
  /**
   * The raw underlying error, folded away. A classified message is friendlier
   * but useless in a bug report, so the original is always one click away —
   * never swallowed (see `model/errors.ts`).
   */
  detail?: string;
  /** Optional recovery affordance, e.g. a Retry button. */
  action?: ReactNode;
  className?: string;
}) {
  if (!message) return null;

  return (
    <div
      role="alert"
      data-testid="error-note"
      className={cn(
        "flex flex-wrap items-center gap-x-3 gap-y-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive",
        className,
      )}
    >
      <AlertTriangle className="size-4 shrink-0" />
      <div className="min-w-0 flex-1 space-y-1">
        <p className="break-words">{message}</p>
        {hint && <p className="text-xs opacity-80">{hint}</p>}
        {detail && (
          <details className="text-xs opacity-80">
            <summary className="cursor-pointer select-none">Details</summary>
            <p className="mt-1 font-mono text-[0.7rem] break-all">{detail}</p>
          </details>
        )}
      </div>
      {action}
    </div>
  );
}
