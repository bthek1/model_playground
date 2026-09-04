// The OUTPUT slot. Always renders — that is the point. A section that appears
// only once a result exists makes the page jump under the user at the moment
// they are least able to absorb it, and leaves a first-time visitor with no idea
// what they are about to get (docs/standards/model-page-pattern.md §4).
//
// Four states in one place: empty → running → result, with error alongside.

import { Loader2 } from "lucide-react";
import type { ReactNode } from "react";

import { ErrorNote } from "@/components/model/ErrorNote";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export function OutputPanel({
  title,
  description,
  meta,
  actions,
  running,
  runningLabel = "Running…",
  error,
  empty,
  children,
}: {
  title: ReactNode;
  description?: ReactNode;
  /** Small right-aligned facts about the result — duration, sample rate, shape. */
  meta?: ReactNode;
  /** Result-scoped buttons: Play, Download, Copy. Hidden until there is a result. */
  actions?: ReactNode;
  running: boolean;
  runningLabel?: string;
  /** Inference error. Rendered here, and the page stays usable. */
  error?: string | null;
  /** What the user will get. Shown when there is no result and nothing running. */
  empty: ReactNode;
  /** The result. Absent/null means "no result yet". */
  children?: ReactNode;
}) {
  const hasResult = children != null && children !== false;

  return (
    <Card
      data-testid="output-panel"
      className="flex min-h-0 flex-1 flex-col overflow-hidden"
    >
      <CardHeader className="shrink-0">
        <CardTitle className="flex flex-wrap items-center gap-2 text-base">
          {title}
          {meta && (
            <span className="font-mono text-xs font-normal text-muted-foreground tabular-nums">
              {meta}
            </span>
          )}
        </CardTitle>
        {description && <CardDescription>{description}</CardDescription>}
      </CardHeader>

      <CardContent className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto">
        <ErrorNote message={error} />

        {running && !hasResult && (
          <div
            data-testid="output-running"
            className="flex items-center gap-2 text-sm text-muted-foreground"
          >
            <Loader2 className="size-4 animate-spin" />
            {runningLabel}
          </div>
        )}

        {hasResult ? (
          <>
            {children}
            {actions && (
              <div className="mt-auto flex shrink-0 flex-wrap gap-2 pt-2">
                {actions}
              </div>
            )}
          </>
        ) : (
          !running && (
            // The column is tall now, so the "what you'll get" state gets the
            // middle of it rather than clinging to the top edge.
            <p
              data-testid="output-empty"
              className="flex flex-1 items-center justify-center text-center text-sm text-balance text-muted-foreground"
            >
              {empty}
            </p>
          )
        )}
      </CardContent>
    </Card>
  );
}
