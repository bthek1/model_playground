// The GENERATE trigger — the one button in RUN that spends (§1.2).
//
// Thirty-five routes wrote the same button by hand: a spinner and a present-
// participle label while the run is in flight, an icon and a verb otherwise,
// and an `onClick={() => void run().catch(() => {})}` whose `.catch` exists only
// because the hook already reports the failure in OUTPUT and an unhandled
// rejection would report it a second time, in the console.
//
// `onRun` may return a promise; a rejection is swallowed here for that reason,
// and nothing else is. Everything a `Button` takes — `variant`, `title`, a
// `data-testid` — passes straight through, so a second trigger on the same page
// (`/fill-mask`'s probes, `/text-ranking`'s embed step) is a second RunButton.
//
// Gating stays with the route: `disabled` is whatever the page says, because
// "ready, and an input is held, and the label list is not empty" is the page's
// knowledge. Only GENERATE is gated on `ready`, never the input sources.

import type { LucideIcon } from "lucide-react";
import { Loader2 } from "lucide-react";
import type { ComponentProps, ReactNode } from "react";

import { Button } from "@/components/ui/button";

export type RunButtonProps = Omit<
  ComponentProps<typeof Button>,
  "onClick" | "children"
> & {
  /** The verb's icon, shown while idle. Omit for a bare verb. */
  icon?: LucideIcon;
  /** The verb — "Classify", "Generate". */
  children: ReactNode;
  /** A run is in flight: show the spinner and `runningLabel` instead. */
  running: boolean;
  /** "Classifying…" — what is happening, while it is. */
  runningLabel: ReactNode;
  /** Start the run. A rejected promise is the hook's to report, not ours. */
  onRun: () => unknown;
};

export function RunButton({
  icon: Icon,
  children,
  running,
  runningLabel,
  onRun,
  ...props
}: RunButtonProps) {
  const onClick = () => {
    const out = onRun();
    if (out instanceof Promise) out.catch(() => {});
  };
  return (
    <Button {...props} onClick={onClick}>
      {running ? (
        <>
          <Loader2 className="size-4 animate-spin" /> {runningLabel}
        </>
      ) : Icon ? (
        <>
          <Icon className="size-4" /> {children}
        </>
      ) : (
        children
      )}
    </Button>
  );
}
