// The RUN slot. The task supplies its own input surface (a textarea, a file
// drop, a mic button, a pair of matrices); this wraps it with the two things
// every task shares: a transport row whose controls are gated on `ready`, and a
// place for input-side errors (a failed decode, a denied mic) that is *not* the
// output panel (docs/standards/model-page-pattern.md §4).

import { useId, type ReactNode } from "react";

import { ErrorNote } from "@/components/model/ErrorNote";

export function InputPanel({
  children,
  controls,
  error,
  /** Shown in place of the controls before a model exists. */
  disabledHint = "Load a model to run it.",
  ready,
}: {
  /** The task's input fields. */
  children?: ReactNode;
  /** Transport buttons — Speak, Transcribe, Classify. Gate them on `ready`. */
  controls: ReactNode;
  /** Input-side failure: decode error, mic permission, bad matrix shape. */
  error?: string | null;
  disabledHint?: string;
  /** Drives the hint under the controls. */
  ready: boolean;
}) {
  const hintId = useId();
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4">
      {children}
      {/* Sticky, not bottom-pinned: the transport sits directly under the input
          it drives, and only detaches to the bottom edge once a tall input
          surface scrolls it out of reach. Pinning it with `mt-auto` instead
          left a chasm on every task whose input is short. */}
      <div className="space-y-2 md:sticky md:bottom-0 md:bg-background md:pb-1 md:pt-2">
        {/* The hint is *described by* the controls, not merely printed under
            them: a greyed-out button with no reason attached is a dead end for
            anyone driving the page by keyboard and screen reader. */}
        <div
          className="flex flex-wrap items-center gap-2"
          aria-describedby={ready ? undefined : hintId}
        >
          {controls}
        </div>
        {!ready && (
          <p id={hintId} className="text-xs text-muted-foreground">
            {disabledHint}
          </p>
        )}
        <ErrorNote message={error} />
      </div>
    </div>
  );
}
