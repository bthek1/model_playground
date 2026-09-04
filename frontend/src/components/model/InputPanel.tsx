// The RUN slot. The task supplies its own input surface (a textarea, a file
// drop, a mic button, a pair of matrices); this wraps it with the two things
// every task shares: a transport row whose controls are gated on `ready`, and a
// place for input-side errors (a failed decode, a denied mic) that is *not* the
// output panel (docs/standards/model-page-pattern.md §4).

import type { ReactNode } from "react";

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
  return (
    <div className="space-y-4">
      {children}
      <div className="flex flex-wrap items-center gap-2">{controls}</div>
      {!ready && (
        <p className="text-xs text-muted-foreground">{disabledHint}</p>
      )}
      <ErrorNote message={error} />
    </div>
  );
}
