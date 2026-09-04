// The LOAD slot. Renders every state of Machine A — idle, loading, warm-up,
// ready, error — and owns the two actions that move it: `load` and `retry`
// (docs/standards/model-page-pattern.md §2, §4).
//
// This is the only slot allowed a progress bar. Inference progress belongs in
// OUTPUT; putting both here is what made "is it still loading?" ambiguous.

import { CheckCircle2, Download, Loader2, RotateCw } from "lucide-react";

import type { SizeEstimate } from "@/audio/size";
import { ErrorNote } from "@/components/model/ErrorNote";
import { Button } from "@/components/ui/button";
import type { ModelStatus as Status } from "@/model/types";

export function ModelStatus({
  status,
  backend,
  progress,
  error,
  size,
  onLoad,
  onRetry,
  /** Disable the load button while an input is being prepared, etc. */
  disabled = false,
}: {
  status: Status;
  backend: string | null;
  progress: { status: string; file?: string; progress?: number } | null;
  error?: string | null;
  /** Download estimate for the selected model — shown before the user commits. */
  size?: SizeEstimate | null;
  onLoad?: () => void;
  onRetry?: () => void;
  disabled?: boolean;
}) {
  if (status === "idle") {
    return (
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <Button size="sm" onClick={onLoad} disabled={disabled || !onLoad}>
          <Download className="size-4" /> Load model
        </Button>
        <p className="text-xs text-muted-foreground">
          {size ? (
            <>
              <span className="tabular-nums">{size.label}</span>
              <span className="mx-1 opacity-50">·</span>
            </>
          ) : null}
          Downloaded once, then cached by the browser.
        </p>
      </div>
    );
  }

  if (status === "error") {
    return (
      <ErrorNote
        message={error ?? "The model failed to load."}
        action={
          onRetry ? (
            <Button size="sm" variant="outline" onClick={onRetry}>
              <RotateCw className="size-4" /> Retry
            </Button>
          ) : undefined
        }
      />
    );
  }

  if (status === "ready") {
    return (
      <p
        data-testid="model-ready"
        className="flex items-center gap-1.5 text-xs text-muted-foreground"
      >
        <CheckCircle2 className="size-3.5 text-emerald-600 dark:text-emerald-500" />
        Model ready · running on{" "}
        <span className="font-medium uppercase">{backend}</span>
      </p>
    );
  }

  // --- loading ---------------------------------------------------------------
  // The engines post a synthetic `warmup` progress after the download, while the
  // first (throwaway) inference compiles shaders / JITs the WASM module.
  if (progress?.status === "warmup") {
    return (
      <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <Loader2 className="size-3.5 animate-spin" />
        Warming up the model…
      </p>
    );
  }

  const pct =
    progress && typeof progress.progress === "number"
      ? Math.round(progress.progress)
      : null;

  return (
    <div className="space-y-1.5">
      <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <Loader2 className="size-3.5 animate-spin" />
        Loading model{progress?.file ? ` · ${progress.file}` : ""}
        {pct != null ? ` · ${pct}%` : ""}
      </p>
      <div className="h-1.5 w-full max-w-sm overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full bg-primary transition-[width]"
          style={{ width: `${pct ?? 8}%` }}
        />
      </div>
    </div>
  );
}
