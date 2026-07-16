// Shared model load/ready status for in-browser audio pipelines (ASR,
// classification, …). Shows a labelled progress bar while weights download and
// the active backend once the model is ready.

import { Loader2 } from "lucide-react";

import type { PipelineProgress } from "@/audio/pipelineTypes";

export function ModelStatus({
  loading,
  ready,
  backend,
  progress,
}: {
  loading: boolean;
  ready: boolean;
  backend: string | null;
  progress: PipelineProgress | null;
}) {
  if (ready) {
    return (
      <p className="text-xs text-muted-foreground">
        Model ready · running on{" "}
        <span className="font-medium uppercase">{backend}</span>
      </p>
    );
  }
  if (!loading) return null;

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
