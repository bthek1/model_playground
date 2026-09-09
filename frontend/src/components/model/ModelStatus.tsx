// The LOAD slot. Renders every state of Machine A — idle, loading, warm-up,
// ready, error — and owns the actions that move it: `load`, `cancel`, `retry`
// (docs/standards/model-page-pattern.md §2, §4).
//
// This is the only slot allowed a progress bar. Inference progress belongs in
// OUTPUT; putting both here is what made "is it still loading?" ambiguous.
//
// The bar is fed the *aggregate* (`loadProgress`), never a raw per-file event —
// see `model/progress.ts` for why. Three things a user asks while waiting, all
// answered on screen: how far along, how much data, and how long so far.

import {
  CheckCircle2,
  Cpu,
  Download,
  HardDriveDownload,
  Loader2,
  RotateCw,
  X,
} from "lucide-react";

import { formatBytes } from "@/model/size";
import { ErrorNote } from "@/components/model/ErrorNote";
import { Button } from "@/components/ui/button";
import { classifyLoadError } from "@/model/errors";
import type { LoadProgress } from "@/model/progress";
import type { ModelStatus as Status } from "@/model/types";

function seconds(ms: number): string {
  return ms < 1000 ? "" : ms < 60_000
    ? `${Math.round(ms / 1000)}s`
    : `${Math.floor(ms / 60_000)}m ${Math.round((ms % 60_000) / 1000)}s`;
}

/** What is happening, in the user's words. */
function phaseLabel(phase: LoadProgress["phase"] | undefined): string {
  if (phase === "warmup") return "Warming up (first inference)…";
  if (phase === "downloading") return "Downloading weights";
  return "Contacting the model host…";
}

export function ModelStatus({
  status,
  backend,
  loadProgress,
  error,
  onLoad,
  onRetry,
  onCancel,
  /** True when this model's weights are already in the browser cache. */
  cached = false,
  /** How long the completed load took, for the ready chip. */
  loadedInMs = null,
  /**
   * The page is re-loading a model the user had loaded before the refresh.
   * Only ever set on a cache hit, so it costs no bandwidth — but it is labelled
   * honestly as a re-load, not as a session that survived.
   */
  restoring = false,
  /** Disable the load button while an input is being prepared, etc. */
  disabled = false,
}: {
  status: Status;
  backend: string | null;
  loadProgress: LoadProgress | null;
  error?: string | null;
  onLoad?: () => void;
  onRetry?: (overrides?: Record<string, unknown>) => void;
  onCancel?: () => void;
  cached?: boolean;
  loadedInMs?: number | null;
  restoring?: boolean;
  disabled?: boolean;
}) {
  if (status === "idle") {
    return (
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2 xl:block xl:space-y-2">
        <Button
          size="sm"
          onClick={onLoad}
          disabled={disabled || !onLoad}
          className="xl:w-full"
        >
          {cached ? (
            <HardDriveDownload className="size-4" />
          ) : (
            <Download className="size-4" />
          )}
          {cached ? "Load model (cached)" : "Load model"}
        </Button>
        {/* The download estimate is quoted once, by ModelPicker directly above
            in the setup rail — the guardrail is satisfied and repeating the
            number here just made the rail say it twice. */}
        <p className="text-xs leading-snug text-muted-foreground">
          {cached
            ? "Already downloaded — loads from the browser cache."
            : "Downloaded once, then cached by the browser."}
        </p>
      </div>
    );
  }

  if (status === "error") {
    const info = classifyLoadError(error ?? "The model failed to load.");
    return (
      <ErrorNote
        message={info.message}
        hint={info.hint}
        detail={info.raw === info.message ? undefined : info.raw}
        action={
          onRetry ? (
            <div className="flex flex-wrap gap-2">
              <Button size="sm" variant="outline" onClick={() => onRetry()}>
                <RotateCw className="size-4" /> Retry
              </Button>
              {info.suggestsCpu && backend !== "wasm" && (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => onRetry({ backend: "wasm" })}
                >
                  <Cpu className="size-4" /> Retry on CPU
                </Button>
              )}
            </div>
          ) : undefined
        }
      />
    );
  }

  if (status === "ready") {
    return (
      <p
        data-testid="model-ready"
        className="inline-flex flex-wrap items-center gap-x-1.5 rounded-md border border-emerald-600/30 bg-emerald-600/10 px-2.5 py-1 text-xs text-muted-foreground"
      >
        <CheckCircle2 className="size-3.5 shrink-0 text-emerald-600 dark:text-emerald-500" />
        Model ready · running on{" "}
        <span className="font-medium uppercase">{backend}</span>
        {loadedInMs != null && loadedInMs >= 1000 && (
          <span className="tabular-nums opacity-80">
            · loaded in {seconds(loadedInMs)}
          </span>
        )}
      </p>
    );
  }

  // --- loading ---------------------------------------------------------------
  const p = loadProgress;
  const pct = p?.percent ?? null;
  const elapsed = p ? seconds(p.elapsedMs) : "";

  return (
    <div
      className="space-y-1.5"
      data-testid="load-progress"
      data-phase={p?.phase ?? "connecting"}
    >
      <p className="flex items-start gap-1.5 text-xs leading-snug break-all text-muted-foreground">
        <Loader2 className="mt-0.5 size-3.5 shrink-0 animate-spin" />
        <span>
          {restoring && p?.phase !== "warmup"
            ? "Restoring from cache…"
            : phaseLabel(p?.phase)}
          {p?.current ? ` · ${p.current}` : ""}
        </span>
      </p>

      <div
        role="progressbar"
        aria-label="Model load progress"
        aria-valuemin={0}
        aria-valuemax={100}
        // Omitted while indeterminate — an assistive technology should say
        // "busy", not invent a number we don't have.
        {...(pct != null ? { "aria-valuenow": pct } : {})}
        className="h-1.5 w-full overflow-hidden rounded-full bg-muted"
      >
        <div
          className={
            pct != null
              ? "h-full rounded-full bg-primary transition-[width]"
              : "h-full w-1/3 animate-pulse rounded-full bg-primary/60"
          }
          style={pct != null ? { width: `${pct}%` } : undefined}
        />
      </div>

      <p className="flex flex-wrap items-center gap-x-2 text-[0.7rem] text-muted-foreground tabular-nums">
        {pct != null && <span>{pct}%</span>}
        {p && p.total > 0 && (
          <span>
            {formatBytes(p.loaded)} / {formatBytes(p.total)}
          </span>
        )}
        {p && p.files.count > 0 && (
          <span>
            {p.files.done} of {p.files.count} files
          </span>
        )}
        {elapsed && <span>{elapsed}</span>}
        {onCancel && (
          <Button
            size="sm"
            variant="ghost"
            data-testid="load-cancel"
            className="ml-auto h-6 px-2 text-[0.7rem]"
            onClick={onCancel}
          >
            <X className="size-3" /> Cancel
          </Button>
        )}
      </p>
    </div>
  );
}
