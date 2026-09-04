// The page shell every task route fills in. Four slots, always in this order,
// always all four present: Select → Load → Run → Output
// (docs/standards/model-page-pattern.md §4).
//
// The slots are named props rather than children on purpose. A route cannot
// reorder them, cannot drop the OUTPUT band when it has nothing to show, and
// cannot quietly grow a fifth stage — which is exactly how the five audio routes
// drifted apart before this existed.
//
// The arrangement is two-dimensional, not a single column: SELECT and LOAD are
// done once per session and collapse into a compact setup rail, while RUN and
// OUTPUT — the pair the user touches on every iteration — sit side by side so a
// result never lands below the fold. Placement is by grid *area*, so the DOM
// order stays 1 → 2 → 3 → 4 at every breakpoint and keyboard/screen-reader
// traversal still matches the numbered pipeline.

import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

/** One numbered band. Exported for the rare route that composes its own shell. */
export function ModelSlot({
  step,
  label,
  hint,
  children,
  className,
  /** Tighter spacing for the setup rail, where the content is a list and a line. */
  dense = false,
}: {
  step: number;
  label: string;
  hint?: string;
  children: ReactNode;
  className?: string;
  dense?: boolean;
}) {
  const id = `slot-${label.toLowerCase().replace(/\s+/g, "-")}`;
  return (
    <section
      aria-labelledby={id}
      data-testid={`slot-${step}`}
      className={cn(dense ? "space-y-2" : "space-y-3", className)}
    >
      <div className="flex items-baseline gap-2">
        <span
          aria-hidden
          className="flex size-5 shrink-0 items-center justify-center rounded-full bg-muted text-[0.65rem] font-semibold text-muted-foreground tabular-nums"
        >
          {step}
        </span>
        <h2
          id={id}
          className="text-xs font-semibold tracking-wide text-muted-foreground uppercase"
        >
          {label}
        </h2>
        {hint && (
          <span className="text-xs text-muted-foreground/80">{hint}</span>
        )}
      </div>
      {children}
    </section>
  );
}

export interface ModelPageProps {
  icon: LucideIcon;
  title: string;
  /** One or two sentences: what this task does and where it runs. */
  description: ReactNode;
  /** 1 — the model catalogue. Usually `<ModelPicker>`. */
  select: ReactNode;
  /** 2 — weight download / warm-up. Usually `<ModelStatus>`. */
  load: ReactNode;
  /** 3 — the task's input surface and transport controls. */
  run: ReactNode;
  /** 4 — always rendered, even with no result yet. Usually `<OutputPanel>`. */
  output: ReactNode;
  /** Override a band's label where the task has better words for it. */
  labels?: Partial<Record<"select" | "load" | "run" | "output", string>>;
  /**
   * Right edge of the header — a backend chip, a device line. For the one-line
   * answers that don't deserve a band of their own.
   */
  aside?: ReactNode;
}

// Grid areas per breakpoint. Below `md` there are no areas at all, so the four
// bands fall back to plain source order in one column — the phone layout, and
// the layout every jsdom test sees.
const GRID = [
  "grid min-h-0 grid-cols-1 gap-6 md:flex-1",
  "md:grid-cols-2 md:grid-rows-[auto_minmax(0,1fr)]",
  "md:[grid-template-areas:'setup_setup'_'work-a_work-b']",
  "xl:grid-cols-[minmax(17rem,20rem)_minmax(0,1fr)_minmax(0,1fr)]",
  "xl:grid-rows-[minmax(0,1fr)]",
  "xl:[grid-template-areas:'setup_work-a_work-b']",
].join(" ");

/** A workbench column: fills its cell, scrolls its own overflow. */
const COLUMN = "flex min-h-0 min-w-0 flex-col md:overflow-y-auto";

// The band inside a workbench column stretches, and hands that stretch on to
// whatever the route put in it — so a textarea or an OutputPanel can fill the
// column height instead of hugging its content at the top of a tall cell.
const FILL =
  "flex min-h-0 flex-1 flex-col [&>*:last-child]:min-h-0 [&>*:last-child]:flex-1";

export function ModelPage({
  icon: Icon,
  title,
  description,
  select,
  load,
  run,
  output,
  labels,
  aside,
}: ModelPageProps) {
  return (
    <div className="mx-auto flex w-full max-w-[1600px] flex-col gap-6 md:h-full md:min-h-0">
      <header className="flex flex-wrap items-start justify-between gap-x-6 gap-y-2">
        <div className="min-w-0">
          <h1 className="mb-1 flex items-center gap-2 text-2xl font-semibold">
            <Icon className="size-6" /> {title}
          </h1>
          <p className="text-sm text-muted-foreground">{description}</p>
        </div>
        {aside && <div className="shrink-0">{aside}</div>}
      </header>

      <div className={GRID}>
        {/* SETUP — slots 1 and 2, one surface. A rail at xl, a strip at md. */}
        <div
          className={cn(
            "flex min-w-0 flex-col gap-5 rounded-lg border bg-muted/30 p-4",
            "md:[grid-area:setup] md:flex-row md:gap-8",
            "xl:max-h-full xl:flex-col xl:gap-5 xl:self-start xl:overflow-y-auto",
          )}
        >
          <ModelSlot
            step={1}
            label={labels?.select ?? "Model"}
            dense
            className="min-w-0 md:flex-1 xl:flex-none"
          >
            {select}
          </ModelSlot>

          <ModelSlot
            step={2}
            label={labels?.load ?? "Load"}
            dense
            className="min-w-0 md:w-80 md:shrink-0 xl:w-auto"
          >
            {load}
          </ModelSlot>
        </div>

        {/* WORKBENCH — the iteration loop, input beside output. */}
        <div className={cn(COLUMN, "md:[grid-area:work-a]")}>
          <ModelSlot step={3} label={labels?.run ?? "Input"} className={FILL}>
            {run}
          </ModelSlot>
        </div>

        <div className={cn(COLUMN, "md:[grid-area:work-b]")}>
          <ModelSlot
            step={4}
            label={labels?.output ?? "Output"}
            className={FILL}
          >
            {output}
          </ModelSlot>
        </div>
      </div>
    </div>
  );
}
