// The page shell every task route fills in. Four slots, always in this order,
// always all four present: Select → Load → Run → Output
// (docs/standards/model-page-pattern.md §4).
//
// The slots are named props rather than children on purpose. A route cannot
// reorder them, cannot drop the OUTPUT band when it has nothing to show, and
// cannot quietly grow a fifth stage — which is exactly how the five audio routes
// drifted apart before this existed.

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
}: {
  step: number;
  label: string;
  hint?: string;
  children: ReactNode;
  className?: string;
}) {
  const id = `slot-${label.toLowerCase().replace(/\s+/g, "-")}`;
  return (
    <section
      aria-labelledby={id}
      data-testid={`slot-${step}`}
      className={cn("space-y-3", className)}
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
}

export function ModelPage({
  icon: Icon,
  title,
  description,
  select,
  load,
  run,
  output,
  labels,
}: ModelPageProps) {
  return (
    <div className="mx-auto max-w-4xl space-y-8 p-8">
      <header>
        <h1 className="mb-1 flex items-center gap-2 text-2xl font-semibold">
          <Icon className="size-6" /> {title}
        </h1>
        <p className="text-sm text-muted-foreground">{description}</p>
      </header>

      <ModelSlot step={1} label={labels?.select ?? "Model"}>
        {select}
      </ModelSlot>

      <ModelSlot step={2} label={labels?.load ?? "Load"}>
        {load}
      </ModelSlot>

      <ModelSlot step={3} label={labels?.run ?? "Input"}>
        {run}
      </ModelSlot>

      <ModelSlot step={4} label={labels?.output ?? "Output"}>
        {output}
      </ModelSlot>
    </div>
  );
}
