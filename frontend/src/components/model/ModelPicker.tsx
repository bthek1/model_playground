// The SELECT slot — shared by every model page, audio or not. Beyond the buttons
// it carries the **size-before-load guardrail**: the selected model's approximate
// download is always shown, and anything past `LARGE_MODEL_BYTES` gets an
// explicit warning — a browser tab's memory budget is far tighter than a
// workstation's, and the download is the user's bandwidth.
//
// Disable it while `loading` or `running`: switching models mid-flight discards
// work the user is waiting on (docs/standards/model-page-pattern.md §4).
//
// Two layouts, because the slot lives in a ~20rem setup rail now. `list` (the
// default) gives every model its own full-width row carrying the label, hint and
// size — a wrapped row of bare buttons is unreadable at rail width, and it hid
// the per-model size behind a `title` tooltip. `row` is the old chip row, for a
// route that has horizontal room to spare.

import { AlertTriangle } from "lucide-react";

import { sizeEstimate, type MeasuredBytes } from "@/audio/size";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

/** The shape every audio model catalogue entry shares. */
export interface PickableModel {
  id: string;
  label: string;
  hint: string;
  /** Parameter count in millions. */
  params: number;
  /** Measured download bytes, where the params estimate would mislead. */
  bytes?: MeasuredBytes;
}

export function ModelPicker<T extends PickableModel>({
  models,
  value,
  onChange,
  disabled = false,
  layout = "list",
}: {
  models: readonly T[];
  value: string;
  onChange: (model: T) => void;
  disabled?: boolean;
  layout?: "list" | "row";
}) {
  const selected = models.find((m) => m.id === value);
  const size = selected ? sizeEstimate(selected.params, selected.bytes) : null;

  return (
    <div className="space-y-2">
      <div
        className={cn(
          layout === "list" ? "flex flex-col gap-1.5" : "flex flex-wrap gap-2",
        )}
      >
        {models.map((m) => {
          const s = sizeEstimate(m.params, m.bytes);
          const isSelected = m.id === value;

          if (layout === "row") {
            return (
              <Button
                key={m.id}
                variant={isSelected ? "default" : "outline"}
                size="sm"
                disabled={disabled}
                onClick={() => onChange(m)}
                title={`${m.hint} — ${s.label}`}
              >
                {m.label}
              </Button>
            );
          }

          return (
            <Button
              key={m.id}
              variant={isSelected ? "default" : "outline"}
              disabled={disabled}
              onClick={() => onChange(m)}
              aria-pressed={isSelected}
              className="h-auto w-full flex-col items-start gap-0.5 px-3 py-2 text-left whitespace-normal"
            >
              <span className="w-full text-sm font-medium">{m.label}</span>
              <span
                className={cn(
                  "w-full text-xs leading-snug font-normal",
                  isSelected
                    ? "text-primary-foreground/75"
                    : "text-muted-foreground",
                )}
              >
                {m.hint}
              </span>
            </Button>
          );
        })}
      </div>

      {selected && size && (
        <div className="space-y-1 text-xs text-muted-foreground">
          <p data-testid="model-size-note">
            {selected.hint} <span className="mx-1 opacity-50">·</span>
            <span className="tabular-nums">{selected.params}M params</span>
            <span className="mx-1 opacity-50">·</span>
            <span className="tabular-nums">{size.label}</span>
          </p>
          {size.large && (
            <p
              data-testid="model-size-warning"
              className="flex items-start gap-1.5 text-amber-600 dark:text-amber-500"
            >
              <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
              <span>
                Large model — expect a slow first load and high memory use.
                Weights are cached after the first download.
              </span>
            </p>
          )}
        </div>
      )}
    </div>
  );
}
