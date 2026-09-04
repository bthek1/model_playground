// The SELECT slot — shared by every model page, audio or not. Beyond the buttons
// it carries the **size-before-load guardrail**: the selected model's approximate
// download is always shown, and anything past `LARGE_MODEL_BYTES` gets an
// explicit warning — a browser tab's memory budget is far tighter than a
// workstation's, and the download is the user's bandwidth.
//
// Disable it while `loading` or `running`: switching models mid-flight discards
// work the user is waiting on (docs/standards/model-page-pattern.md §4).

import { AlertTriangle } from "lucide-react";

import { sizeEstimate, type MeasuredBytes } from "@/audio/size";
import { Button } from "@/components/ui/button";

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
}: {
  models: readonly T[];
  value: string;
  onChange: (model: T) => void;
  disabled?: boolean;
}) {
  const selected = models.find((m) => m.id === value);
  const size = selected ? sizeEstimate(selected.params, selected.bytes) : null;

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-2">
        {models.map((m) => {
          const s = sizeEstimate(m.params, m.bytes);
          return (
            <Button
              key={m.id}
              variant={m.id === value ? "default" : "outline"}
              size="sm"
              disabled={disabled}
              onClick={() => onChange(m)}
              title={`${m.hint} — ${s.label}`}
            >
              {m.label}
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
              className="flex items-center gap-1.5 text-amber-600 dark:text-amber-500"
            >
              <AlertTriangle className="size-3.5 shrink-0" />
              Large model — expect a slow first load and high memory use. Weights
              are cached after the first download.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
