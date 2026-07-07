// Small presentational controls shared across the training page and its settings
// overlays. Extracted from the old single-file training route so the same
// NumberField / Stat render inside the popup dialogs and the floating HUD.

import { Label } from "@/components/ui/label";

/** Format a fraction in [0,1] as a percentage string. */
export function pct(x: number): string {
  return `${(x * 100).toFixed(2)}%`;
}

export function NumberField({
  label,
  value,
  onChange,
  step,
  min,
  disabled,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  step: number;
  min: number;
  disabled?: boolean;
}) {
  return (
    <div className="grid gap-1.5">
      <Label>{label}</Label>
      <input
        type="number"
        value={value}
        step={step}
        min={min}
        disabled={disabled}
        onChange={(e) => {
          const v = Number(e.target.value);
          if (Number.isFinite(v)) onChange(v);
        }}
        className="h-8 w-full rounded-lg border border-input bg-transparent px-2.5 py-1 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 disabled:opacity-50 dark:bg-input/30"
      />
    </div>
  );
}

export function Stat({
  label,
  value,
  sub,
}: {
  label: string;
  value: string;
  sub?: string;
}) {
  return (
    <div className="rounded-lg border bg-card/70 p-3 backdrop-blur-sm">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="font-mono text-lg tabular-nums">{value}</p>
      {sub && <p className="text-xs text-muted-foreground">{sub}</p>}
    </div>
  );
}
