// The SELECT slot of the three graph pages: which message-passing architecture
// to train. The same four, from the same `GNN_ARCHITECTURES` table, on every
// page. `/graph` also prints each one's note, because it is the page that
// introduces them; the two later pages assume it has been read.

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { GNN_ARCHITECTURES, type GnnArch } from "@/webgpu/gnn";

const ARCHES: readonly GnnArch[] = ["gcn", "sage", "gin", "gat"];

export function ArchPicker({
  value,
  onChange,
  disabled,
  showNotes = false,
}: {
  value: GnnArch;
  onChange: (arch: GnnArch) => void;
  disabled: boolean;
  /** Print each architecture's one-line note under its name. */
  showNotes?: boolean;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      {ARCHES.map((arch) => {
        const meta = GNN_ARCHITECTURES[arch];
        const selected = arch === value;
        return (
          <Button
            key={arch}
            variant={selected ? "default" : "outline"}
            disabled={disabled}
            aria-pressed={selected}
            onClick={() => onChange(arch)}
            className="h-auto w-full flex-col items-start gap-0.5 px-3 py-2 text-left whitespace-normal"
          >
            <span className="flex w-full items-baseline gap-2 text-sm font-medium">
              {meta.label}
              <span
                className={cn(
                  "font-mono text-[0.65rem] font-normal",
                  selected ? "text-primary-foreground/70" : "text-muted-foreground",
                )}
              >
                {meta.aggregation}
              </span>
            </span>
            {showNotes && (
              <span
                className={cn(
                  "w-full text-xs leading-snug font-normal",
                  selected ? "text-primary-foreground/75" : "text-muted-foreground",
                )}
              >
                {meta.note}
              </span>
            )}
          </Button>
        );
      })}
    </div>
  );
}
