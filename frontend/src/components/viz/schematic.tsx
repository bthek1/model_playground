// Shared stage/arrow/chip primitives for the model-visualization standard
// (docs/standards/model-visualization.md §3). Every model view builds its
// structure as a left-to-right dataflow of labelled Stages joined by Arrows,
// with ParamChips carrying scalar facts (shapes, parameter counts). The Arrow
// glyph rotates to point down when stages stack vertically, so a schematic
// reflows on narrow screens.
//
// These are the canonical primitives — the training architecture schematic and
// the tensor-arithmetic dataflow both compose them rather than forking. See
// ModelArchitecture.tsx and routes/tensor.tsx.

/** A labelled box for one phase of the pipeline (INPUT, WEIGHTS, OUTPUT). */
export function Stage({
  title,
  sub,
  children,
  className,
}: {
  title: string;
  sub?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`rounded-lg border bg-card/60 p-2.5 ${className ?? ""}`}>
      <p className="text-[11px] font-semibold tracking-wide">{title}</p>
      {sub && <p className="mb-2 text-[10px] text-muted-foreground">{sub}</p>}
      {children}
    </div>
  );
}

/** The connector between two stages, carrying the operation label (`W·x + b`). */
export function Arrow({ label }: { label: string }) {
  return (
    <div className="flex shrink-0 flex-col items-center justify-center gap-1 text-muted-foreground">
      <span className="font-mono text-xs">{label}</span>
      {/* Big long-arrow connector; rotates to point down when stages stack. */}
      <span aria-hidden className="rotate-90 text-5xl leading-none xl:rotate-0">
        ⟶
      </span>
    </div>
  );
}

/** A compact `label value` pill for a single scalar fact. */
export function ParamChip({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: boolean;
}) {
  return (
    <span
      className={`rounded px-1.5 py-0.5 font-mono text-[11px] ${
        accent ? "bg-primary/15 text-primary" : "bg-muted text-muted-foreground"
      }`}
    >
      {label} <span className="font-semibold">{value}</span>
    </span>
  );
}
