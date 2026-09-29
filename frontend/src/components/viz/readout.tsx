// Label/value cells for a `<dl>` readout — the numbers a training page reports
// beside its chart. Two densities: `Fact` is a compact row for a panel of
// dataset or run facts, `Score` a headline figure for a scoreboard.
//
// They render `<dt>`/`<dd>` pairs, so the caller owns the `<dl>` and its grid.

export function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-2">
      <dt>{label}</dt>
      <dd className="font-mono text-foreground tabular-nums">{value}</dd>
    </div>
  );
}

export function Score({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="font-mono text-lg tabular-nums">{value}</dd>
    </div>
  );
}
