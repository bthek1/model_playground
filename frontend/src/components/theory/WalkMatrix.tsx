// Mᵏ as a grid, with the source's row outlined — M being A or A + I.
//
// A walk count is a single non-negative magnitude, so the ramp is sequential
// (model-visualization §4: `--muted` → `--primary`), log-scaled, and a zero cell
// is drawn empty and nothing else: "no walk of length k" is the fact the page
// is about, and it must not be confused with "few". The row's support is also
// written out as text below, because a 34×34 grid's empty cells are not
// something anyone should have to count.

import { cn } from "@/lib/utils";

import { countOpacity } from "./graphGeometry";

const SIZE = 300;

export function WalkMatrix({
  matrix,
  n,
  source,
  k,
  selfLoops,
  className,
}: {
  className?: string;
  matrix: Float64Array;
  n: number;
  source: number;
  k: number;
  selfLoops: boolean;
}) {
  const cell = SIZE / n;
  let max = 0;
  for (const v of matrix) if (v > max) max = v;
  const name = selfLoops ? `(A + I)^${k}` : `A^${k}`;

  return (
    <svg
      viewBox={`-1 -1 ${SIZE + 2} ${SIZE + 2}`}
      className={cn("h-auto", className ?? "w-full max-w-[20rem]")}
      role="img"
      aria-label={`${name}: a ${n} by ${n} matrix of walk counts, largest ${max}. Row ${source} is outlined.`}
      data-testid="walk-matrix"
    >
      <rect x={0} y={0} width={SIZE} height={SIZE} fill="var(--muted)" fillOpacity={0.35} />
      {Array.from({ length: n * n }, (_, i) => {
        const v = matrix[i];
        if (v === 0) return null;
        return (
          <rect
            key={i}
            x={(i % n) * cell}
            y={Math.floor(i / n) * cell}
            width={cell}
            height={cell}
            fill="var(--primary)"
            fillOpacity={countOpacity(v, max)}
          />
        );
      })}
      <rect
        x={0}
        y={source * cell}
        width={SIZE}
        height={cell}
        fill="none"
        stroke="var(--foreground)"
        strokeWidth={1.5}
      />
    </svg>
  );
}
