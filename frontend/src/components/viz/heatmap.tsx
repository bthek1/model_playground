// Shared diverging-heatmap primitives for the model-visualization standard
// (docs/standards/model-visualization.md §3, §4). A signed tensor is painted
// per-pixel to a <canvas>: red = positive, blue = negative, and **alpha encodes
// magnitude** (|value| / maxAbs) so near-zero values fade into the card and the
// same canvas reads correctly in both light and dark themes.
//
// These are the canonical primitives — the training weight templates and the
// tensor-arithmetic result view both compose them, so neither forks the drawing
// logic. See ModelWeights.tsx (per-class templates) and routes/tensor.tsx
// (op result) for callers.

import { useEffect, useMemo, useRef } from "react";

// Diverging endpoints (Tailwind red-500 / blue-500).
export const POS: readonly [number, number, number] = [239, 68, 68];
export const NEG: readonly [number, number, number] = [59, 130, 246];

/**
 * Largest absolute value in a tensor, memoised by identity. Used to normalise a
 * heatmap so tiles are directly comparable; never returns 0, so it's always a
 * safe divisor (`|| 1` handles the all-zero case).
 */
export function useMaxAbs(values: Float32Array): number {
  return useMemo(() => {
    let m = 0;
    for (let i = 0; i < values.length; i++) {
      const a = Math.abs(values[i]);
      if (a > m) m = a;
    }
    return m || 1;
  }, [values]);
}

/**
 * Paint a `width`×`height` diverging heatmap into a 2D context. `at(i)` returns
 * the signed value for pixel `i` in row-major order — pass a strided accessor to
 * draw a column/slice of a larger buffer without copying it.
 */
export function paintDiverging(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  maxAbs: number,
  at: (i: number) => number,
): void {
  const img = ctx.createImageData(width, height);
  for (let i = 0; i < width * height; i++) {
    const t = at(i) / maxAbs; // roughly [-1, 1]
    const [r, g, b] = t >= 0 ? POS : NEG;
    const o = i * 4;
    img.data[o] = r;
    img.data[o + 1] = g;
    img.data[o + 2] = b;
    img.data[o + 3] = Math.round(Math.min(1, Math.abs(t)) * 255);
  }
  ctx.putImageData(img, 0, 0);
}

/**
 * A diverging-heatmap tile drawn per-pixel from a row-major `Float32Array`.
 * `at` defaults to contiguous indexing; pass a custom accessor for strided data
 * (e.g. a single class column of a weight matrix). Size it with a Tailwind
 * `size-*` class or an inline `width`/`height`; `[image-rendering:pixelated]`
 * keeps individual values crisp when scaled up.
 */
export function HeatmapTile({
  values,
  rows,
  cols,
  maxAbs,
  at,
  className,
  ...props
}: {
  values: Float32Array;
  rows: number;
  cols: number;
  maxAbs: number;
  at?: (i: number) => number;
} & React.CanvasHTMLAttributes<HTMLCanvasElement>) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    paintDiverging(ctx, cols, rows, maxAbs, at ?? ((i) => values[i]));
  }, [values, rows, cols, maxAbs, at]);

  return (
    <canvas
      ref={canvasRef}
      width={cols}
      height={rows}
      className={`rounded border bg-muted/30 [image-rendering:pixelated] ${className ?? ""}`}
      {...props}
    />
  );
}

/**
 * The value→color key that must accompany every diverging heatmap: numeric
 * `±maxAbs` end-labels, a gradient swatch, and a plain-words gloss so meaning
 * never rides on color alone. `posLabel`/`negLabel` name the two directions
 * ("for"/"against" for a classifier, "positive"/"negative" for raw values);
 * `note` is optional right-aligned state (e.g. "after epoch 5").
 */
export function DivergingLegend({
  maxAbs,
  posLabel,
  negLabel,
  note,
}: {
  maxAbs: number;
  posLabel: string;
  negLabel: string;
  note?: string;
}) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
      <div className="flex items-center gap-2">
        <span className="font-mono tabular-nums">−{maxAbs.toFixed(2)}</span>
        <span
          className="h-2.5 w-28 rounded"
          style={{
            background: `linear-gradient(to right, rgb(${NEG.join()}), rgba(148,163,184,0.15), rgb(${POS.join()}))`,
          }}
        />
        <span className="font-mono tabular-nums">+{maxAbs.toFixed(2)}</span>
      </div>
      <span>
        <span className="text-[rgb(59,130,246)]">blue</span> = {negLabel} ·{" "}
        <span className="text-[rgb(239,68,68)]">red</span> = {posLabel}
      </span>
      {note && <span className="ml-auto font-mono tabular-nums">{note}</span>}
    </div>
  );
}
