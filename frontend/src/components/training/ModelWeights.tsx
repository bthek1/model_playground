// Visualises a trained linear (784 → 10) classifier as its 10 weight
// "templates". Each output class has 784 weights that reshape to a 28×28 grid —
// literally a picture of what that digit detector responds to. Colour encodes
// the signed weight value: red = positive (evidence *for* the class at that
// pixel), blue = negative (evidence *against*), transparent ≈ zero.
//
// The per-pixel drawing and legend are the shared diverging-heatmap primitives
// (components/viz/heatmap.tsx); this component only supplies the strided
// accessor that maps a class column out of the row-major weight matrix.

import { useMemo } from "react";

import {
  DivergingLegend,
  HeatmapTile,
  useMaxAbs,
} from "@/components/viz/heatmap";

const SIDE = 28; // 28×28 input image

/**
 * @param weights row-major `784 × numClasses` (weight for feature d, class c is
 *   `weights[d * numClasses + c]`).
 * @param bias one value per class.
 */
export function ModelWeights({
  weights,
  bias,
  numClasses = 10,
  epoch,
  tileSize = "size-16",
}: {
  weights: Float32Array;
  bias?: Float32Array;
  numClasses?: number;
  epoch?: number;
  /** Tailwind size class for each template tile (e.g. "size-16", "size-32"). */
  tileSize?: string;
}) {
  // Normalise colour intensity by the largest-magnitude weight so classes are
  // directly comparable.
  const maxAbs = useMaxAbs(weights);

  return (
    <div className="space-y-3">
      <div className="flex gap-2 overflow-x-auto pb-1">
        {Array.from({ length: numClasses }, (_, c) => (
          <div key={c} className="flex flex-none flex-col items-center gap-1">
            <ClassTemplate
              weights={weights}
              cls={c}
              numClasses={numClasses}
              maxAbs={maxAbs}
              tileSize={tileSize}
            />
            <span className="font-mono text-xs tabular-nums text-muted-foreground">
              {c}
            </span>
            {bias && (
              <span className="font-mono text-[10px] tabular-nums text-muted-foreground/70">
                b {bias[c] >= 0 ? "+" : ""}
                {bias[c].toFixed(2)}
              </span>
            )}
          </div>
        ))}
      </div>

      <DivergingLegend
        maxAbs={maxAbs}
        posLabel="for"
        negLabel="against"
        note={
          epoch !== undefined && epoch >= 0
            ? `after epoch ${epoch + 1}`
            : undefined
        }
      />
    </div>
  );
}

function ClassTemplate({
  weights,
  cls,
  numClasses,
  maxAbs,
  tileSize,
}: {
  weights: Float32Array;
  cls: number;
  numClasses: number;
  maxAbs: number;
  tileSize: string;
}) {
  // Weights are stored row-major (feature-major), so this class's 784 values are
  // strided across the buffer — read them directly instead of copying a slice.
  const at = useMemo(
    () => (d: number) => weights[d * numClasses + cls],
    [weights, numClasses, cls],
  );

  return (
    <HeatmapTile
      values={weights}
      rows={SIDE}
      cols={SIDE}
      maxAbs={maxAbs}
      at={at}
      className={tileSize}
      aria-label={`Weight template for digit ${cls}`}
    />
  );
}
