// An embedding, drawn.
//
// A page whose output is "768 numbers were produced" has not produced an output,
// which is the whole reason this exists — and it is the visualization standard's
// grammar applied unchanged (docs/standards/model-visualization.md §3/§4): one
// row of a diverging canvas, red positive, blue negative, alpha for magnitude,
// with the legend that must accompany it.
//
// The canvas is `dim` pixels wide and one tall, stretched by CSS with
// `image-rendering: pixelated`, so a 384-d and a 768-d vector occupy the same
// width and the eye compares *structure* rather than length. The numbers that
// matter — the width, and the norm — are printed rather than implied, because
// "the vectors are normalised" is a claim the page can simply show.

import { DivergingLegend, HeatmapTile, useMaxAbs } from "@/components/viz/heatmap";
import { l2norm } from "@/model/similarity";

export function VectorStrip({
  vector,
  /** Shown above the strip — usually the text that produced it. */
  label,
  /** Right-hand note, e.g. the truncation state. */
  note,
  "data-testid": testId = "vector-strip",
}: {
  vector: Float32Array;
  label?: React.ReactNode;
  note?: React.ReactNode;
  "data-testid"?: string;
}) {
  const maxAbs = useMaxAbs(vector);
  const norm = l2norm(vector);

  return (
    <div className="space-y-2" data-testid={testId}>
      {(label || note) && (
        <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
          {label && <div className="min-w-0 text-xs">{label}</div>}
          {note && (
            <span className="font-mono text-xs text-muted-foreground tabular-nums">
              {note}
            </span>
          )}
        </div>
      )}
      <HeatmapTile
        values={vector}
        rows={1}
        cols={vector.length}
        maxAbs={maxAbs}
        className="h-7 w-full"
        data-testid={`${testId}-canvas`}
      />
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
        <span className="font-mono tabular-nums" data-testid={`${testId}-dim`}>
          {vector.length}-d
        </span>
        {/* The norm, to three places. 1.000 is the demonstration that
            normalisation happened; anything else is the page telling on itself. */}
        <span className="font-mono tabular-nums" data-testid={`${testId}-norm`}>
          ‖v‖ = {norm.toFixed(3)}
        </span>
      </div>
      <DivergingLegend
        maxAbs={maxAbs}
        posLabel="positive"
        negLabel="negative"
        note={`max |v| = ${maxAbs.toFixed(3)}`}
      />
    </div>
  );
}
