// A grid of proteins, each drawn small and ringed by whether the model got it
// right.
//
// Graph classification's output is one label per graph, which the roadmap fairly
// called "a much smaller visual payoff" than a colour per node. The payoff it
// does have is *comparison*: forty little molecules side by side, the wrong ones
// outlined, is a picture of what the model finds hard — and clicking one shows
// the graph the decision was actually made on.
//
// Each tile is a canvas rather than SVG for the same reason GraphCanvas is, and
// the drawing is deliberately plain: at 96 px a protein is a shape, not a
// diagram. Nodes are dots, edges are hairlines, and the only colour that carries
// meaning is the border.

import { useEffect, useRef } from "react";

import type { GraphLayoutPayload } from "@/webgpu/proteinSession";
import { cn } from "@/lib/utils";

export interface ProteinTileProps {
  layout: GraphLayoutPayload | undefined;
  /** Whether the model's answer for this graph was right. */
  correct: boolean;
  selected: boolean;
  /** CSS pixels; the canvas is drawn at devicePixelRatio above this. */
  size: number;
}

/** Padding inside a tile, in CSS pixels, so a node on the edge is not clipped. */
export const TILE_PAD = 6;

/**
 * Map a graph's unit-square layout onto a square tile.
 *
 * Pure and exported for the reason `layoutToPixels` is: happy-dom gives a canvas
 * no 2D context, so anything left inside the paint effect is untested. The
 * degenerate cases are the interesting ones — a two-node protein whose layout
 * collapsed to a point, and the single-node graphs PROTEINS really contains.
 */
export function tilePixels(
  x: Float32Array,
  y: Float32Array,
  size: number,
): { px: Float32Array; py: Float32Array } {
  const n = x.length;
  const px = new Float32Array(n);
  const py = new Float32Array(n);
  const span = Math.max(0, size - 2 * TILE_PAD);

  // The layout is normalised into the unit square over the *whole* graph, but a
  // small protein can still occupy a sliver of it. Re-fitting to the tile's own
  // extent is what stops a chain of four nodes drawing as a dot in one corner.
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < n; i++) {
    if (x[i] < minX) minX = x[i];
    if (x[i] > maxX) maxX = x[i];
    if (y[i] < minY) minY = y[i];
    if (y[i] > maxY) maxY = y[i];
  }
  const width = maxX - minX;
  const height = maxY - minY;
  // A collapsed extent would divide by zero; centre the graph instead.
  const scale = Math.max(width, height) > 1e-6 ? span / Math.max(width, height) : 0;
  const offsetX = TILE_PAD + (span - width * scale) / 2;
  const offsetY = TILE_PAD + (span - height * scale) / 2;

  for (let i = 0; i < n; i++) {
    px[i] = n === 0 ? 0 : offsetX + (x[i] - minX) * scale;
    py[i] = n === 0 ? 0 : offsetY + (y[i] - minY) * scale;
  }
  return { px, py };
}

export function ProteinTile({
  layout,
  correct,
  selected,
  size,
}: ProteinTileProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !layout) return;
    const dpr =
      typeof window !== "undefined" ? Math.min(window.devicePixelRatio || 1, 2) : 1;
    canvas.width = Math.round(size * dpr);
    canvas.height = Math.round(size * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, size, size);

    const { px, py } = tilePixels(layout.x, layout.y, size);

    ctx.strokeStyle = "rgba(128,128,128,0.45)";
    ctx.lineWidth = 0.6;
    ctx.beginPath();
    for (let i = 0; i < layout.nNodes; i++) {
      for (let e = layout.rowPtr[i]; e < layout.rowPtr[i + 1]; e++) {
        const j = layout.colIdx[e];
        if (j <= i) continue; // each undirected bond is stored twice
        ctx.moveTo(px[i], py[i]);
        ctx.lineTo(px[j], py[j]);
      }
    }
    ctx.stroke();

    // One neutral colour for every node: the classes here belong to the *graph*,
    // not to its nodes, and colouring the dots would imply a per-node answer the
    // model never gave.
    ctx.fillStyle = correct ? "rgba(16,185,129,0.95)" : "rgba(244,63,94,0.95)";
    const radius = Math.max(1, size / 60);
    for (let i = 0; i < layout.nNodes; i++) {
      ctx.beginPath();
      ctx.arc(px[i], py[i], radius, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [layout, correct, size]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: size, height: size, display: "block" }}
      className={cn(
        "rounded-md border-2 bg-background",
        correct ? "border-emerald-500/60" : "border-rose-500/70",
        selected && "ring-2 ring-primary",
      )}
      aria-hidden
    />
  );
}

export interface ProteinGalleryProps {
  /** The graphs to draw, in the order they should appear. */
  graphs: number[];
  labels: Uint8Array;
  predicted: Uint8Array;
  layouts: Map<number, GraphLayoutPayload>;
  /** Asked for each tile that has no layout yet. */
  onNeedLayout: (graph: number) => void;
  selected: number | null;
  onSelect: (graph: number) => void;
  classNames: readonly string[];
  size?: number;
}

export function ProteinGallery({
  graphs,
  labels,
  predicted,
  layouts,
  onNeedLayout,
  selected,
  onSelect,
  classNames,
  size = 84,
}: ProteinGalleryProps) {
  // Ask for what is missing. The worker remembers what it has laid out, and
  // `requestLayout` de-duplicates in-flight asks, so this is idempotent.
  useEffect(() => {
    for (const g of graphs) if (!layouts.has(g)) onNeedLayout(g);
  }, [graphs, layouts, onNeedLayout]);

  return (
    <ul
      data-testid="protein-gallery"
      className="flex flex-wrap gap-2"
    >
      {graphs.map((g) => {
        const correct = predicted[g] === labels[g];
        return (
          <li key={g}>
            <button
              type="button"
              onClick={() => onSelect(g)}
              aria-pressed={selected === g}
              title={`Protein #${g}: ${classNames[labels[g]]}, predicted ${classNames[predicted[g]]}`}
              className="block rounded-md focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
            >
              <ProteinTile
                layout={layouts.get(g)}
                correct={correct}
                selected={selected === g}
                size={size}
              />
              <span className="sr-only">
                Protein {g}, really {classNames[labels[g]]}, predicted{" "}
                {classNames[predicted[g]]}
              </span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}
