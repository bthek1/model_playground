// The graph itself: 2708 papers and 5278 citations, drawn as **one canvas**.
//
// A DOM node per graph node is the obvious implementation and the wrong one —
// 2708 SVG circles will not animate, and this canvas is repainted on every epoch
// while the model trains. That repaint is the whole point of the page: twenty
// labelled nodes per class propagate outward through the graph, and watching the
// colours settle into communities is what a notebook cannot show.
//
// The edges are painted once into an offscreen layer and blitted, because they
// never change — only the node colours do. Repainting 5278 line segments per
// epoch alongside them would triple the per-frame cost to redraw an identical
// picture.

import { useEffect, useMemo, useRef, useState } from "react";

/**
 * Seven categorical colours, one per Cora topic, chosen to stay distinguishable
 * on both the light and the dark card background. Fixed values rather than theme
 * tokens for the same reason the diverging heatmap uses fixed endpoints: the
 * palette encodes *meaning*, and a class that changed hue with the theme would
 * be a different class to the eye. (model-visualization.md §3.)
 */
export const CLASS_COLORS = [
  "#ef4444", // red
  "#f59e0b", // amber
  "#10b981", // emerald
  "#3b82f6", // blue
  "#a855f7", // purple
  "#ec4899", // pink
  "#14b8a6", // teal
] as const;

export type ColorBy = "predicted" | "true";

export interface GraphCanvasProps {
  nNodes: number;
  rowPtr: Uint32Array;
  colIdx: Uint32Array;
  /** Layout coordinates, normalised into the unit square. */
  x: Float32Array;
  y: Float32Array;
  labels: Uint8Array;
  /** Null before the first epoch; every node then reads as unclassified. */
  predictions: Uint8Array | null;
  /** The 20-per-class labelled nodes, ringed so the supervision is visible. */
  trainMask: Uint8Array;
  colorBy: ColorBy;
}

/** Padding inside the canvas, in CSS pixels, so ringed nodes are not clipped. */
export const PAD = 8;

export interface PixelLayout {
  px: Float32Array;
  py: Float32Array;
  /** Side of the square the graph was drawn into, in CSS pixels. */
  span: number;
}

/**
 * Map unit-square layout coordinates onto the canvas box.
 *
 * Pure, and exported, because it is the only real arithmetic in this file and
 * happy-dom gives a canvas no 2D context — the painting below cannot be asserted
 * in a unit test, but this can. The graph is drawn into the largest centred
 * square that fits, so the aspect ratio of the layout survives a column that is
 * much wider than it is tall.
 */
export function layoutToPixels(
  x: Float32Array,
  y: Float32Array,
  nNodes: number,
  width: number,
  height: number,
): PixelLayout {
  const span = Math.max(0, Math.min(width, height) - 2 * PAD);
  // Centre the square in the box. Adding any further offset here shifts the
  // drawing off-centre and makes the far padding smaller than the near one.
  const offsetX = (width - span) / 2;
  const offsetY = (height - span) / 2;

  const px = new Float32Array(nNodes);
  const py = new Float32Array(nNodes);
  for (let i = 0; i < nNodes; i++) {
    px[i] = offsetX + x[i] * span;
    py[i] = offsetY + y[i] * span;
  }
  return { px, py, span };
}

export function GraphCanvas({
  nNodes,
  rowPtr,
  colIdx,
  x,
  y,
  labels,
  predictions,
  trainMask,
  colorBy,
}: GraphCanvasProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const edgeLayer = useRef<HTMLCanvasElement | null>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  // Track the host's box rather than assuming one: this canvas lives in a
  // workbench column whose width changes at every breakpoint.
  useEffect(() => {
    const host = hostRef.current;
    if (!host || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(([entry]) => {
      const box = entry.contentRect;
      setSize({ width: Math.round(box.width), height: Math.round(box.height) });
    });
    observer.observe(host);
    return () => observer.disconnect();
  }, []);

  const dpr =
    typeof window !== "undefined" ? Math.min(window.devicePixelRatio || 1, 2) : 1;

  // Pixel coordinates, recomputed only when the box or the layout changes.
  const points = useMemo(
    () => layoutToPixels(x, y, nNodes, size.width, size.height),
    [size, nNodes, x, y],
  );

  // --- the edge layer, painted once per size -------------------------------
  useEffect(() => {
    const { width, height } = size;
    if (width === 0 || height === 0) return;
    const layer = document.createElement("canvas");
    layer.width = Math.round(width * dpr);
    layer.height = Math.round(height * dpr);
    const ctx = layer.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    const { px, py } = points;
    ctx.strokeStyle = "rgba(128,128,128,0.28)";
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    for (let i = 0; i < nNodes; i++) {
      for (let e = rowPtr[i]; e < rowPtr[i + 1]; e++) {
        const j = colIdx[e];
        if (j <= i) continue; // each undirected citation is stored twice
        ctx.moveTo(px[i], py[i]);
        ctx.lineTo(px[j], py[j]);
      }
    }
    // One stroke for all 5278 segments: a stroke() per edge is the difference
    // between a frame and a slideshow.
    ctx.stroke();
    edgeLayer.current = layer;
  }, [size, dpr, points, nNodes, rowPtr, colIdx]);

  // --- nodes, repainted per epoch ------------------------------------------
  useEffect(() => {
    const canvas = canvasRef.current;
    const { width, height } = size;
    if (!canvas || width === 0 || height === 0) return;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    if (edgeLayer.current) {
      ctx.drawImage(edgeLayer.current, 0, 0, width, height);
    }

    const { px, py } = points;
    const classOf = colorBy === "true" ? labels : predictions;
    const radius = Math.max(1.4, points.span / 420);

    for (let i = 0; i < nNodes; i++) {
      // Before the first epoch there is no prediction; a neutral dot says so
      // rather than implying every node is class 0.
      const cls = classOf ? classOf[i] : null;
      ctx.fillStyle =
        cls == null
          ? "rgba(128,128,128,0.55)"
          : CLASS_COLORS[cls % CLASS_COLORS.length];
      ctx.beginPath();
      ctx.arc(px[i], py[i], trainMask[i] ? radius * 1.9 : radius, 0, Math.PI * 2);
      ctx.fill();

      if (trainMask[i]) {
        // The 140 nodes the model was actually told about.
        ctx.strokeStyle = "rgba(255,255,255,0.9)";
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }
  }, [size, dpr, points, nNodes, labels, predictions, trainMask, colorBy]);

  return (
    <div
      ref={hostRef}
      data-testid="graph-canvas"
      className="relative min-h-56 w-full flex-1 overflow-hidden rounded-md border bg-background"
    >
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={`Cora citation graph: ${nNodes} papers, coloured by ${
          colorBy === "true" ? "their true topic" : "the model's predicted topic"
        }. Ringed nodes are the labelled training set.`}
        style={{ width: "100%", height: "100%", display: "block" }}
      />
    </div>
  );
}
