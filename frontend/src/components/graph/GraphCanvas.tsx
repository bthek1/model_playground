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

import {
  type MouseEvent as ReactMouseEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

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
  /**
   * Pairs to draw as dashed lines over the graph — `/link-prediction`'s
   * candidates. These are **predictions, not data**, which is why they are drawn
   * in a different style from the citations underneath them.
   */
  predicted?: Uint32Array;
  /** Two nodes to ring and join, when the viewer has picked a pair. */
  selected?: readonly [number, number] | null;
  /** Called with the node under a click, or null when the click missed one. */
  onPickNode?: (node: number | null) => void;
  /**
   * The scale this canvas is being displayed at, when a pan/zoom surface is
   * scaling it. Strokes are divided by it so a dot stays the same size on
   * screen at every zoom — which is what you want of a node-link view: zooming
   * in should pull overlapping nodes apart, not grow them into blobs.
   */
  zoom?: number;
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

/**
 * The node nearest a point, or null if nothing is near enough.
 *
 * The inverse of `layoutToPixels`, and pure for the same reason plus a sharper
 * one: a mis-mapped click returns a *plausible* node — a real index, in the
 * right part of the picture — so no count-based assertion downstream can catch
 * it. This is the same trap `/mask-generation`'s click and `/pose`'s keypoints
 * fell into, and the fix is the same: make the arithmetic testable and test it.
 *
 * `tolerance` is in the same CSS pixels as the layout, so a caller that is being
 * scaled by a pan/zoom surface should divide by the scale to keep the target the
 * same size under the pointer.
 */
export function pixelsToNode(
  points: PixelLayout,
  px: number,
  py: number,
  tolerance: number,
): number | null {
  let best: number | null = null;
  let bestDist = tolerance * tolerance;
  for (let i = 0; i < points.px.length; i++) {
    const dx = points.px[i] - px;
    const dy = points.py[i] - py;
    const dist = dx * dx + dy * dy;
    // `<=` so a click exactly on a node at exactly the tolerance still lands,
    // and so the first of two coincident nodes wins rather than neither.
    if (dist <= bestDist) {
      bestDist = dist;
      best = i;
    }
  }
  return best;
}

/**
 * How much to divide a stroke width by so it holds its size on screen.
 *
 * Pure and exported for the same reason `layoutToPixels` is: happy-dom gives a
 * canvas no 2D context, so anything left inside the paint effect is untested.
 * Quantised to 5 % steps because the edge layer is repainted whenever this
 * changes and a wheel drag reports a new scale every frame.
 */
export function strokeScale(zoom: number): number {
  return Math.max(0.05, Math.round(zoom * 20) / 20);
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
  predicted,
  selected = null,
  onPickNode,
  zoom = 1,
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

  const q = strokeScale(zoom);

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
    ctx.lineWidth = 0.5 / q;
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
  }, [size, dpr, q, points, nNodes, rowPtr, colIdx]);

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
    const radius = Math.max(1.4, points.span / 420) / q;

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
        ctx.lineWidth = 1 / q;
        ctx.stroke();
      }
    }

    // Predictions last, so they sit above the graph they are claims about.
    //
    // Both endpoints are ringed as well as joined, and that is not decoration.
    // The pairs a link predictor scores highest are the ones that already share
    // neighbours, so the layout has put them next to each other and the line
    // between them is a few pixels long — drawn but unfindable among 2708 dots.
    // The rings are what make a prediction locatable at the zoom the whole
    // graph fits in.
    if (predicted && predicted.length > 0) {
      ctx.save();
      ctx.strokeStyle = "rgba(244,63,94,0.9)"; // rose-500
      ctx.lineWidth = 1.5 / q;
      ctx.setLineDash([5 / q, 3 / q]);
      ctx.beginPath();
      for (let i = 0; i < predicted.length; i += 2) {
        const a = predicted[i];
        const b = predicted[i + 1];
        if (a >= nNodes || b >= nNodes) continue;
        ctx.moveTo(px[a], py[a]);
        ctx.lineTo(px[b], py[b]);
      }
      ctx.stroke();

      ctx.setLineDash([]);
      ctx.beginPath();
      for (let i = 0; i < predicted.length; i++) {
        const node = predicted[i];
        if (node >= nNodes) continue;
        const ring = Math.max(radius * 2.5, 4 / q);
        ctx.moveTo(px[node] + ring, py[node]);
        ctx.arc(px[node], py[node], ring, 0, Math.PI * 2);
      }
      ctx.stroke();
      ctx.restore();
    }

    if (selected) {
      const [a, b] = selected;
      ctx.save();
      ctx.strokeStyle = "rgba(14,165,233,0.95)"; // sky-500
      ctx.lineWidth = 2 / q;
      if (a < nNodes && b < nNodes && a !== b) {
        ctx.beginPath();
        ctx.moveTo(px[a], py[a]);
        ctx.lineTo(px[b], py[b]);
        ctx.stroke();
      }
      for (const node of [a, b]) {
        if (node >= nNodes) continue;
        ctx.beginPath();
        ctx.arc(px[node], py[node], (radius + 3) / q, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.restore();
    }
  }, [
    size,
    dpr,
    q,
    points,
    nNodes,
    labels,
    predictions,
    trainMask,
    colorBy,
    predicted,
    selected,
  ]);

  // A click is in CSS pixels of the *displayed* canvas, and the canvas is
  // measured in the layout's own pixels — the two differ by whatever a pan/zoom
  // surface is doing to it. Reading the element's own client rect is what keeps
  // that conversion correct without the canvas having to know it is being
  // scaled: the rect is already post-transform.
  const handleClick = (event: ReactMouseEvent<HTMLCanvasElement>) => {
    if (!onPickNode) return;
    const canvas = event.currentTarget;
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    const scale = rect.width / size.width;
    const x = (event.clientX - rect.left) / scale;
    const y = (event.clientY - rect.top) / scale;
    // A generous target in layout pixels, shrunk by however much the view is
    // magnified, so the node under the pointer stays the same size to click.
    onPickNode(pixelsToNode(points, x, y, Math.max(6, 10 / q)));
  };

  return (
    <div
      ref={hostRef}
      data-testid="graph-canvas"
      // A drawing surface, not a framed card: the caller owns the frame,
      // because on /graph the frame is the pan/zoom viewport and this box is
      // the thing that moves inside it.
      className="relative min-h-56 w-full flex-1 overflow-hidden"
    >
      <canvas
        ref={canvasRef}
        onClick={handleClick}
        role="img"
        aria-label={`Cora citation graph: ${nNodes} papers, coloured by ${
          colorBy === "true" ? "their true topic" : "the model's predicted topic"
        }. Ringed nodes are the labelled training set.`}
        style={{ width: "100%", height: "100%", display: "block" }}
      />
    </div>
  );
}
