// Unit-square layout → SVG coordinates, and the colour ramp the diagram uses.
// Pure, so the two things a picture can get quietly wrong — where a node lands
// and how strongly it is shaded — are testable without a renderer.

export interface Box {
  width: number;
  height: number;
  /** Clear space kept round the drawing: a node's radius plus its label. */
  pad: number;
}

/**
 * Fit a layout to its **own extent**, centred, aspect ratio kept. Fitting to the
 * unit square instead leaves the path graph (all its y within ±0.08) as a thin
 * line in the middle of an empty box; a single axis of zero extent (one node,
 * or a straight line) is centred on that axis rather than divided by zero.
 */
export function fitToBox(
  layout: { x: readonly number[]; y: readonly number[] },
  { width, height, pad }: Box,
): { x: number[]; y: number[] } {
  const n = layout.x.length;
  if (n === 0) return { x: [], y: [] };
  const minX = Math.min(...layout.x);
  const maxX = Math.max(...layout.x);
  const minY = Math.min(...layout.y);
  const maxY = Math.max(...layout.y);
  const spanX = maxX - minX;
  const spanY = maxY - minY;
  const innerW = Math.max(0, width - 2 * pad);
  const innerH = Math.max(0, height - 2 * pad);

  const scales = [spanX > 0 ? innerW / spanX : Infinity, spanY > 0 ? innerH / spanY : Infinity];
  const finite = scales.filter(Number.isFinite);
  const scale = finite.length ? Math.min(...finite) : 0;

  const offX = pad + (innerW - spanX * scale) / 2;
  const offY = pad + (innerH - spanY * scale) / 2;
  return {
    x: layout.x.map((v) => offX + (v - minX) * scale),
    y: layout.y.map((v) => offY + (v - minY) * scale),
  };
}

/**
 * Fill opacity for a node at `distance` hops once BFS has run `k` steps.
 * Sequential, near = strong (model-visualization §4): the source is the
 * strongest, the frontier the faintest reached, and unreached nodes are drawn
 * separately in the muted colour. Capped at 0.55 so the distance printed on
 * the node stays legible over the fill in both themes.
 */
export function reachedOpacity(distance: number, k: number): number {
  if (distance < 0 || distance > k) return 0;
  if (k === 0) return 0.55;
  return 0.55 - (0.35 * distance) / k;
}

/**
 * Walk-count cell opacity, on a log scale. Counts on one row span 1 to
 * thousands, and a linear ramp would draw every cell but the hub's as empty —
 * which reads as "no walk", the one thing a zero cell must mean alone.
 */
export function countOpacity(count: number, max: number): number {
  if (count <= 0 || max <= 0) return 0;
  return 0.15 + (0.85 * Math.log1p(count)) / Math.log1p(max);
}
