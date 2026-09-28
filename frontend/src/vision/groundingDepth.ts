// Reading a distance for each box off a depth map — `/robotics`'s geometry.
//
// Pure and React-free, because both of its failure modes are silent and only a
// unit test over hand-built maps can pin them:
//
//  1. **The direction.** Depth Anything emits *inverse* depth — a big value is
//     **near** — while a metric model emits metres, where a big value is far.
//     Getting that backwards produces a complete, confident, correctly-formatted
//     answer with the ordering inverted: the far car is "nearest". So the
//     direction is an argument read from the catalogue (`DepthModel.metric`),
//     never a constant here, and what comes back is an explicit **rank**, so the
//     route never has to know which way the raw number points.
//
//  2. **The statistic.** One pixel at the box's centre is the obvious read and
//     the wrong one: on a thin object — a pole, a bicycle frame, a person seen
//     side-on — the centre pixel is often background, and the page then reports
//     the wall's distance with the object's label on it. So this takes the
//     **median over the central half** of the box: central, because a detector's
//     box edges are background by construction; median, because a box over a
//     mug in front of a wall is bimodal and a mean lands in the gap between the
//     two, at a depth where nothing is.
//
// The raw value is kept alongside the rank, labelled as what it is (the model's
// own units — relative inverse depth, not metres), exactly as `transform.ts` on
// `/tabular-regression` keeps units below the route so the route cannot mislabel
// them. `nearness` puts it on a scale a reader can use: where the box sits
// between the farthest (0) and the nearest (1) point **in this frame**.

import type { Box, Detection } from "./draw";

/** A single-channel map, row-major, as `depthDims` reads it off the tensor. */
export interface DepthMap {
  data: ArrayLike<number>;
  width: number;
  height: number;
}

/** The pixel space the boxes are in — the frame both models were given. */
export interface FrameSize {
  width: number;
  height: number;
}

export interface GroundedDetection extends Detection {
  /**
   * Median of the depth model's raw output over the box's central region, in
   * the model's own units. For Depth Anything that is **relative inverse depth**
   * on a per-image scale: bigger is nearer, and it is not metres. `null` when
   * the box covers no pixel of the map.
   */
  depth: number | null;
  /**
   * 0 = the farthest point in this frame, 1 = the nearest. Per-frame, so two
   * frames' numbers are not comparable. `null` with `depth`.
   */
  nearness: number | null;
  /** 1 = nearest of the measured boxes. `null` with `depth`. */
  rank: number | null;
  /** True on exactly one box whenever any box was measured. */
  nearest: boolean;
}

/**
 * How much of the box, per side, is sampled: the central 50% of its width and
 * of its height, i.e. the middle quarter of its area.
 */
export const CENTRAL_FRACTION = 0.5;

/**
 * Samples at most this many pixels per box. A box over half a 640x640 frame is
 * ~50k pixels, and a median over a strided subset is indistinguishable from the
 * full one while costing a sort of 4k values instead of 50k.
 */
export const MAX_SAMPLES = 4096;

/** Median of a non-empty list. Sorts a copy. */
export function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 === 1
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * The integer pixel range `[start, end)` whose pixel *centres* fall inside the
 * continuous interval `[lo, hi)` — with the guarantee that a non-empty interval
 * yields at least one pixel.
 *
 * Pixel `i` covers `[i, i + 1)` and its centre is `i + 0.5`. The central
 * region of a one-pixel-wide box at `[5, 6)` is `[5.25, 5.75)`; a sliver
 * narrower than that, or one straddling a pixel edge, can contain no centre at
 * all, and a centres-only rule would leave the box unmeasured — while a naive
 * `round()` of its edges can land on the neighbouring pixel, which is the
 * background. So the pixel containing the interval's midpoint is always
 * included, and the range widens only to pixels whose centre the interval
 * actually covers.
 */
function pixelSpan(lo: number, hi: number, size: number): [number, number] {
  if (!(hi > lo) || size <= 0) return [0, 0];
  const clampLo = Math.max(0, lo);
  const clampHi = Math.min(size, hi);
  if (!(clampHi > clampLo)) return [0, 0];
  let start = Math.ceil(clampLo - 0.5);
  let end = Math.ceil(clampHi - 0.5);
  const mid = Math.min(size - 1, Math.floor((clampLo + clampHi) / 2));
  start = Math.min(Math.max(0, start), mid);
  end = Math.max(Math.min(size, end), mid + 1);
  return [start, end];
}

/**
 * Median depth over the central region of `box`, which is in `frame` pixels,
 * read off `map`, which may be a different resolution (the depth pipeline
 * interpolates to its own size). `null` for an empty or off-map box.
 */
export function sampleBoxDepth(
  box: Box,
  map: DepthMap,
  frame: FrameSize,
  central = CENTRAL_FRACTION,
): number | null {
  const { width: mw, height: mh, data } = map;
  if (mw <= 0 || mh <= 0 || frame.width <= 0 || frame.height <= 0) return null;
  const w = box.xmax - box.xmin;
  const h = box.ymax - box.ymin;
  if (!(w > 0) || !(h > 0)) return null;

  const sx = mw / frame.width;
  const sy = mh / frame.height;
  const inset = (1 - central) / 2;
  const [x0, x1] = pixelSpan(
    (box.xmin + w * inset) * sx,
    (box.xmax - w * inset) * sx,
    mw,
  );
  const [y0, y1] = pixelSpan(
    (box.ymin + h * inset) * sy,
    (box.ymax - h * inset) * sy,
    mh,
  );
  const cols = x1 - x0;
  const rows = y1 - y0;
  if (cols <= 0 || rows <= 0) return null;

  // Stride evenly in both axes so a large box is subsampled, never cropped.
  const stride = Math.max(1, Math.ceil(Math.sqrt((cols * rows) / MAX_SAMPLES)));
  const values: number[] = [];
  for (let y = y0; y < y1; y += stride) {
    for (let x = x0; x < x1; x += stride) {
      const v = data[y * mw + x];
      if (Number.isFinite(v)) values.push(v);
    }
  }
  return values.length > 0 ? median(values) : null;
}

/**
 * Attach a depth, a per-frame nearness and a rank to every detection, in the
 * order they were given.
 *
 * `metric` is the catalogue's `DepthModel.metric`: false (every shipped entry)
 * means inverse depth, so a **bigger** raw value ranks **nearer**; true means
 * metres, so a smaller one does. Ties keep input order, so the higher-scoring
 * box wins a tie when the caller passes detections sorted by score.
 */
export function groundDepths(
  detections: readonly Detection[],
  map: DepthMap,
  frame: FrameSize,
  { metric = false, central = CENTRAL_FRACTION } = {},
): GroundedDetection[] {
  const depths = detections.map((d) =>
    sampleBoxDepth(d.box, map, frame, central),
  );

  // The frame's own range, for `nearness`. Computed over the map rather than
  // over the boxes, so a single box is not trivially "the nearest thing there
  // is" and two boxes are not stretched to 0 and 1.
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < map.data.length; i++) {
    const v = map.data[i];
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const span = hi - lo;

  // Closeness in one direction regardless of the convention: bigger = nearer.
  const closeness = (v: number) => (metric ? -v : v);

  const order = depths
    .map((v, i) => ({ v, i }))
    .filter((e): e is { v: number; i: number } => e.v != null)
    .sort((a, b) => closeness(b.v) - closeness(a.v) || a.i - b.i);
  const rankOf = new Map(order.map((e, r) => [e.i, r + 1]));

  return detections.map((d, i) => {
    const v = depths[i];
    let nearness: number | null = null;
    if (v != null) {
      const t = span > 0 && Number.isFinite(span) ? (v - lo) / span : 1;
      nearness = Math.min(1, Math.max(0, metric ? 1 - t : t));
    }
    const rank = rankOf.get(i) ?? null;
    return { ...d, depth: v, nearness, rank, nearest: rank === 1 };
  });
}

/** The nearest measured detection, or null when nothing was measured. */
export function nearestOf(
  grounded: readonly GroundedDetection[],
): GroundedDetection | null {
  return grounded.find((g) => g.nearest) ?? null;
}

/** "nearest", "2nd", "3rd", "4th"… for a rank. Never a unit. */
export function rankLabel(rank: number | null): string {
  if (rank == null) return "not measured";
  if (rank === 1) return "nearest";
  const tens = rank % 100;
  const suffix =
    tens >= 11 && tens <= 13
      ? "th"
      : rank % 10 === 1
        ? "st"
        : rank % 10 === 2
          ? "nd"
          : rank % 10 === 3
            ? "rd"
            : "th";
  return `${rank}${suffix} nearest`;
}
