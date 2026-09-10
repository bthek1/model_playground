// Depth map + camera intrinsics → a coloured point cloud.
//
// The whole of `/image-to-3d`'s geometry lives here, and it is arithmetic: for a
// pinhole camera, a pixel at (u, v) with depth z sits at
//
//     x = (u - cx) * z / f
//     y = (v - cy) * z / f
//     z = z
//
// which is the projection equation read backwards. `/depth` already produces the
// depth map, so this page adds no model at all — which is what makes it, per
// docs/roadmaps/vision.md §3.12, the cheapest impressive 3-D demo available.
//
// ---
//
// **`f` is an assumption, and the page has to say so.**
//
// Depth Anything V2 predicts *relative inverse* depth with no camera intrinsics
// attached: it does not know, and cannot know, the focal length of the lens that
// took the picture. So `f` is a slider with a plausible default, the geometry it
// produces is plausible rather than metric, and two clouds from two photos are
// not comparable. Depth Pro is the exception — it predicts metres *and* a focal
// length — which is exactly why {@link unproject} takes `inverse` as a flag
// rather than assuming one convention.
//
// **The inverse-depth conversion is the part that fails silently.** Feed an
// inverse-depth map straight in as `z` and you get a recognisable cloud that is
// inside out: near things land far away, and the scene reads as a bowl. It looks
// like a point cloud, so nothing complains — which is why the tests below check
// that a nearer pixel ends up with a smaller z, and not merely that points were
// produced.

import type { Pixels } from "./draw";

/** Interleaved `[x, y, z, r, g, b]` per point, ready for a vertex buffer. */
export interface PointCloud {
  /** 6 floats per point: position in metres-ish, colour in 0–1. */
  data: Float32Array;
  count: number;
  /** Axis-aligned bounds, for framing the camera and for a sanity check. */
  bounds: { min: [number, number, number]; max: [number, number, number] };
}

/** Floats per point in {@link PointCloud.data}. */
export const POINT_STRIDE = 6;

/**
 * Default focal length, as a multiple of the image's larger side.
 *
 * ~0.8 corresponds to a horizontal field of view around 64°, which is the range
 * most phone and compact cameras sit in. It is a *plausible* lens, not the one
 * that took the picture.
 */
export const DEFAULT_FOCAL_RATIO = 0.8;

/**
 * Smallest inverse-depth value the far plane is allowed to reach, which fixes
 * how deep the scene can be: `z` runs from 1 (nearest) to `1 / MIN_INVERSE`
 * (furthest), so 0.1 gives a 10:1 depth range — about what a real photograph
 * spans.
 *
 * Without a floor the far plane goes to infinity: a normalised map has a pixel
 * at exactly 0, its reciprocal is unbounded, and the bounds it produces put the
 * whole visible scene in a speck at the origin. That renders as an empty canvas
 * with a few stray points, which reads as "the model failed" rather than as a
 * scaling choice.
 */
const MIN_INVERSE = 0.1;

/** Guard against a zero depth blowing up to infinity on the metric path. */
const EPS = 1e-3;

export interface UnprojectOptions {
  /** Focal length in pixels. Defaults to `DEFAULT_FOCAL_RATIO * max(w, h)`. */
  focal?: number;
  /**
   * Take every `stride`-th pixel on each axis. One point per pixel at 1080p is
   * two million points before any culling, which is more than a first render
   * should ask a laptop for.
   */
  stride?: number;
  /**
   * True when the map is **inverse** depth (Depth Anything: big = near), false
   * when it is already a distance (Depth Pro: big = far). Getting this backwards
   * turns the scene inside out and still looks like a point cloud.
   */
  inverse?: boolean;
  /** Colours, sampled per point. Any size — it is sampled proportionally. */
  colour?: Pixels;
}

/** A depth map: values plus the grid they sit on. */
export interface DepthMap {
  data: ArrayLike<number>;
  width: number;
  height: number;
}

/**
 * Unproject a depth map into a coloured point cloud.
 *
 * The depth values are normalised to 0–1 across the map first, because relative
 * depth has an arbitrary per-image scale — without that, a map whose values
 * happen to run 300–800 produces a cloud hundreds of units from the origin and
 * the camera starts inside nothing.
 */
export function unproject(
  depth: DepthMap,
  {
    focal,
    stride = 2,
    inverse = true,
    colour,
  }: UnprojectOptions = {},
): PointCloud {
  const { width, height } = depth;
  const step = Math.max(1, Math.floor(stride));
  const f = focal ?? DEFAULT_FOCAL_RATIO * Math.max(width, height);
  const cx = width / 2;
  const cy = height / 2;

  // Normalise the map to 0–1. Relative depth is on an arbitrary scale, so this
  // is what keeps the cloud near the origin whatever the model emitted.
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < depth.data.length; i++) {
    const v = depth.data[i];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
    lo = 0;
    hi = 1;
  }
  const span = hi - lo || 1;

  const cols = Math.ceil(width / step);
  const rows = Math.ceil(height / step);
  const count = cols * rows;
  const data = new Float32Array(count * POINT_STRIDE);

  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];

  let n = 0;
  for (let v = 0; v < height; v += step) {
    for (let u = 0; u < width; u += step) {
      const t = (depth.data[v * width + u] - lo) / span; // 0–1

      // **Inverse depth: a big value means NEAR, so distance is its
      // reciprocal.** Getting this the wrong way round turns the scene inside
      // out — near things land far away and a room reads as a bowl — while
      // still producing a perfectly plausible-looking point cloud, which is why
      // `pointCloud.test.ts` asserts the *ordering* rather than the count.
      //
      // `t` is remapped into [MIN_INVERSE, 1] before the reciprocal so the far
      // plane stays finite; see MIN_INVERSE.
      const z = inverse ? 1 / (MIN_INVERSE + (1 - MIN_INVERSE) * t) : t + EPS;

      const x = ((u - cx) * z) / f;
      const y = ((v - cy) * z) / f;

      const o = n * POINT_STRIDE;
      data[o] = x;
      data[o + 1] = y;
      data[o + 2] = z;

      if (colour) {
        const [r, g, b] = sampleColour(colour, u / width, v / height);
        data[o + 3] = r;
        data[o + 4] = g;
        data[o + 5] = b;
      } else {
        // Grey by luminance of the depth itself, so a cloud with no picture
        // attached is still readable rather than uniformly black.
        data[o + 3] = data[o + 4] = data[o + 5] = t;
      }

      // Bounds are taken from the values **as stored**, not from the doubles
      // that produced them: the buffer is float32, so a double read back is
      // rounded and can land a hair outside a bound computed before the write.
      // A consumer that clamps to these bounds would then drop a point.
      const px = data[o];
      const py = data[o + 1];
      const pz = data[o + 2];
      if (px < min[0]) min[0] = px;
      if (py < min[1]) min[1] = py;
      if (pz < min[2]) min[2] = pz;
      if (px > max[0]) max[0] = px;
      if (py > max[1]) max[1] = py;
      if (pz > max[2]) max[2] = pz;

      n++;
    }
  }

  if (n === 0) {
    return {
      data,
      count: 0,
      bounds: { min: [0, 0, 0], max: [0, 0, 0] },
    };
  }

  return { data, count: n, bounds: { min, max } };
}

/**
 * Sample a colour buffer at fractional coordinates, as 0–1 RGB.
 *
 * Proportional rather than absolute because the depth map and the source photo
 * are routinely different sizes — the pipeline resizes the map back to the
 * source, but a model that does not would otherwise stripe the colours.
 */
function sampleColour(
  image: Pixels,
  fx: number,
  fy: number,
): [number, number, number] {
  const x = Math.min(image.width - 1, Math.max(0, Math.round(fx * image.width)));
  const y = Math.min(image.height - 1, Math.max(0, Math.round(fy * image.height)));
  const c = image.channels;
  const i = (y * image.width + x) * c;

  if (c === 1) {
    const v = image.data[i] / 255;
    return [v, v, v];
  }
  return [image.data[i] / 255, image.data[i + 1] / 255, image.data[i + 2] / 255];
}

/** How many points a stride yields, without building the cloud. */
export function pointCount(
  width: number,
  height: number,
  stride: number,
): number {
  const step = Math.max(1, Math.floor(stride));
  return Math.ceil(width / step) * Math.ceil(height / step);
}
