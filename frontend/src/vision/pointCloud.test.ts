import { describe, expect, it } from "vitest";

import type { Pixels } from "./draw";
import {
  DEFAULT_FOCAL_RATIO,
  POINT_STRIDE,
  pointCount,
  unproject,
  type DepthMap,
} from "./pointCloud";

/** A depth map with one value everywhere. */
function flat(width: number, height: number, value: number): DepthMap {
  return { data: new Float32Array(width * height).fill(value), width, height };
}

/** Read one point out of the interleaved buffer. */
function pointAt(cloud: { data: Float32Array }, i: number) {
  const o = i * POINT_STRIDE;
  return {
    x: cloud.data[o],
    y: cloud.data[o + 1],
    z: cloud.data[o + 2],
    r: cloud.data[o + 3],
    g: cloud.data[o + 4],
    b: cloud.data[o + 5],
  };
}

describe("unproject", () => {
  it("turns a constant depth map into a plane", () => {
    // Every pixel at the same depth is, by definition, a fronto-parallel plane.
    // If z varies here, the normalisation or the reciprocal is wrong.
    const cloud = unproject(flat(8, 8, 0.5), { stride: 1 });
    const z0 = pointAt(cloud, 0).z;
    for (let i = 1; i < cloud.count; i++) {
      expect(pointAt(cloud, i).z).toBeCloseTo(z0, 6);
    }
  });

  it("puts the centre pixel on the optical axis", () => {
    // (u, v) = (cx, cy) must give x = y = 0 whatever the depth or the focal
    // length. An off-by-one in the principal point tilts the whole cloud.
    // At an even size cx = cy = 4 lands exactly on a pixel, so the assertion
    // is exact rather than approximate.
    const cloud = unproject(flat(8, 8, 0.5), { stride: 1 });
    const centre = pointAt(cloud, 4 * 8 + 4); // u = v = 4 = cx = cy
    expect(centre.x).toBe(0);
    expect(centre.y).toBe(0);
  });

  it("halves the x/y spread when the focal length doubles", () => {
    // x = (u - cx) * z / f, so f -> 2f must halve x. A long lens flattens the
    // scene; a wrong `f` silently changes the shape of everything.
    const depth = flat(16, 16, 0.5);
    const near = unproject(depth, { stride: 1, focal: 100 });
    const far = unproject(depth, { stride: 1, focal: 200 });

    const a = pointAt(near, 0);
    const b = pointAt(far, 0);
    expect(b.x).toBeCloseTo(a.x / 2, 6);
    expect(b.y).toBeCloseTo(a.y / 2, 6);
    // Depth itself is unchanged by the lens.
    expect(b.z).toBeCloseTo(a.z, 6);
  });

  it("defaults the focal length to a plausible lens, not to 1", () => {
    const depth = flat(64, 32, 0.5);
    const auto = unproject(depth, { stride: 1 });
    const explicit = unproject(depth, {
      stride: 1,
      focal: DEFAULT_FOCAL_RATIO * 64,
    });
    expect(pointAt(auto, 0).x).toBeCloseTo(pointAt(explicit, 0).x, 6);
  });

  it("puts a nearer pixel closer to the camera on an inverse-depth map", () => {
    // The silent one. Depth Anything emits *inverse* depth — a big number is
    // near — so feeding it straight in as z turns the scene inside out. It
    // still looks like a point cloud, so only the ordering catches it.
    const data = new Float32Array([0, 1, 0, 1]); // pixel 1 is the near one
    const cloud = unproject({ data, width: 2, height: 2 }, {
      stride: 1,
      inverse: true,
    });
    expect(pointAt(cloud, 1).z).toBeLessThan(pointAt(cloud, 0).z);
  });

  it("keeps a metric map the right way round", () => {
    // Depth Pro emits metres: a big number is far. Same data, opposite order.
    const data = new Float32Array([0, 1, 0, 1]);
    const cloud = unproject({ data, width: 2, height: 2 }, {
      stride: 1,
      inverse: false,
    });
    expect(pointAt(cloud, 1).z).toBeGreaterThan(pointAt(cloud, 0).z);
  });

  it("normalises an arbitrary value range rather than trusting it", () => {
    // Relative depth has no fixed scale. A map running 300-800 must produce the
    // same cloud as the same map scaled to 0-1, or the camera starts nowhere
    // near the points.
    const small = { data: new Float32Array([0, 0.5, 1, 0.25]), width: 2, height: 2 };
    const large = {
      data: new Float32Array([300, 550, 800, 425]),
      width: 2,
      height: 2,
    };
    const a = unproject(small, { stride: 1 });
    const b = unproject(large, { stride: 1 });
    for (let i = 0; i < a.count; i++) {
      expect(pointAt(b, i).z).toBeCloseTo(pointAt(a, i).z, 5);
    }
  });

  it("subsamples by stride, on both axes", () => {
    const cloud = unproject(flat(100, 60, 0.5), { stride: 4 });
    expect(cloud.count).toBe(25 * 15);
    expect(cloud.count).toBe(pointCount(100, 60, 4));
  });

  it("samples colour from the source picture", () => {
    const colour: Pixels = {
      data: new Uint8ClampedArray([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0]),
      width: 2,
      height: 2,
      channels: 3,
    };
    const cloud = unproject(flat(2, 2, 0.5), { stride: 1, colour });
    const first = pointAt(cloud, 0);
    expect(first.r).toBeCloseTo(1, 5);
    expect(first.g).toBeCloseTo(0, 5);
  });

  it("reports bounds that actually contain every point", () => {
    const cloud = unproject(
      { data: new Float32Array([0, 0.3, 0.7, 1]), width: 2, height: 2 },
      { stride: 1 },
    );
    for (let i = 0; i < cloud.count; i++) {
      const p = pointAt(cloud, i);
      expect(p.x).toBeGreaterThanOrEqual(cloud.bounds.min[0]);
      expect(p.x).toBeLessThanOrEqual(cloud.bounds.max[0]);
      expect(p.z).toBeGreaterThanOrEqual(cloud.bounds.min[2]);
      expect(p.z).toBeLessThanOrEqual(cloud.bounds.max[2]);
    }
  });

  it("produces a non-degenerate z range from a varying map", () => {
    // A collapsed z range is what a broken normalisation looks like, and every
    // point still renders. This is the `@slow` spec's assertion, in miniature.
    const data = new Float32Array(64);
    for (let i = 0; i < 64; i++) data[i] = i / 63;
    const cloud = unproject({ data, width: 8, height: 8 }, { stride: 1 });
    expect(cloud.bounds.max[2] - cloud.bounds.min[2]).toBeGreaterThan(0.1);
  });

  it("survives an empty map without producing NaN bounds", () => {
    const cloud = unproject({ data: new Float32Array(), width: 0, height: 0 });
    expect(cloud.count).toBe(0);
    expect(cloud.bounds.min.every(Number.isFinite)).toBe(true);
  });
});
