// `/robotics`'s geometry: a distance per box, read off a depth map. Both of the
// failures this guards are silent — an inverted convention returns a confident
// answer with the ordering backwards, and a centre-pixel read on a thin object
// returns the wall's distance with the object's label on it — so every test
// here builds a map where the right answer and the wrong one differ.

import { describe, expect, it } from "vitest";

import type { Detection } from "./draw";
import {
  groundDepths,
  median,
  nearestOf,
  rankLabel,
  sampleBoxDepth,
  type DepthMap,
} from "./groundingDepth";

/** A `w`x`h` map filled with `bg`, with rectangles painted over it. */
function map(
  w: number,
  h: number,
  bg: number,
  rects: { x0: number; y0: number; x1: number; y1: number; v: number }[] = [],
): DepthMap {
  const data = new Float32Array(w * h).fill(bg);
  for (const r of rects) {
    for (let y = r.y0; y < r.y1; y++) {
      for (let x = r.x0; x < r.x1; x++) data[y * w + x] = r.v;
    }
  }
  return { data, width: w, height: h };
}

const det = (
  label: string,
  xmin: number,
  ymin: number,
  xmax: number,
  ymax: number,
  score = 0.5,
): Detection => ({ label, score, box: { xmin, ymin, xmax, ymax } });

const FRAME = { width: 40, height: 20 };

describe("median", () => {
  it("takes the middle of an odd list and the mean of the middle pair of an even one", () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(median([4, 1, 3, 2])).toBe(2.5);
  });

  it("does not reorder its input", () => {
    const xs = [3, 1, 2];
    median(xs);
    expect(xs).toEqual([3, 1, 2]);
  });
});

describe("sampleBoxDepth", () => {
  it("reads the object, not the background its box edges include", () => {
    // A box 20 px wide around a 10 px object: the outer quarter on each side is
    // background, and a mean over the whole box would land between the two.
    const m = map(40, 20, 1, [{ x0: 15, y0: 5, x1: 25, y1: 15, v: 9 }]);
    expect(sampleBoxDepth({ xmin: 10, ymin: 0, xmax: 30, ymax: 20 }, m, FRAME)).toBe(9);
  });

  it("is a median, so a bimodal box does not report a depth where nothing is", () => {
    // Left 60% of the box is the object (9), right 40% the wall (1). A mean
    // would say ~5.8; nothing in the picture is at 5.8.
    const m = map(40, 20, 1, [{ x0: 0, y0: 0, x1: 22, y1: 20, v: 9 }]);
    const v = sampleBoxDepth({ xmin: 10, ymin: 0, xmax: 30, ymax: 20 }, m, FRAME);
    expect(v === 9 || v === 1).toBe(true);
  });

  it("reads a one-pixel-wide box from its own column, never its neighbour's", () => {
    // A pole: column 7 is near (9), both neighbours are background (1).
    const m = map(40, 20, 1, [{ x0: 7, y0: 0, x1: 8, y1: 20, v: 9 }]);
    expect(sampleBoxDepth({ xmin: 7, ymin: 2, xmax: 8, ymax: 18 }, m, FRAME)).toBe(9);
  });

  it("still measures a sliver narrower than a pixel", () => {
    const m = map(40, 20, 1, [{ x0: 7, y0: 0, x1: 8, y1: 20, v: 9 }]);
    expect(
      sampleBoxDepth({ xmin: 7.4, ymin: 2, xmax: 7.6, ymax: 18 }, m, FRAME),
    ).toBe(9);
  });

  it("maps frame pixels onto a map of a different resolution", () => {
    // The map is half the frame's resolution; the object sits at the same
    // *relative* place. Reading the box in frame pixels would miss it.
    const m = map(20, 10, 1, [{ x0: 12, y0: 3, x1: 18, y1: 8, v: 7 }]);
    expect(
      sampleBoxDepth({ xmin: 24, ymin: 6, xmax: 36, ymax: 16 }, m, FRAME),
    ).toBe(7);
  });

  it("returns null for an empty, inverted or off-map box", () => {
    const m = map(40, 20, 1);
    expect(sampleBoxDepth({ xmin: 5, ymin: 5, xmax: 5, ymax: 10 }, m, FRAME)).toBeNull();
    expect(sampleBoxDepth({ xmin: 9, ymin: 5, xmax: 5, ymax: 10 }, m, FRAME)).toBeNull();
    expect(sampleBoxDepth({ xmin: 50, ymin: 5, xmax: 60, ymax: 10 }, m, FRAME)).toBeNull();
    expect(
      sampleBoxDepth({ xmin: 0, ymin: 0, xmax: 1, ymax: 1 }, { data: [], width: 0, height: 0 }, FRAME),
    ).toBeNull();
  });

  it("clips a box that runs off the edge rather than reading out of bounds", () => {
    const m = map(40, 20, 3);
    expect(sampleBoxDepth({ xmin: 30, ymin: -10, xmax: 60, ymax: 30 }, m, FRAME)).toBe(3);
  });

  it("subsamples a huge box without changing a uniform answer", () => {
    const m = map(400, 400, 4);
    expect(
      sampleBoxDepth({ xmin: 0, ymin: 0, xmax: 400, ymax: 400 }, m, { width: 400, height: 400 }),
    ).toBe(4);
  });
});

describe("groundDepths", () => {
  // Two objects: a near one on the left, a far one on the right, and a
  // mid-distance background. The *values* differ by convention; the scene
  // does not.
  const scene = (metric: boolean) => {
    const near = metric ? 0.5 : 20; // metres vs inverse depth
    const far = metric ? 8 : 2;
    const bg = metric ? 4 : 5;
    return map(40, 20, bg, [
      { x0: 2, y0: 2, x1: 12, y1: 18, v: near },
      { x0: 28, y0: 6, x1: 36, y1: 14, v: far },
    ]);
  };
  const boxes = [det("far thing", 27, 5, 37, 15, 0.9), det("near thing", 1, 1, 13, 19, 0.3)];

  for (const metric of [false, true]) {
    const name = metric ? "metric depth (big = far)" : "inverse depth (big = near)";

    it(`ranks the near object nearer under ${name}`, () => {
      const out = groundDepths(boxes, scene(metric), FRAME, { metric });
      const near = out.find((g) => g.label === "near thing")!;
      const far = out.find((g) => g.label === "far thing")!;
      expect(near.rank).toBe(1);
      expect(far.rank).toBe(2);
      expect(near.nearest).toBe(true);
      expect(far.nearest).toBe(false);
      expect(near.nearness!).toBeGreaterThan(far.nearness!);
      expect(nearestOf(out)?.label).toBe("near thing");
    });
  }

  it("inverts the ordering if the convention is read the wrong way — which is why it is an argument", () => {
    // The same inverse-depth map, mislabelled as metric: the far thing wins.
    // This is the bug the catalogue flag exists to prevent, demonstrated.
    const out = groundDepths(boxes, scene(false), FRAME, { metric: true });
    expect(nearestOf(out)?.label).toBe("far thing");
  });

  it("keeps the input order and the raw value in the model's own units", () => {
    const out = groundDepths(boxes, scene(false), FRAME);
    expect(out.map((g) => g.label)).toEqual(["far thing", "near thing"]);
    expect(out[0].depth).toBe(2);
    expect(out[1].depth).toBe(20);
    // Scores and boxes pass through untouched.
    expect(out[0].score).toBe(0.9);
    expect(out[0].box).toEqual(boxes[0].box);
  });

  it("puts nearness on the frame's own range, not the boxes'", () => {
    // One box is not trivially "the nearest thing there is": the near object
    // is the frame's maximum, the background sits between.
    const [bg] = groundDepths([det("wall", 14, 0, 26, 20)], scene(false), FRAME);
    expect(bg.nearness!).toBeGreaterThan(0);
    expect(bg.nearness!).toBeLessThan(1);
  });

  it("leaves an unmeasurable box unranked, and ranks the rest", () => {
    const out = groundDepths(
      [det("empty", 5, 5, 5, 5), ...boxes],
      scene(false),
      FRAME,
    );
    expect(out[0]).toMatchObject({ depth: null, nearness: null, rank: null, nearest: false });
    expect(out.filter((g) => g.nearest)).toHaveLength(1);
  });

  it("returns nothing nearest when there is nothing", () => {
    expect(groundDepths([], scene(false), FRAME)).toEqual([]);
    expect(nearestOf([])).toBeNull();
  });

  it("breaks a tie by input order, so the higher-scoring box wins when sorted by score", () => {
    const flat = map(40, 20, 3);
    const out = groundDepths([det("first", 0, 0, 10, 10), det("second", 20, 0, 30, 10)], flat, FRAME);
    expect(out.map((g) => g.rank)).toEqual([1, 2]);
    // A constant map has no range; nearness is defined rather than NaN.
    expect(Number.isFinite(out[0].nearness!)).toBe(true);
  });
});

describe("rankLabel", () => {
  it("says an ordering, never a unit", () => {
    expect(rankLabel(1)).toBe("nearest");
    expect(rankLabel(2)).toBe("2nd nearest");
    expect(rankLabel(3)).toBe("3rd nearest");
    expect(rankLabel(4)).toBe("4th nearest");
    expect(rankLabel(11)).toBe("11th nearest");
    expect(rankLabel(22)).toBe("22nd nearest");
    expect(rankLabel(null)).toBe("not measured");
  });
});
