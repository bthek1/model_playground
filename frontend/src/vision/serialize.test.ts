import { describe, expect, it } from "vitest";

import { toCloneable, type PlainTensor } from "./serialize";

/**
 * A stand-in for a Transformers.js `Tensor`: `data` and `dims` are **prototype
 * getters**, which is exactly why the real one cannot be structured-cloned.
 */
class FakeTensor {
  constructor(
    private readonly buffer: Float32Array,
    private readonly shape: number[],
  ) {}
  get data() {
    return this.buffer;
  }
  get dims() {
    return this.shape;
  }
  get type() {
    return "float32";
  }
}

class FakeRawImage {
  constructor(
    public data: Uint8ClampedArray,
    public width: number,
    public height: number,
    public channels: number,
  ) {}
}

describe("toCloneable", () => {
  it("flattens a tensor whose data and dims are prototype getters", () => {
    // The real failure this guards: "#<_Tensor> could not be cloned".
    const tensor = new FakeTensor(new Float32Array([1, 2, 3, 4]), [1, 2, 2]);
    expect(structuredClone(tensor)).not.toMatchObject({ dims: [1, 2, 2] });

    const out = toCloneable({ predicted_depth: tensor }) as {
      predicted_depth: PlainTensor;
    };
    expect(Array.from(out.predicted_depth.data)).toEqual([1, 2, 3, 4]);
    expect(out.predicted_depth.dims).toEqual([1, 2, 2]);
    expect(out.predicted_depth.type).toBe("float32");

    // And the whole thing now survives a real structured clone.
    expect(() => structuredClone(out)).not.toThrow();
  });

  it("copies the buffer rather than aliasing one the runtime may reuse", () => {
    const buffer = new Float32Array([1, 2]);
    const out = toCloneable(new FakeTensor(buffer, [2])) as PlainTensor;
    buffer[0] = 99;
    expect(Array.from(out.data)).toEqual([1, 2]);
  });

  it("flattens an image without mistaking it for a tensor", () => {
    const img = new FakeRawImage(new Uint8ClampedArray([1, 2, 3]), 1, 1, 3);
    expect(toCloneable(img)).toEqual({
      data: new Uint8ClampedArray([1, 2, 3]),
      width: 1,
      height: 1,
      channels: 3,
    });
  });

  it("walks arrays and nested objects, the segmentation and detection shapes", () => {
    const masks = [
      {
        label: "sky",
        score: 0.9,
        mask: new FakeRawImage(new Uint8ClampedArray([255]), 1, 1, 1),
      },
    ];
    const out = toCloneable(masks) as { label: string; mask: { width: number } }[];
    expect(out[0].label).toBe("sky");
    expect(out[0].mask.width).toBe(1);
    expect(() => structuredClone(out)).not.toThrow();
  });

  it("passes plain results through untouched", () => {
    const rows = [{ label: "tabby", score: 0.5 }];
    expect(toCloneable(rows)).toEqual(rows);
    expect(toCloneable("text")).toBe("text");
    expect(toCloneable(null)).toBeNull();
  });

  it("gives up rather than hanging on a self-referential result", () => {
    const loop: Record<string, unknown> = {};
    loop.self = loop;
    expect(() => toCloneable(loop)).not.toThrow();
  });
});
