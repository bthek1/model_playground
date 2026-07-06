import { describe, expect, it } from "vitest";

import {
  IMAGE_SIZE,
  labelsFromOnehot,
  normalizePixels,
  sliceDataset,
} from "./mnist";

describe("normalizePixels", () => {
  it("reads the red channel of RGBA bytes and scales to [0,1]", () => {
    // two pixels: (255,_,_,_) -> 1, (0,_,_,_) -> 0
    const rgba = new Uint8Array([255, 10, 20, 255, 0, 5, 5, 255]);
    expect(Array.from(normalizePixels(rgba))).toEqual([1, 0]);
  });
});

describe("labelsFromOnehot", () => {
  it("collapses one-hot rows to integer classes", () => {
    // 3 rows, 10 classes each: labels 3, 0, 9
    const onehot = new Uint8Array(30);
    onehot[0 * 10 + 3] = 1;
    onehot[1 * 10 + 0] = 1;
    onehot[2 * 10 + 9] = 1;
    expect(Array.from(labelsFromOnehot(onehot, 3))).toEqual([3, 0, 9]);
  });
});

describe("sliceDataset", () => {
  it("splits into disjoint train/test blocks with correct shapes", () => {
    const count = 5;
    const images = new Float32Array(count * IMAGE_SIZE);
    // Tag each image by its first pixel so we can check the split boundaries.
    for (let i = 0; i < count; i++) images[i * IMAGE_SIZE] = i;
    const labels = Uint8Array.from([0, 1, 2, 3, 4]);

    const split = sliceDataset({ images, labels, count }, 3, 2);
    expect(split.xTrain.length).toBe(3 * IMAGE_SIZE);
    expect(split.xTest.length).toBe(2 * IMAGE_SIZE);
    expect(Array.from(split.yTrain)).toEqual([0, 1, 2]);
    expect(Array.from(split.yTest)).toEqual([3, 4]);
    // First test image is original index 3.
    expect(split.xTest[0]).toBe(3);
  });

  it("throws when the pool is too small", () => {
    const images = new Float32Array(2 * IMAGE_SIZE);
    const labels = new Uint8Array(2);
    expect(() => sliceDataset({ images, labels, count: 2 }, 2, 1)).toThrow(
      /only 2/,
    );
  });
});
