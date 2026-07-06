import { describe, expect, it } from "vitest";

import { isBinaryOp, runTensorOp, tensorOpSymbol } from "./tensorops";
import type { TensorOp } from "./types";

const ALL_OPS: TensorOp[] = [
  "add",
  "sub",
  "mul",
  "div",
  "matmul",
  "transpose",
  "scale",
];

describe("tensorOpSymbol", () => {
  it("maps every op to a non-empty symbol", () => {
    for (const op of ALL_OPS) expect(tensorOpSymbol(op)).toBeTruthy();
    expect(tensorOpSymbol("add")).toBe("+");
    expect(tensorOpSymbol("matmul")).toBe("·");
    expect(tensorOpSymbol("transpose")).toBe("ᵀ");
  });
});

describe("isBinaryOp", () => {
  it("is true for the two-operand ops", () => {
    for (const op of ["add", "sub", "mul", "div", "matmul"] as TensorOp[]) {
      expect(isBinaryOp(op)).toBe(true);
    }
  });

  it("is false for the unary ops", () => {
    for (const op of ["transpose", "scale"] as TensorOp[]) {
      expect(isBinaryOp(op)).toBe(false);
    }
  });
});

// happy-dom has no navigator.gpu, but every shape check in runTensorOp runs on
// the CPU *before* getGPUDevice() is touched — so these rejections are exercised
// without a real GPU.
describe("runTensorOp — shape validation before any GPU work", () => {
  it("rejects an element-wise op with no B operand", async () => {
    await expect(
      runTensorOp({ op: "add", a: new Float32Array([1]), aRows: 1, aCols: 1 }),
    ).rejects.toThrow(/second matrix B/);
  });

  it("rejects an element-wise op when A and B shapes differ", async () => {
    await expect(
      runTensorOp({
        op: "sub",
        a: new Float32Array([1, 2]),
        aRows: 1,
        aCols: 2,
        b: new Float32Array([1, 2, 3]),
        bRows: 1,
        bCols: 3,
      }),
    ).rejects.toThrow(/Shapes must match/);
  });

  it("rejects matmul when inner dimensions disagree", async () => {
    await expect(
      runTensorOp({
        op: "matmul",
        a: new Float32Array([1, 2, 3, 4, 5, 6]),
        aRows: 2,
        aCols: 3,
        b: new Float32Array([1, 2, 3, 4]),
        bRows: 2,
        bCols: 2,
      }),
    ).rejects.toThrow(/Inner dimensions must match/);
  });

  it("rejects matmul with no B operand", async () => {
    await expect(
      runTensorOp({
        op: "matmul",
        a: new Float32Array([1]),
        aRows: 1,
        aCols: 1,
      }),
    ).rejects.toThrow(/needs a second matrix B/);
  });

  it("rejects an unknown op", async () => {
    await expect(
      runTensorOp({
        op: "bogus" as TensorOp,
        a: new Float32Array([1]),
        aRows: 1,
        aCols: 1,
      }),
    ).rejects.toThrow(/Unsupported op/);
  });
});
