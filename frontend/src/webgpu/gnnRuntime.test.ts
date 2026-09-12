import { describe, expect, it } from "vitest";

import { GraphAggregator } from "./gnnRuntime";

// happy-dom has no navigator.gpu. Every check below runs on the CPU *before*
// getGPUDevice() is reached, so the rejections are exercised without a real GPU;
// the kernel's numeric agreement with the CPU reference is checked in a real
// browser by e2e/specs/webgpu/graph.spec.ts, which is the only place it can be.
describe("GraphAggregator.create — validation before any GPU work", () => {
  const rowPtr = Uint32Array.from([0, 1, 2]);
  const colIdx = Uint32Array.from([1, 0]);
  const scales = Float32Array.from([1, 1]);

  it("rejects a rowPtr that is not nNodes+1 long", async () => {
    await expect(
      GraphAggregator.create(Uint32Array.from([0, 1]), colIdx, 2, scales, scales),
    ).rejects.toThrow(/rowPtr must have nNodes\+1/);
  });

  it("rejects a rowPtr that disagrees with the edge count", async () => {
    await expect(
      GraphAggregator.create(rowPtr, Uint32Array.from([1]), 2, scales, scales),
    ).rejects.toThrow(/does not end at the edge count/);
  });

  it("rejects a scale vector of the wrong length", async () => {
    await expect(
      GraphAggregator.create(rowPtr, colIdx, 2, Float32Array.from([1]), scales),
    ).rejects.toThrow(/one entry per node/);
    await expect(
      GraphAggregator.create(rowPtr, colIdx, 2, scales, Float32Array.from([1])),
    ).rejects.toThrow(/one entry per node/);
  });
});
