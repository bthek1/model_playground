import { describe, expect, it, vi } from "vitest";

import { cpuMatmul } from "@/webgpu/linearModel";

import { backward, forward, initNet, softmax } from "./policyNet";

describe("policyNet", () => {
  it("runs its arithmetic through the injected matmul, and agrees with the CPU reference", () => {
    const net = initNet({ inputs: 4, hidden: 5, outputs: 2 }, 1);
    const x = Float32Array.from([0.1, -0.3, 0.2, 0.05, 0.4, 0.4, -0.1, 0]);
    const stub = vi.fn(cpuMatmul);
    const viaStub = forward(net, x, 2, stub);
    const viaRef = forward(net, x, 2);
    expect(stub).toHaveBeenCalledTimes(2); // two layers, two matmuls
    expect(Array.from(viaStub.out)).toEqual(Array.from(viaRef.out));

    const d = Float32Array.from([0.3, -0.3, -0.1, 0.1]);
    stub.mockClear();
    const g1 = backward(net, viaStub, d, stub);
    const g2 = backward(net, viaRef, d);
    expect(stub).toHaveBeenCalledTimes(3); // dW2, dH, dW1
    expect(Array.from(g1.w1)).toEqual(Array.from(g2.w1));
  });

  it("uses the shared softmax: rows sum to 1 and a shift changes nothing", () => {
    const logits = Float32Array.from([1, 2, -3, 0.5]);
    const p = softmax(logits, 2, 2);
    expect(p[0] + p[1]).toBeCloseTo(1, 6);
    expect(p[2] + p[3]).toBeCloseTo(1, 6);
    const shifted = softmax(logits.map((v) => v + 100), 2, 2);
    for (let i = 0; i < 4; i++) expect(shifted[i]).toBeCloseTo(p[i], 6);
  });

  it("starts a policy head near uniform when its output layer is scaled down", () => {
    const net = initNet({ inputs: 4, hidden: 32, outputs: 2 }, 3, 0.1);
    const p = softmax(forward(net, Float32Array.from([0.04, 0.02, -0.03, 0.01]), 1).out, 1, 2);
    expect(Math.abs(p[0] - 0.5)).toBeLessThan(0.05);
  });
});
