// The fit worker. A thin wrapper around `fitEngine.ts`, which holds all the
// logic and is unit-tested there against a CPU matmul — the same split as
// `text/pipeline.worker.ts` and `vision/vision.worker.ts`.
//
// Two things happen here and nowhere else: the real WGSL matmul is wired in,
// and `self` is narrowed to what we use. (No `/// <reference lib="webworker" />`
// — it collides with the app's DOM lib, as the other workers note.)

import { cpuMatmul } from "@/webgpu/linearModel";
import { runMatmul } from "@/webgpu/runtime";

import { createFitHandler } from "./fitEngine";
import type { TabularRequest, TabularResponse } from "./types";

const ctx = self as unknown as {
  onmessage: ((event: MessageEvent<TabularRequest>) => void) | null;
  postMessage: (message: TabularResponse) => void;
};

const handle = createFitHandler((message) => ctx.postMessage(message), {
  // Resolved per fit rather than memoised here: `getGPUDevice()` inside
  // `runMatmul` already memoises the device for this realm, and a fit that
  // wants the CPU must not pay a device probe to find that out.
  gpuMatmul: async () => {
    // One tiny dispatch, which both proves a device can be acquired and
    // compiles the matmul shader before the first real step needs it.
    await runMatmul({ a: new Float32Array([1]), b: new Float32Array([1]), m: 1, k: 1, n: 1 });
    return async (a, b, m, k, n) => (await runMatmul({ a, b, m, k, n })).data;
  },
  cpuMatmul: async (a, b, m, k, n) => cpuMatmul(a, b, m, k, n),
});

ctx.onmessage = (event) => {
  void handle(event.data);
};
