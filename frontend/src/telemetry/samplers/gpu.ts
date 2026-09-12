// The GPU: what it is, and what we are holding on it.
//
// Two halves, and neither is utilisation:
//
//   identity   adapter vendor/architecture/limits/features, from
//              `webgpu/capabilities.ts`. Probed once — it cannot change without
//              a device loss, and `requestAdapter()` is not free. This is the
//              same probe the home page's capability card uses; a second one
//              would be a second answer to the same question.
//
//   memory     the ledger in `webgpu/allocations.ts` — bytes this app allocated
//              and has not released, aggregated across the page and every live
//              WebGPU worker. Not VRAM: the browser exposes no VRAM figure at
//              all, and ONNX Runtime's own GPU buffers are invisible to us.
//
// A zero here is a real measurement (we have allocated nothing), which is why
// this sampler returns `ok(0)` rather than `unavailable` on an idle GPU.

import { detectWebGPU } from "@/webgpu/capabilities";
import { aggregateSnapshot, type AllocationSnapshot } from "@/webgpu/allocations";
import type { WebGPUCapabilities } from "@/webgpu/types";

import { ok, unavailable, type GpuMemorySample, type Metric } from "../types";

let identityPromise: Promise<Metric<WebGPUCapabilities>> | null = null;

const NO_GPU: Record<string, string> = {
  unsupported:
    "This browser doesn't expose WebGPU, so nothing about the GPU is readable.",
  "no-adapter": "WebGPU is present but offered no adapter — no compatible GPU or driver.",
  "no-device": "An adapter was found but no GPU device could be acquired.",
};

/**
 * Adapter identity, probed once per page. `detectWebGPU` never throws, so the
 * only failure this has to model is "there is no usable GPU".
 */
export function gpuIdentity(): Promise<Metric<WebGPUCapabilities>> {
  identityPromise ??= detectWebGPU().then((caps) =>
    caps.status === "ready" ? ok(caps) : unavailable(NO_GPU[caps.status]),
  );
  return identityPromise;
}

/** Bytes we are holding on the device. Requires a GPU to mean anything. */
export async function sampleGpuMemory(
  now?: number,
): Promise<Metric<GpuMemorySample>> {
  const identity = await gpuIdentity();
  if (identity.status === "unavailable") return identity;

  const snapshot: AllocationSnapshot = aggregateSnapshot(now);
  return ok({
    liveBytes: snapshot.liveBytes,
    peakBytes: snapshot.peakBytes,
    buffers: snapshot.buffers,
    lastPassMs: snapshot.lastPassMs,
  });
}

/** Test seam: drop the memoised probe. Not used by the app. */
export function resetGpuIdentity(): void {
  identityPromise = null;
}
