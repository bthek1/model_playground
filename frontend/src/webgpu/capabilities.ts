import type { AdapterInfo, WebGPUCapabilities } from "./types";

// Limits most relevant to compute workloads — shown in the playground so users
// can reason about the largest model / batch their GPU will accept.
const REPORTED_LIMITS = [
  "maxComputeWorkgroupsPerDimension",
  "maxComputeInvocationsPerWorkgroup",
  "maxComputeWorkgroupStorageSize",
  "maxStorageBufferBindingSize",
  "maxBufferSize",
] as const;

export function isWebGPUSupported(): boolean {
  return (
    typeof navigator !== "undefined" &&
    "gpu" in navigator &&
    Boolean(navigator.gpu)
  );
}

function toAdapterInfo(info: GPUAdapterInfo): AdapterInfo {
  return {
    vendor: info.vendor || "unknown",
    architecture: info.architecture || "unknown",
    device: info.device || "unknown",
    description: info.description || "",
  };
}

function readLimits(limits: GPUSupportedLimits): Record<string, number> {
  const out: Record<string, number> = {};
  for (const key of REPORTED_LIMITS) {
    const value = (limits as unknown as Record<string, number>)[key];
    if (typeof value === "number") out[key] = value;
  }
  return out;
}

/**
 * Probe the environment for WebGPU and return a serialisable capability report.
 * Safe to call anywhere (including a Web Worker); never throws.
 */
export async function detectWebGPU(): Promise<WebGPUCapabilities> {
  const empty = {
    adapter: null,
    isFallbackAdapter: false,
    features: [],
    limits: {},
  };

  if (!isWebGPUSupported()) {
    return { status: "unsupported", ...empty };
  }

  let adapter: GPUAdapter | null = null;
  try {
    adapter = await navigator.gpu.requestAdapter();
  } catch {
    adapter = null;
  }
  if (!adapter) {
    return { status: "no-adapter", ...empty };
  }

  const report = {
    adapter: toAdapterInfo(adapter.info),
    isFallbackAdapter: Boolean(
      (adapter as unknown as { isFallbackAdapter?: boolean }).isFallbackAdapter,
    ),
    features: [...adapter.features].sort(),
    limits: readLimits(adapter.limits),
  };

  // An adapter alone is not proof of access — actually acquiring a GPUDevice is
  // the real gate for running compute (the driver may be blocked or missing).
  // Discard this probe device; the runtime acquires its own via getGPUDevice().
  try {
    const device = await adapter.requestDevice();
    device.destroy();
  } catch {
    return { status: "no-device", ...report };
  }

  return { status: "ready", ...report };
}
