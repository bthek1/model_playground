// Shared types for the raw-WebGPU runtime. These describe capabilities we
// surface to the UI and the shape of compute jobs the worker accepts.

export type WebGPUStatus = "unsupported" | "no-adapter" | "ready";

export interface AdapterInfo {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
}

export interface WebGPUCapabilities {
  status: WebGPUStatus;
  adapter: AdapterInfo | null;
  isFallbackAdapter: boolean;
  /** Optional GPU features the adapter advertises (e.g. "shader-f16"). */
  features: string[];
  /** Subset of GPUSupportedLimits relevant to compute workloads. */
  limits: Record<string, number>;
}

/** Inputs for a single matrix multiply C(MxN) = A(MxK) * B(KxN). */
export interface MatmulJob {
  a: Float32Array;
  b: Float32Array;
  m: number;
  k: number;
  n: number;
}

export interface MatmulResult {
  data: Float32Array;
  /** Wall-clock time for encode → submit → readback, in milliseconds. */
  gpuTimeMs: number;
  /** Throughput estimate: 2*M*K*N / time. */
  gflops: number;
}
