// Shared types for the raw-WebGPU runtime. These describe capabilities we
// surface to the UI and the shape of compute jobs the worker accepts.

export type WebGPUStatus =
  | "unsupported"
  | "no-adapter"
  | "no-device"
  | "ready";

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

/** Basic matrix/tensor operations the playground can dispatch to the GPU. */
export type TensorOp =
  | "add" // element-wise A + B (same shape)
  | "sub" // element-wise A − B (same shape)
  | "mul" // element-wise (Hadamard) A ⊙ B (same shape)
  | "div" // element-wise A ÷ B (same shape)
  | "matmul" // matrix product A(MxK) · B(KxN)
  | "transpose" // Aᵀ (unary)
  | "scale"; // scalar · A (unary)

/**
 * A single tensor-arithmetic job. Operands are stored row-major and flattened;
 * `b` and `scalar` are only required for the operations that consume them.
 */
export interface TensorOpJob {
  op: TensorOp;
  a: Float32Array;
  aRows: number;
  aCols: number;
  b?: Float32Array;
  bRows?: number;
  bCols?: number;
  scalar?: number;
}

export interface TensorOpResult {
  data: Float32Array;
  rows: number;
  cols: number;
  /** Wall-clock time for encode → submit → readback, in milliseconds. */
  gpuTimeMs: number;
}
