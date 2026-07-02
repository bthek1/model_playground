import { useCallback, useEffect, useRef, useState } from "react";

import type { MatmulResult } from "@/webgpu/types";
import { createWebGPUWorker, runMatmulInWorker } from "@/webgpu/workerClient";

export interface BenchmarkState {
  running: boolean;
  result: (MatmulResult & { correct: boolean }) | null;
  error: string | null;
}

function randomMatrix(length: number): Float32Array {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) out[i] = Math.random() * 2 - 1;
  return out;
}

/**
 * Runs a square matmul on the GPU worker and reports throughput. Also does a
 * single-entry CPU cross-check so the UI can flag an incorrect kernel/driver.
 * The worker is created lazily and torn down on unmount.
 */
export function useGpuBenchmark(size = 512) {
  const workerRef = useRef<Worker | null>(null);
  const [state, setState] = useState<BenchmarkState>({
    running: false,
    result: null,
    error: null,
  });

  useEffect(() => {
    return () => {
      workerRef.current?.terminate();
      workerRef.current = null;
    };
  }, []);

  const run = useCallback(async () => {
    setState({ running: true, result: null, error: null });
    try {
      workerRef.current ??= createWebGPUWorker();
      const m = size;
      const k = size;
      const n = size;
      const a = randomMatrix(m * k);
      const b = randomMatrix(k * n);

      // CPU reference for C[0,0] = sum_i A[0,i] * B[i,0] — computed before the
      // input buffers are transferred to (and detached by) the worker.
      let reference = 0;
      for (let i = 0; i < k; i++) reference += a[i] * b[i * n];

      const result = await runMatmulInWorker(workerRef.current, {
        a,
        b,
        m,
        k,
        n,
      });
      const tolerance = 1e-2 * Math.max(1, Math.abs(reference));
      const correct = Math.abs(result.data[0] - reference) < tolerance;

      setState({ running: false, result: { ...result, correct }, error: null });
    } catch (error) {
      setState({
        running: false,
        result: null,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }, [size]);

  return { ...state, run };
}
