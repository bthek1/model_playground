import { useCallback, useEffect, useRef, useState } from "react";

import type { TensorOpJob, TensorOpResult } from "@/webgpu/types";
import {
  createWebGPUWorker,
  runTensorOpInWorker,
} from "@/webgpu/workerClient";

export interface TensorOpState {
  running: boolean;
  result: TensorOpResult | null;
  error: string | null;
}

/**
 * Runs basic matrix/tensor arithmetic on the WebGPU worker so the compute (and
 * buffer transfers) stay off the UI thread. The worker is created lazily and
 * torn down on unmount. `run` rejects into `error` for a bad shape/driver.
 */
export function useTensorOp() {
  const workerRef = useRef<Worker | null>(null);
  const [state, setState] = useState<TensorOpState>({
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

  const run = useCallback(async (job: TensorOpJob) => {
    setState({ running: true, result: null, error: null });
    try {
      workerRef.current ??= createWebGPUWorker();
      const result = await runTensorOpInWorker(workerRef.current, job);
      setState({ running: false, result, error: null });
    } catch (error) {
      setState({
        running: false,
        result: null,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }, []);

  const reset = useCallback(() => {
    setState({ running: false, result: null, error: null });
  }, []);

  return { ...state, run, reset };
}
