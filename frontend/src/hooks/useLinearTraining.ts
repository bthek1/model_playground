import { useCallback, useEffect, useRef, useState } from "react";

import {
  IMAGE_SIZE,
  loadMnistPool,
  type MnistPool,
  NUM_CLASSES,
  sliceDataset,
} from "@/lib/mnist";
import type { TrainMetrics, TrainResult } from "@/webgpu/linearModel";
import {
  createWebGPUWorker,
  trainLinearInWorker,
  type TrainingHandle,
  type WeightSnapshot,
} from "@/webgpu/workerClient";

export type DatasetStatus = "idle" | "loading" | "ready" | "error";

export interface TrainingSettings {
  learningRate: number;
  batchSize: number;
  epochs: number;
  trainSize: number;
  testSize: number;
}

/**
 * Owns the MNIST dataset, the compute worker, and a streaming linear-model
 * training run. Metric updates are batched to animation frames so a fast worker
 * can't flood React with renders.
 */
export function useLinearTraining() {
  const workerRef = useRef<Worker | null>(null);
  const poolRef = useRef<MnistPool | null>(null);
  const handleRef = useRef<TrainingHandle | null>(null);

  const [datasetStatus, setDatasetStatus] = useState<DatasetStatus>("idle");
  const [datasetProgress, setDatasetProgress] = useState(0);
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const [poolCount, setPoolCount] = useState(0);

  const [training, setTraining] = useState(false);
  const [metrics, setMetrics] = useState<TrainMetrics[]>([]);
  const [result, setResult] = useState<TrainResult | null>(null);
  const [snapshot, setSnapshot] = useState<WeightSnapshot | null>(null);
  const [trainError, setTrainError] = useState<string | null>(null);

  // rAF-batched metric flushing.
  const metricsRef = useRef<TrainMetrics[]>([]);
  const flushScheduled = useRef(false);
  const scheduleFlush = useCallback(() => {
    if (flushScheduled.current) return;
    flushScheduled.current = true;
    requestAnimationFrame(() => {
      flushScheduled.current = false;
      setMetrics(metricsRef.current.slice());
    });
  }, []);

  useEffect(() => {
    return () => {
      workerRef.current?.terminate();
      workerRef.current = null;
    };
  }, []);

  const loadData = useCallback(async (count: number) => {
    setDatasetStatus("loading");
    setDatasetError(null);
    setDatasetProgress(0);
    try {
      const pool = await loadMnistPool(count, setDatasetProgress);
      poolRef.current = pool;
      setPoolCount(pool.count);
      setDatasetStatus("ready");
    } catch (error) {
      setDatasetError(error instanceof Error ? error.message : String(error));
      setDatasetStatus("error");
    }
  }, []);

  const start = useCallback(
    (settings: TrainingSettings) => {
      const pool = poolRef.current;
      if (!pool) {
        setTrainError("Load the MNIST dataset first.");
        return;
      }
      let data;
      try {
        data = sliceDataset(pool, settings.trainSize, settings.testSize);
      } catch (error) {
        setTrainError(error instanceof Error ? error.message : String(error));
        return;
      }

      metricsRef.current = [];
      setMetrics([]);
      setResult(null);
      setSnapshot(null);
      setTrainError(null);
      setTraining(true);

      workerRef.current ??= createWebGPUWorker();
      const handle = trainLinearInWorker(
        workerRef.current,
        {
          shape: { inputDim: IMAGE_SIZE, numClasses: NUM_CLASSES },
          hp: {
            learningRate: settings.learningRate,
            batchSize: settings.batchSize,
            epochs: settings.epochs,
          },
          data,
        },
        (m) => {
          metricsRef.current.push(m);
          scheduleFlush();
        },
        (snap) => setSnapshot(snap),
      );
      handleRef.current = handle;

      handle.promise
        .then((res) => {
          setResult(res);
          setSnapshot({ epoch: -1, weights: res.weights, bias: res.bias });
          setMetrics(metricsRef.current.slice());
        })
        .catch((error: unknown) => {
          setTrainError(error instanceof Error ? error.message : String(error));
        })
        .finally(() => {
          setTraining(false);
          handleRef.current = null;
        });
    },
    [scheduleFlush],
  );

  const stop = useCallback(() => {
    handleRef.current?.cancel();
  }, []);

  return {
    // dataset
    datasetStatus,
    datasetProgress,
    datasetError,
    poolCount,
    loadData,
    // training
    training,
    metrics,
    result,
    snapshot,
    trainError,
    start,
    stop,
  };
}
