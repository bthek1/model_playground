// The sampling loop, and the only thing in the panel that costs anything.
//
// Two rules shape it, both from the plan:
//
//  1. A monitor must not be the thing it measures. Nothing runs unless the panel
//     is open *and* the document is visible. One interval at 1 Hz drives every
//     metric; the only per-frame work is reading `requestAnimationFrame`'s
//     timestamp, which is how main-thread lag is measured at all. The expensive
//     sample — walking the model cache — runs once every `STORAGE_EVERY_TICKS`.
//
//  2. Samples are not application state. They live in ring buffers behind refs
//     and reach React through exactly one `setState` per tick. In Zustand they
//     would re-render the app to move a sparkline; in TanStack Query they would
//     be pretending to be server state.
//
// Closing the panel clears the history rather than pausing it: a chart drawn
// across a ten-minute gap is a lie about continuity, and there is no honest way
// to draw the gap in a sparkline.

import { useCallback, useEffect, useRef, useState } from "react";

import { watchAllocations } from "@/webgpu/allocations";
import type { WebGPUCapabilities } from "@/webgpu/types";

import { activeDownload, consumeBusyMs } from "./activity";
import {
  busyFraction,
  cores,
  createLagTracker,
  createLongTaskWatcher,
} from "./samplers/cpu";
import { gpuIdentity, sampleGpuMemory } from "./samplers/gpu";
import { deviceMemoryGiB, sampleMemory } from "./samplers/memory";
import { sampleStorage } from "./samplers/storage";
import { createSeries } from "./series";
import type {
  CpuSample,
  DownloadSample,
  GpuMemorySample,
  MemorySample,
  Metric,
  StaticCapacity,
  StorageSample,
} from "./types";

export const DEFAULT_INTERVAL_MS = 1_000;

/**
 * The storage estimate walks every key in the Cache Storage bucket to count
 * models. At 1 Hz that would be the panel's own biggest cost, and the number it
 * produces changes only when a model finishes downloading.
 */
export const STORAGE_EVERY_TICKS = 10;

export interface TelemetryView {
  /** Samples taken since the panel opened. Zero means the loop is not running. */
  tick: number;
  intervalMs: number;
  memory: Metric<MemorySample>;
  /** Used JS heap bytes, oldest → newest. */
  memoryHistory: number[];
  cpu: CpuSample;
  lagHistory: number[];
  busyHistory: number[];
  /** Why long-task counts are missing, or null when they aren't. */
  longTaskReason: string | null;
  gpu: Metric<GpuMemorySample> | null;
  gpuHistory: number[];
  identity: Metric<WebGPUCapabilities> | null;
  storage: Metric<StorageSample> | null;
  download: DownloadSample | null;
  downloadHistory: number[];
  capacity: StaticCapacity;
}

function idleView(intervalMs: number): TelemetryView {
  return {
    tick: 0,
    intervalMs,
    memory: { status: "unavailable", reason: "Not sampling." },
    memoryHistory: [],
    cpu: { lagMs: 0, longTasks: null, longTaskMs: null, busyFraction: 0 },
    lagHistory: [],
    busyHistory: [],
    longTaskReason: null,
    gpu: null,
    gpuHistory: [],
    identity: null,
    storage: null,
    download: null,
    downloadHistory: [],
    capacity: { cores: cores(), deviceMemoryGiB: deviceMemoryGiB() },
  };
}

function visible(): boolean {
  return typeof document === "undefined" || !document.hidden;
}

/**
 * Sample the machine while `active`. Returns one immutable view per tick; the
 * caller re-renders on it and nothing else.
 */
export function useTelemetry(
  active: boolean,
  intervalMs: number = DEFAULT_INTERVAL_MS,
): TelemetryView {
  const [view, setView] = useState<TelemetryView>(() => idleView(intervalMs));

  // Created once and reused, so opening the panel twice doesn't allocate five
  // more buffers. Cleared at the start of every run — see the header note.
  const series = useRef({
    memory: createSeries<number>(),
    lag: createSeries<number>(),
    busy: createSeries<number>(),
    gpu: createSeries<number>(),
    download: createSeries<number>(),
  });

  const [awake, setAwake] = useState(() => active && visible());

  // The document can hide and re-show without the panel closing; a hidden tab
  // throttles timers and rAF to the point where every sample is a lie about the
  // machine, so the loop stops rather than recording noise.
  useEffect(() => {
    if (!active || typeof document === "undefined") {
      setAwake(active && visible());
      return;
    }
    const onVisibility = () => setAwake(visible());
    onVisibility();
    document.addEventListener("visibilitychange", onVisibility);
    return () => document.removeEventListener("visibilitychange", onVisibility);
  }, [active]);

  const reset = useCallback(() => {
    for (const s of Object.values(series.current)) s.clear();
  }, []);

  useEffect(() => {
    if (!awake) {
      setView(idleView(intervalMs));
      return;
    }

    reset();
    watchAllocations(true);

    const buffers = series.current;
    const lag = createLagTracker();
    const longTasks = createLongTaskWatcher();
    const capacity: StaticCapacity = {
      cores: cores(),
      deviceMemoryGiB: deviceMemoryGiB(),
    };

    let stopped = false;
    let tick = 0;
    // A tick is skipped rather than queued if the previous one is still in
    // flight (the storage walk is the only sampler slow enough to matter). A
    // monitor that queues work behind itself eventually reports the past.
    let sampling = false;
    let identity: Metric<WebGPUCapabilities> | null = null;
    let storage: Metric<StorageSample> | null = null;
    let previousDownload: { loadedBytes: number; at: number } | null = null;

    void gpuIdentity().then((result) => {
      if (!stopped) identity = result;
    });

    // One rAF chain, purely to time frames. It reschedules itself rather than
    // being driven by the interval, because the gap between frames *is* the
    // measurement.
    let frame = 0;
    const onFrame = (timestamp: number) => {
      if (stopped) return;
      lag.frame(timestamp);
      frame = requestAnimationFrame(onFrame);
    };
    frame = requestAnimationFrame(onFrame);

    const sample = async () => {
      if (sampling) return;
      sampling = true;
      const now = Date.now();
      // Captured, not read back: every `await` below can let later ticks
      // increment the counter, and `tick === 1` must mean *this* sample.
      const current = ++tick;

      const memory = sampleMemory();
      if (memory.status === "ok") buffers.memory.push(memory.value.usedBytes);

      const blocked = longTasks.consume();
      const cpu: CpuSample = {
        lagMs: lag.consumeMaxLagMs(),
        longTasks: blocked?.count ?? null,
        longTaskMs: blocked?.totalMs ?? null,
        busyFraction: busyFraction(consumeBusyMs(now), intervalMs),
      };
      buffers.lag.push(cpu.lagMs);
      buffers.busy.push(cpu.busyFraction);

      const gpu = await sampleGpuMemory(now);
      if (stopped) {
        sampling = false;
        return;
      }
      if (gpu.status === "ok") buffers.gpu.push(gpu.value.liveBytes);

      // The expensive one. `tick === 1` so the card is populated immediately
      // instead of blank for the first ten seconds.
      if (current === 1 || current % STORAGE_EVERY_TICKS === 0) {
        storage = await sampleStorage({ countModels: true });
        if (stopped) {
          sampling = false;
          return;
        }
      }

      // Rate comes from differencing `model/progress.ts`'s byte aggregate, not
      // from PerformanceObserver: the Hub's responses are cross-origin without
      // `Timing-Allow-Origin`, so resource timings report a transfer size of 0.
      const inFlight = activeDownload();
      let download: DownloadSample | null = null;
      if (inFlight) {
        const elapsed = previousDownload ? (now - previousDownload.at) / 1000 : 0;
        const delta = previousDownload
          ? inFlight.loadedBytes - previousDownload.loadedBytes
          : 0;
        download = {
          loadedBytes: inFlight.loadedBytes,
          totalBytes: inFlight.totalBytes,
          bytesPerSecond: elapsed > 0 && delta >= 0 ? delta / elapsed : null,
        };
        previousDownload = { loadedBytes: inFlight.loadedBytes, at: now };
        buffers.download.push(download.bytesPerSecond ?? 0);
      } else {
        previousDownload = null;
      }

      setView({
        tick: current,
        intervalMs,
        memory,
        memoryHistory: buffers.memory.toArray(),
        cpu,
        lagHistory: buffers.lag.toArray(),
        busyHistory: buffers.busy.toArray(),
        longTaskReason: longTasks.unavailableReason,
        gpu,
        gpuHistory: buffers.gpu.toArray(),
        identity,
        storage,
        download,
        downloadHistory: buffers.download.toArray(),
        capacity,
      });
      sampling = false;
    };

    void sample();
    const timer = setInterval(() => void sample(), intervalMs);

    return () => {
      stopped = true;
      clearInterval(timer);
      cancelAnimationFrame(frame);
      longTasks.stop();
      watchAllocations(false);
    };
  }, [awake, intervalMs, reset]);

  return view;
}
