// What the system panel is allowed to say.
//
// The panel's subject is the user's machine, and a web page can only see a
// sliver of it: there is no API for host CPU percent, none for GPU utilisation,
// and none for VRAM. Everything here therefore reports **load and capacity**,
// never utilisation, and every metric is explicit about which of the two it is.
//
// The type that enforces it is `Metric<T>`. A sampler either has a real
// measurement or it has a reason it hasn't — there is no third state, and in
// particular no zero standing in for "unknown". That matters more here than in
// most modules: a memory card reading "0 MB" in Firefox looks like a working
// card reporting an idle tab, and the user would believe it.

/** A measurement that exists. */
export interface MetricOk<T> {
  status: "ok";
  value: T;
}

/**
 * A measurement this browser cannot make. `reason` is user-facing copy, not a
 * log line — it is rendered in place of the number.
 */
export interface MetricUnavailable {
  status: "unavailable";
  reason: string;
}

export type Metric<T> = MetricOk<T> | MetricUnavailable;

export function ok<T>(value: T): MetricOk<T> {
  return { status: "ok", value };
}

export function unavailable(reason: string): MetricUnavailable {
  return { status: "unavailable", reason };
}

export function isOk<T>(metric: Metric<T>): metric is MetricOk<T> {
  return metric.status === "ok";
}

/**
 * JS heap of the **calling realm only** — the page, never the workers where
 * inference actually allocates. Chrome-only and quantised. Indicative.
 */
export interface MemorySample {
  usedBytes: number;
  totalBytes: number;
  /** The heap ceiling this realm will be allowed to reach. */
  limitBytes: number;
}

/**
 * Main-thread responsiveness, which is the only CPU signal a page gets. `lagMs`
 * is how far the last animation frame overran its budget; a busy *worker* shows
 * up in `busyFraction`, not here, because a worker that never touches the DOM
 * leaves the main thread perfectly responsive.
 */
export interface CpuSample {
  lagMs: number;
  /** Null where the browser has no long-task entry type — not zero. */
  longTasks: number | null;
  longTaskMs: number | null;
  /** Share of the last interval with at least one inference in flight, 0–1. */
  busyFraction: number;
}

/** Origin storage — where Transformers.js keeps downloaded weights. */
export interface StorageSample {
  usageBytes: number;
  quotaBytes: number;
  /** Models with at least one file in the `transformers-cache` bucket. */
  cachedModels: number | null;
}

/** Bytes this page's own WGSL runtime is holding on the device. */
export interface GpuMemorySample {
  liveBytes: number;
  peakBytes: number;
  buffers: number;
  /** Duration of the last timed compute pass, where `timestamp-query` exists. */
  lastPassMs: number | null;
}

/** An in-flight weight download, as `model/progress.ts` already aggregates it. */
export interface DownloadSample {
  loadedBytes: number;
  totalBytes: number;
  /** Bytes per second over the sampling interval, or null on the first sample. */
  bytesPerSecond: number | null;
}

export interface StaticCapacity {
  /** `navigator.hardwareConcurrency` — logical cores, not free cores. */
  cores: Metric<number>;
  /** `navigator.deviceMemory` — a coarse GiB class, quantised by the browser. */
  deviceMemoryGiB: Metric<number>;
}
