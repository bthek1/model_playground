// CPU, as a page experiences it.
//
// No browser reports host CPU utilisation, and it would be the wrong number
// anyway: what matters here is whether the *app* is still responsive while a
// model works. So this measures three things and calls them what they are:
//
//   lag           how far the last animation frames overran their budget —
//                 main-thread contention, which is what a user feels
//   long tasks    the count and duration of >50 ms blocks, when the browser
//                 implements the entry type (Chromium does; Safari/Firefox don't)
//   busy fraction share of the interval with an inference in flight, reported by
//                 `telemetry/activity.ts` from the worker hooks themselves
//
// A worker pinning four cores shows up in the busy fraction and *not* in lag,
// which is the honest answer: the page really is responsive, the machine really
// is working.

import { ok, unavailable, type Metric } from "../types";

/** Assumed frame budget. A 120 Hz display simply reports less lag, never more. */
export const FRAME_BUDGET_MS = 1000 / 60;

/**
 * A gap longer than this isn't contention, it's a suspended or throttled tab
 * (the sampler stops when the document hides, but a frame can straddle that).
 * Counting it as lag would put a 4-second spike on the chart for going away.
 */
export const STALL_MS = 5_000;

export interface LagTracker {
  /** Feed it `requestAnimationFrame`'s timestamp. */
  frame: (timestampMs: number) => void;
  /** Worst overrun since the last call, in ms, then reset. */
  consumeMaxLagMs: () => number;
  /** Frames seen since the last consume — zero means no basis to report lag. */
  frames: () => number;
}

export function createLagTracker(frameBudgetMs = FRAME_BUDGET_MS): LagTracker {
  let previous: number | null = null;
  let worst = 0;
  let frames = 0;

  return {
    frame(timestampMs) {
      if (previous != null) {
        const gap = timestampMs - previous;
        if (gap >= 0 && gap < STALL_MS) {
          frames++;
          const lag = Math.max(0, gap - frameBudgetMs);
          if (lag > worst) worst = lag;
        }
      }
      previous = timestampMs;
    },
    consumeMaxLagMs() {
      const value = worst;
      worst = 0;
      frames = 0;
      return value;
    },
    frames: () => frames,
  };
}

export interface LongTaskWatcher {
  /** Long tasks since the last call, then reset. `null` when unsupported. */
  consume: () => { count: number; totalMs: number } | null;
  stop: () => void;
  /** Null when the watcher is live; user-facing copy when it isn't. */
  unavailableReason: string | null;
}

const NO_LONG_TASKS =
  "Long-task timing isn't implemented in this browser — Chromium only.";

/**
 * Watch for >50 ms main-thread blocks. Starts observing immediately, so create
 * it when the panel opens and `stop()` it when the panel closes.
 */
export function createLongTaskWatcher(): LongTaskWatcher {
  const idle: LongTaskWatcher = {
    consume: () => null,
    stop: () => {},
    unavailableReason: NO_LONG_TASKS,
  };

  if (typeof PerformanceObserver === "undefined") {
    return { ...idle, unavailableReason: "This browser has no PerformanceObserver." };
  }
  const supported = PerformanceObserver.supportedEntryTypes;
  if (!supported || !supported.includes("longtask")) return idle;

  let count = 0;
  let totalMs = 0;
  let observer: PerformanceObserver;
  try {
    observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        count++;
        totalMs += entry.duration;
      }
    });
    observer.observe({ type: "longtask", buffered: false });
  } catch {
    // Supported-but-refused (an iframe policy, a hardened build): same outcome.
    return idle;
  }

  return {
    consume() {
      const snapshot = { count, totalMs };
      count = 0;
      totalMs = 0;
      return snapshot;
    },
    stop() {
      observer.disconnect();
    },
    unavailableReason: null,
  };
}

/** Logical cores. Capacity, not availability — nothing says how many are free. */
export function cores(): Metric<number> {
  if (typeof navigator === "undefined") {
    return unavailable("No navigator in this environment.");
  }
  const count = navigator.hardwareConcurrency;
  if (typeof count !== "number" || !Number.isFinite(count) || count <= 0) {
    return unavailable("This browser doesn't report a core count.");
  }
  return ok(count);
}

/** Busy milliseconds over an interval, as a 0–1 fraction. */
export function busyFraction(busyMs: number, intervalMs: number): number {
  if (!Number.isFinite(busyMs) || !Number.isFinite(intervalMs) || intervalMs <= 0) {
    return 0;
  }
  return Math.min(1, Math.max(0, busyMs / intervalMs));
}
