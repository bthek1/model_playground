// What the app itself is doing, reported by the code that already knows.
//
// Two of the panel's numbers cannot be observed from outside: whether an
// inference is in flight, and how fast a weight download is moving.
// `model/useModelWorker.ts` already tracks both — the in-flight *count* of
// machine B and the byte aggregate from `model/progress.ts` — so this module is
// a one-way bridge from there to the panel rather than a second measurement.
//
// Deliberately not Zustand: the panel is one subscriber, these values change on
// every run and every progress event, and pushing them through the global store
// would re-render the app to move a sparkline. Deliberately not a worker message
// either — the `ModelRequest`/`ModelResponse` envelope carries tasks, and a
// telemetry variant on it would be seen by six workers to serve one panel.
//
// Keyed by worker key, because a route can hold two hooks (`/pose` holds two
// models) and "busy" means *any* of them is busy.

interface DownloadReport {
  loadedBytes: number;
  totalBytes: number;
}

const inflight = new Map<string, number>();
const downloads = new Map<string, DownloadReport>();

// Busy time is accumulated rather than sampled, so a 50 ms inference between
// two ticks still counts. `busySince` is the start of the currently-open busy
// interval; `busyAccumMs` is everything already closed.
let busySince: number | null = null;
let busyAccumMs = 0;

function anyInflight(): boolean {
  for (const n of inflight.values()) if (n > 0) return true;
  return false;
}

/**
 * Report the in-flight request count for one worker. Called from an effect in
 * `useModelWorker`, including with `0` on teardown — a route unmounting mid-run
 * must not leave the panel reading "busy" forever.
 */
export function reportInflight(
  key: string,
  count: number,
  now: number = Date.now(),
): void {
  if (count > 0) inflight.set(key, count);
  else inflight.delete(key);

  const busy = anyInflight();
  if (busy && busySince === null) {
    busySince = now;
  } else if (!busy && busySince !== null) {
    busyAccumMs += now - busySince;
    busySince = null;
  }
}

/**
 * Busy milliseconds since the last call, and reset. The currently-open interval
 * is included up to `now` and then re-opened, so a long-running inference
 * contributes to every tick it spans instead of only the one it ends in.
 */
export function consumeBusyMs(now: number = Date.now()): number {
  let total = busyAccumMs;
  busyAccumMs = 0;
  if (busySince !== null) {
    total += now - busySince;
    busySince = now;
  }
  return total;
}

/** Report aggregate download bytes for one worker, or `null` when it finishes. */
export function reportDownload(
  key: string,
  report: DownloadReport | null,
): void {
  if (report && report.totalBytes > 0) downloads.set(key, report);
  else downloads.delete(key);
}

/**
 * The sum of every download in flight, or `null` when nothing is downloading.
 * Summed rather than picked, because `/pose` pulls two checkpoints at once and
 * either one alone would understate what the link is doing.
 */
export function activeDownload(): DownloadReport | null {
  if (downloads.size === 0) return null;
  let loadedBytes = 0;
  let totalBytes = 0;
  for (const d of downloads.values()) {
    loadedBytes += d.loadedBytes;
    totalBytes += d.totalBytes;
  }
  return { loadedBytes, totalBytes };
}

/** Test seam: forget everything. Not used by the app. */
export function resetActivity(): void {
  inflight.clear();
  downloads.clear();
  busySince = null;
  busyAccumMs = 0;
}
