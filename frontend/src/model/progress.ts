// Aggregate load progress for the LOAD slot.
//
// Transformers.js reports progress **per file** (`initiate` → `progress`* →
// `done`, once for each of the 4–8 files a model pulls), and the raw payload was
// being rendered straight into the bar. That bar restarts at 0 for every file,
// so on a 220 MB ASR download it visibly runs forwards, snaps back, and then
// stalls at 100% through a warm-up that hasn't started yet.
//
// So we keep a table of files and report the aggregate *by bytes*. Pure and
// React-free on purpose — the awkward cases (a file with no `total`, an event
// arriving out of order, a percent that would go backwards) are exactly what
// unit tests are for. `useModelWorker` owns the state; this owns the arithmetic.
//
// Deliberately no ETA: a rate estimated from the first seconds of a multi-file
// download is wrong by a factor of several, and a wrong ETA is worse than none.
// Bytes and elapsed time are facts; extrapolation isn't.

import type { ModelProgress } from "./types";

export interface FileProgress {
  loaded: number;
  /** 0 when the server didn't say (no Content-Length) — such a file is excluded
   *  from the aggregate rather than guessed at. */
  total: number;
  done: boolean;
}

/**
 * The accumulated view of a load. `percent` is carried in the state rather than
 * derived on read because it is **monotonic**: a newly-announced file enlarges
 * the denominator, and without the clamp the bar would jump backwards.
 */
export interface ProgressState {
  files: Record<string, FileProgress>;
  phase: "connecting" | "downloading" | "warmup";
  /** Null while no file has announced a size — the bar is indeterminate. */
  percent: number | null;
  /** The file that moved most recently, for the detail line. */
  current?: string;
}

/** What the LOAD slot renders. */
export interface LoadProgress {
  phase: ProgressState["phase"];
  /** 0–100 across all files with a known size. Null → indeterminate. */
  percent: number | null;
  loaded: number;
  total: number;
  files: { done: number; count: number };
  current?: string;
  elapsedMs: number;
}

export const initialProgress: ProgressState = {
  files: {},
  phase: "connecting",
  percent: null,
};

/** Bytes downloaded / announced, over the files whose size is known. */
function totals(files: Record<string, FileProgress>) {
  let loaded = 0;
  let total = 0;
  for (const f of Object.values(files)) {
    if (f.total <= 0) continue;
    loaded += Math.min(f.loaded, f.total);
    total += f.total;
  }
  return { loaded, total };
}

/**
 * Bytes for one event. Transformers.js sends `loaded`/`total` on a `progress`,
 * but not always — some transports only give the percentage, in which case a
 * previously-announced total is the only way back to bytes.
 */
function bytesOf(e: ModelProgress, prev: FileProgress): FileProgress {
  const total = e.total && e.total > 0 ? e.total : prev.total;
  let loaded = prev.loaded;
  if (typeof e.loaded === "number") loaded = e.loaded;
  else if (typeof e.progress === "number" && total > 0)
    loaded = (total * e.progress) / 100;
  // Bytes never un-download: an out-of-order event must not rewind a file.
  return { loaded: Math.max(loaded, prev.loaded), total, done: prev.done };
}

/** Fold one worker progress event into the table. Returns a new state. */
export function reduceProgress(
  prev: ProgressState,
  e: ModelProgress,
): ProgressState {
  // Warm-up is a phase, not the tail of the download: the bytes are all in and
  // the first (throwaway) inference is compiling shaders / JITing the WASM.
  if (e.status === "warmup") {
    return { ...prev, phase: "warmup", current: undefined };
  }

  const key = e.file ?? e.name;
  if (!key) return prev;

  const entry = prev.files[key] ?? { loaded: 0, total: 0, done: false };
  let next: FileProgress;
  switch (e.status) {
    case "done":
      next = {
        total: entry.total,
        loaded: entry.total > 0 ? entry.total : entry.loaded,
        done: true,
      };
      break;
    case "initiate":
    case "download":
    case "progress":
    default:
      next = bytesOf(e, entry);
      if (next.total > 0 && next.loaded >= next.total) next.done = true;
      break;
  }

  const files = { ...prev.files, [key]: next };
  const { loaded, total } = totals(files);
  const percent =
    total > 0
      ? Math.max(prev.percent ?? 0, Math.min(100, Math.round((loaded / total) * 100)))
      : prev.percent;

  return {
    files,
    // Once warming up, a stray late file event must not drag us back to
    // "downloading" — the phase only moves forwards.
    phase: prev.phase === "warmup" ? "warmup" : loaded > 0 ? "downloading" : "connecting",
    percent,
    current: e.status === "done" ? prev.current : key,
  };
}

/** Flatten the table into the numbers the LOAD slot shows. */
export function summarize(s: ProgressState, elapsedMs: number): LoadProgress {
  const entries = Object.values(s.files);
  const { loaded, total } = totals(s.files);
  return {
    phase: s.phase,
    // Warm-up has no measurable progress, so it reports indeterminate rather
    // than sitting at a misleading 100%.
    percent: s.phase === "warmup" ? null : s.percent,
    loaded,
    total,
    files: { done: entries.filter((f) => f.done).length, count: entries.length },
    current: s.current,
    elapsedMs,
  };
}
