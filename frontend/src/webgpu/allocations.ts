// How many bytes this app is holding on the GPU.
//
// WebGPU exposes no VRAM figure — not total, not free, not per-process — so the
// only honest number is the one we keep ourselves: bytes *we* allocated and have
// not destroyed. That makes this a ledger, not a probe, and the panel says so.
//
// The complication is realms. Every WGSL kernel runs in `webgpu/worker.ts`,
// which owns its own `GPUDevice` and its own module instances, so a counter kept
// on the page would read zero through an entire training run while the worker
// held half a gigabyte. The only main-thread allocation in the app is
// `PointRenderer`'s vertex buffer.
//
// So each realm keeps a local ledger and the page aggregates. The transport is a
// `MessagePort` handed to the worker by `createWebGPUWorker()` — not a
// `BroadcastChannel`, which is origin-wide and would fold a second tab's
// allocations into this page's total, and not the job protocol, which is about
// jobs. The port is also the liveness signal: a terminated worker simply stops
// publishing and is dropped after `STALE_MS`.
//
// Nothing is published unless someone is watching (`watchAllocations(true)`,
// called only while the system panel is open). A monitor must not be the thing
// it measures.

export interface AllocationSnapshot {
  liveBytes: number;
  /**
   * High-water mark. Aggregated across realms this is an upper bound — two
   * realms that peaked at different moments are still summed — which is the
   * right direction to err for a ceiling.
   */
  peakBytes: number;
  buffers: number;
  /** Last timed compute pass, where `timestamp-query` is available. */
  lastPassMs: number | null;
}

/** How often a publishing realm sends its snapshot while being watched. */
export const PUBLISH_MS = 1_000;
/** A realm silent for this long is gone (worker terminated, page navigated). */
export const STALE_MS = 3 * PUBLISH_MS;

// ---------------------------------------------------------------- local ledger

// Sizes are remembered per buffer so `releaseBuffer` needs no byte count from
// its caller — asking for one at every call site is how the two numbers drift.
// Weak so a buffer dropped without release cannot leak the entry.
const sizes = new WeakMap<GPUBuffer, number>();
let liveBytes = 0;
let peakBytes = 0;
let buffers = 0;
let lastPassMs: number | null = null;

/** Record a freshly created buffer. Returns it, so it can wrap a constructor. */
export function trackBuffer(buffer: GPUBuffer, byteLength: number): GPUBuffer {
  if (Number.isFinite(byteLength) && byteLength > 0) {
    sizes.set(buffer, byteLength);
    liveBytes += byteLength;
    buffers++;
    if (liveBytes > peakBytes) peakBytes = liveBytes;
  }
  return buffer;
}

/**
 * Destroy a tracked buffer and drop it from the ledger. Callers use this in
 * place of `buffer.destroy()`; a buffer that was never tracked is still
 * destroyed, so this is always safe to call.
 */
export function releaseBuffer(buffer: GPUBuffer): void {
  const size = sizes.get(buffer);
  if (size != null) {
    sizes.delete(buffer);
    liveBytes = Math.max(0, liveBytes - size);
    buffers = Math.max(0, buffers - 1);
  }
  buffer.destroy();
}

/**
 * Record the GPU-side duration of a compute pass. Our own WGSL passes only —
 * a Transformers.js / ONNX Runtime session is opaque to us and must never be
 * implied by this number.
 */
export function recordPassMs(ms: number): void {
  if (Number.isFinite(ms) && ms >= 0) lastPassMs = ms;
}

/** This realm's ledger. */
export function localSnapshot(): AllocationSnapshot {
  return { liveBytes, peakBytes, buffers, lastPassMs };
}

// ------------------------------------------------------------------- transport

interface PortState {
  port: MessagePort;
  /** Latest snapshot from the far side, with the time it arrived. */
  snapshot: AllocationSnapshot | null;
  at: number;
}

const remotes = new Set<PortState>();
let watching = false;

/** Page side: adopt the port belonging to a newly created WebGPU worker. */
export function registerAllocationPort(port: MessagePort): void {
  const state: PortState = { port, snapshot: null, at: 0 };
  remotes.add(state);
  port.onmessage = (event: MessageEvent<AllocationSnapshot>) => {
    state.snapshot = event.data;
    state.at = Date.now();
  };
  // Assigning `onmessage` starts the port implicitly.
  if (watching) port.postMessage({ watch: true });
}

/**
 * Page side: ask every realm to publish (or stop). Called when the system panel
 * opens and closes — while it is shut, no realm sends anything at all.
 */
export function watchAllocations(enabled: boolean): void {
  watching = enabled;
  for (const remote of remotes) remote.port.postMessage({ watch: enabled });
  if (!enabled) {
    for (const remote of remotes) {
      remote.snapshot = null;
      remote.at = 0;
    }
  }
}

/**
 * Page side: this realm plus every live worker realm. Realms that have gone
 * quiet are forgotten rather than shown as a stale total.
 */
export function aggregateSnapshot(now: number = Date.now()): AllocationSnapshot {
  const total = { ...localSnapshot() };
  for (const remote of remotes) {
    if (!remote.snapshot) continue;
    if (now - remote.at > STALE_MS) {
      remote.snapshot = null;
      continue;
    }
    total.liveBytes += remote.snapshot.liveBytes;
    total.peakBytes += remote.snapshot.peakBytes;
    total.buffers += remote.snapshot.buffers;
    if (remote.snapshot.lastPassMs != null) {
      total.lastPassMs = remote.snapshot.lastPassMs;
    }
  }
  return total;
}

let publishTimer: ReturnType<typeof setInterval> | null = null;

/**
 * Worker side: publish this realm's ledger over `port` while the page is
 * watching. The interval is the liveness signal too — see `STALE_MS`.
 */
export function attachAllocationPort(port: MessagePort): void {
  port.onmessage = (event: MessageEvent<{ watch?: boolean }>) => {
    if (event.data?.watch) {
      if (publishTimer != null) return;
      port.postMessage(localSnapshot());
      publishTimer = setInterval(
        () => port.postMessage(localSnapshot()),
        PUBLISH_MS,
      );
    } else if (publishTimer != null) {
      clearInterval(publishTimer);
      publishTimer = null;
    }
  };
}

/** Test seam: forget everything, in both roles. Not used by the app. */
export function resetAllocations(): void {
  liveBytes = 0;
  peakBytes = 0;
  buffers = 0;
  lastPassMs = null;
  remotes.clear();
  watching = false;
  if (publishTimer != null) {
    clearInterval(publishTimer);
    publishTimer = null;
  }
}
