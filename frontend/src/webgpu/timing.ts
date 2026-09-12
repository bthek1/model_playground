// GPU-side timing for our own compute passes.
//
// `performance.now()` around a submit measures encode + submit + readback on the
// CPU — useful, and what `MatmulResult.gpuTimeMs` already reports — but it is
// not how long the GPU spent. `timestamp-query` is: the device writes its own
// clock into a query set at the start and end of the pass, and the difference is
// the pass duration in nanoseconds.
//
// It is an optional feature (requested in `device.ts` only when the adapter
// advertises it), so every entry point here degrades to "no timing" rather than
// failing a job. And it times *our* WGSL only — an ONNX Runtime session
// dispatches its own passes and is entirely opaque to us, which is why the
// panel labels this number rather than calling it "GPU time".

import { recordPassMs } from "./allocations";

const QUERY_COUNT = 2; // start, end
const RESULT_BYTES = QUERY_COUNT * 8; // two uint64 timestamps

export interface PassTimer {
  /** Spread into `beginComputePass`'s descriptor. */
  timestampWrites: GPUComputePassTimestampWrites;
  /** Queue the resolve + copy. Call after `pass.end()`, before `submit`. */
  resolve: (encoder: GPUCommandEncoder) => void;
  /**
   * Read the timestamps back, record the duration in the ledger and free the
   * GPU objects. Resolves to null if anything went wrong — a timing failure
   * must never fail the job it was measuring.
   */
  readMs: () => Promise<number | null>;
}

/** Null when this device can't time a pass. Callers treat that as normal. */
export function createPassTimer(device: GPUDevice): PassTimer | null {
  if (!device.features?.has?.("timestamp-query")) return null;
  if (typeof device.createQuerySet !== "function") return null;

  let querySet: GPUQuerySet;
  let resolveBuffer: GPUBuffer;
  let readBuffer: GPUBuffer;
  try {
    querySet = device.createQuerySet({ type: "timestamp", count: QUERY_COUNT });
    resolveBuffer = device.createBuffer({
      size: RESULT_BYTES,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    readBuffer = device.createBuffer({
      size: RESULT_BYTES,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  } catch {
    return null;
  }

  // Not counted in the allocation ledger: 32 bytes of instrumentation would be
  // noise in a figure whose whole job is to show what the *workload* costs.
  const free = () => {
    querySet.destroy();
    resolveBuffer.destroy();
    readBuffer.destroy();
  };

  return {
    timestampWrites: {
      querySet,
      beginningOfPassWriteIndex: 0,
      endOfPassWriteIndex: 1,
    },

    resolve(encoder) {
      encoder.resolveQuerySet(querySet, 0, QUERY_COUNT, resolveBuffer, 0);
      encoder.copyBufferToBuffer(resolveBuffer, 0, readBuffer, 0, RESULT_BYTES);
    },

    async readMs() {
      try {
        await readBuffer.mapAsync(GPUMapMode.READ);
        const [start, end] = new BigUint64Array(
          readBuffer.getMappedRange().slice(0),
        );
        readBuffer.unmap();
        // A zero or inverted delta means the driver didn't fill the query.
        if (end <= start) return null;
        const ms = Number(end - start) / 1e6;
        recordPassMs(ms);
        return ms;
      } catch {
        return null;
      } finally {
        free();
      }
    },
  };
}
