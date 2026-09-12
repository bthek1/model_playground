import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { localSnapshot, resetAllocations } from "./allocations";
import { createPassTimer } from "./timing";

/** A device whose query set resolves to `[start, end]` nanosecond timestamps. */
function fakeDevice({
  features = ["timestamp-query"],
  timestamps = [1_000n, 1_420_000n],
  mapRejects = false,
}: {
  features?: string[];
  timestamps?: [bigint, bigint];
  mapRejects?: boolean;
} = {}) {
  const destroyed: string[] = [];
  const readBytes = new ArrayBuffer(16);
  new BigUint64Array(readBytes).set(timestamps);

  const buffers: GPUBuffer[] = [];
  const device = {
    features: new Set(features),
    createQuerySet: vi.fn(() => ({
      destroy: () => destroyed.push("querySet"),
    })),
    createBuffer: vi.fn(() => {
      const buffer = {
        destroy: () => destroyed.push(`buffer${buffers.length}`),
        mapAsync: mapRejects
          ? vi.fn().mockRejectedValue(new Error("device lost"))
          : vi.fn().mockResolvedValue(undefined),
        getMappedRange: () => readBytes,
        unmap: vi.fn(),
      } as unknown as GPUBuffer;
      buffers.push(buffer);
      return buffer;
    }),
  } as unknown as GPUDevice;

  return { device, destroyed };
}

function fakeEncoder() {
  return {
    resolveQuerySet: vi.fn(),
    copyBufferToBuffer: vi.fn(),
  } as unknown as GPUCommandEncoder;
}

beforeEach(() => {
  resetAllocations();
  // The WebGPU bitflag globals only exist in a browser; `createPassTimer`
  // treats their absence as "cannot time" (which is why the tests above that
  // expect null still pass without them).
  vi.stubGlobal("GPUBufferUsage", {
    QUERY_RESOLVE: 1 << 0,
    COPY_SRC: 1 << 1,
    MAP_READ: 1 << 2,
    COPY_DST: 1 << 3,
  });
  vi.stubGlobal("GPUMapMode", { READ: 1 });
});

afterEach(() => {
  resetAllocations();
  vi.unstubAllGlobals();
});

describe("createPassTimer", () => {
  it("is null on a device without the timestamp-query feature", () => {
    const { device } = fakeDevice({ features: ["shader-f16"] });
    expect(createPassTimer(device)).toBeNull();
  });

  it("is null where createQuerySet is missing entirely", () => {
    const device = { features: new Set(["timestamp-query"]) } as unknown as GPUDevice;
    expect(createPassTimer(device)).toBeNull();
  });

  it("is null — not throwing — when the device refuses to allocate", () => {
    const device = {
      features: new Set(["timestamp-query"]),
      createQuerySet: () => {
        throw new Error("too many query sets");
      },
      createBuffer: vi.fn(),
    } as unknown as GPUDevice;
    expect(createPassTimer(device)).toBeNull();
  });

  it("writes the beginning and end of the pass", () => {
    const { device } = fakeDevice();
    const timer = createPassTimer(device)!;
    expect(timer.timestampWrites.beginningOfPassWriteIndex).toBe(0);
    expect(timer.timestampWrites.endOfPassWriteIndex).toBe(1);
  });

  it("queues a resolve and a copy for readback", () => {
    const { device } = fakeDevice();
    const timer = createPassTimer(device)!;
    const encoder = fakeEncoder();

    timer.resolve(encoder);
    expect(encoder.resolveQuerySet).toHaveBeenCalledWith(
      timer.timestampWrites.querySet,
      0,
      2,
      expect.anything(),
      0,
    );
    expect(encoder.copyBufferToBuffer).toHaveBeenCalled();
  });

  it("reports the pass duration in ms and records it in the ledger", async () => {
    const { device, destroyed } = fakeDevice({ timestamps: [1_000n, 1_421_000n] });
    const timer = createPassTimer(device)!;

    await expect(timer.readMs()).resolves.toBeCloseTo(1.42, 5);
    expect(localSnapshot().lastPassMs).toBeCloseTo(1.42, 5);
    // Query set and both buffers are freed, whatever happened.
    expect(destroyed).toHaveLength(3);
  });

  it("reports nothing when the driver left the query unfilled", async () => {
    const { device } = fakeDevice({ timestamps: [0n, 0n] });
    await expect(createPassTimer(device)!.readMs()).resolves.toBeNull();
    expect(localSnapshot().lastPassMs).toBeNull();
  });

  it("swallows a readback failure — timing must not fail the job", async () => {
    const { device, destroyed } = fakeDevice({ mapRejects: true });
    await expect(createPassTimer(device)!.readMs()).resolves.toBeNull();
    expect(destroyed).toHaveLength(3);
  });

  it("does not count its own 32 bytes in the allocation ledger", async () => {
    const { device } = fakeDevice();
    createPassTimer(device);
    expect(localSnapshot()).toMatchObject({ liveBytes: 0, buffers: 0 });
  });
});
