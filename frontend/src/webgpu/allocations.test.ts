import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  aggregateSnapshot,
  attachAllocationPort,
  localSnapshot,
  PUBLISH_MS,
  recordPassMs,
  registerAllocationPort,
  releaseBuffer,
  resetAllocations,
  STALE_MS,
  trackBuffer,
  watchAllocations,
} from "./allocations";

function fakeBuffer(): GPUBuffer {
  return { destroy: vi.fn() } as unknown as GPUBuffer;
}

/**
 * A synchronous MessagePort pair. The real thing delivers on a task, which
 * would make every assertion below a race; the logic under test is the
 * bookkeeping, not the platform's queueing.
 */
function fakePortPair(): [MessagePort, MessagePort] {
  const a = { onmessage: null } as unknown as MessagePort;
  const b = { onmessage: null } as unknown as MessagePort;
  a.postMessage = (data: unknown) =>
    b.onmessage?.({ data } as MessageEvent<unknown>);
  b.postMessage = (data: unknown) =>
    a.onmessage?.({ data } as MessageEvent<unknown>);
  return [a, b];
}

beforeEach(() => resetAllocations());
afterEach(() => {
  resetAllocations();
  vi.useRealTimers();
});

describe("the local ledger", () => {
  it("starts empty", () => {
    expect(localSnapshot()).toEqual({
      liveBytes: 0,
      peakBytes: 0,
      buffers: 0,
      lastPassMs: null,
    });
  });

  it("counts a tracked buffer and returns it unchanged", () => {
    const buffer = fakeBuffer();
    expect(trackBuffer(buffer, 1024)).toBe(buffer);
    expect(localSnapshot()).toMatchObject({
      liveBytes: 1024,
      peakBytes: 1024,
      buffers: 1,
    });
  });

  it("releases bytes back and destroys the buffer", () => {
    const buffer = trackBuffer(fakeBuffer(), 2048);
    releaseBuffer(buffer);
    expect(buffer.destroy).toHaveBeenCalledOnce();
    expect(localSnapshot()).toMatchObject({ liveBytes: 0, buffers: 0 });
  });

  it("keeps the peak after everything is released — the ceiling is the point", () => {
    const a = trackBuffer(fakeBuffer(), 1000);
    const b = trackBuffer(fakeBuffer(), 3000);
    expect(localSnapshot().peakBytes).toBe(4000);
    releaseBuffer(a);
    releaseBuffer(b);
    expect(localSnapshot()).toMatchObject({ liveBytes: 0, peakBytes: 4000 });
  });

  it("destroys an untracked buffer without going negative", () => {
    const buffer = fakeBuffer();
    releaseBuffer(buffer);
    expect(buffer.destroy).toHaveBeenCalledOnce();
    expect(localSnapshot()).toMatchObject({ liveBytes: 0, buffers: 0 });
  });

  it("ignores a zero or nonsense size rather than counting a buffer of unknown cost", () => {
    trackBuffer(fakeBuffer(), 0);
    trackBuffer(fakeBuffer(), NaN);
    expect(localSnapshot()).toMatchObject({ liveBytes: 0, buffers: 0 });
  });

  it("records a pass duration, and rejects a negative one", () => {
    recordPassMs(1.25);
    expect(localSnapshot().lastPassMs) .toBe(1.25);
    recordPassMs(-1);
    expect(localSnapshot().lastPassMs).toBe(1.25);
  });
});

describe("aggregating across realms", () => {
  it("is just the local ledger with no workers attached", () => {
    trackBuffer(fakeBuffer(), 500);
    expect(aggregateSnapshot()).toMatchObject({ liveBytes: 500, buffers: 1 });
  });

  it("publishes nothing until the page starts watching", () => {
    vi.useFakeTimers();
    const [pagePort, workerPort] = fakePortPair();
    registerAllocationPort(pagePort);
    attachAllocationPort(workerPort);

    trackBuffer(fakeBuffer(), 4096); // "worker" realm — same module in a test
    vi.advanceTimersByTime(PUBLISH_MS * 3);
    // Only the local ledger; the port has sent nothing.
    expect(aggregateSnapshot()).toMatchObject({ liveBytes: 4096, buffers: 1 });
  });

  it("sums a watched realm on top of the local one", () => {
    const [pagePort, workerPort] = fakePortPair();
    registerAllocationPort(pagePort);
    attachAllocationPort(workerPort);
    trackBuffer(fakeBuffer(), 1000);

    watchAllocations(true); // triggers an immediate publish of {1000, …}

    // The ledger is shared in this test, so the aggregate doubles — which is
    // exactly the summation being asserted.
    expect(aggregateSnapshot()).toMatchObject({
      liveBytes: 2000,
      buffers: 2,
    });
  });

  it("forgets a realm that has gone quiet — a terminated worker is not a total", () => {
    const [pagePort, workerPort] = fakePortPair();
    registerAllocationPort(pagePort);
    attachAllocationPort(workerPort);
    trackBuffer(fakeBuffer(), 1000);
    watchAllocations(true);

    const now = Date.now();
    expect(aggregateSnapshot(now).liveBytes).toBe(2000);
    expect(aggregateSnapshot(now + STALE_MS + 1).liveBytes).toBe(1000);
  });

  it("stops publishing and drops remote totals when the panel closes", () => {
    vi.useFakeTimers();
    const [pagePort, workerPort] = fakePortPair();
    registerAllocationPort(pagePort);
    attachAllocationPort(workerPort);
    trackBuffer(fakeBuffer(), 1000);

    watchAllocations(true);
    expect(aggregateSnapshot().liveBytes).toBe(2000);

    watchAllocations(false);
    expect(aggregateSnapshot().liveBytes).toBe(1000);

    // And the worker's interval is gone, so nothing arrives later either.
    vi.advanceTimersByTime(PUBLISH_MS * 5);
    expect(aggregateSnapshot().liveBytes).toBe(1000);
  });

  it("publishes on the interval while watched", () => {
    vi.useFakeTimers();
    const [pagePort, workerPort] = fakePortPair();
    const received: unknown[] = [];
    registerAllocationPort(pagePort);
    attachAllocationPort(workerPort);
    const original = pagePort.onmessage!;
    pagePort.onmessage = (event) => {
      received.push(event.data);
      original.call(pagePort, event);
    };

    watchAllocations(true);
    expect(received).toHaveLength(1); // immediate, so the panel is never blank
    vi.advanceTimersByTime(PUBLISH_MS * 2);
    expect(received).toHaveLength(3);
  });

  it("a port registered while already watching starts publishing at once", () => {
    watchAllocations(true);
    const [pagePort, workerPort] = fakePortPair();
    attachAllocationPort(workerPort);
    trackBuffer(fakeBuffer(), 800);
    registerAllocationPort(pagePort);
    expect(aggregateSnapshot().liveBytes).toBe(1600);
  });
});
