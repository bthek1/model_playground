import { afterEach, describe, expect, it, vi } from "vitest";

import {
  busyFraction,
  cores,
  createLagTracker,
  createLongTaskWatcher,
  FRAME_BUDGET_MS,
  STALL_MS,
} from "./cpu";

afterEach(() => vi.unstubAllGlobals());

describe("createLagTracker", () => {
  it("reports no lag before two frames have been seen", () => {
    const tracker = createLagTracker();
    tracker.frame(0);
    expect(tracker.frames()).toBe(0);
    expect(tracker.consumeMaxLagMs()).toBe(0);
  });

  it("reports zero for frames that hit their budget", () => {
    const tracker = createLagTracker();
    tracker.frame(0);
    tracker.frame(FRAME_BUDGET_MS);
    tracker.frame(FRAME_BUDGET_MS * 2);
    expect(tracker.consumeMaxLagMs()).toBe(0);
  });

  it("reports the overrun of a slow frame", () => {
    const tracker = createLagTracker(16);
    tracker.frame(0);
    tracker.frame(100);
    expect(tracker.consumeMaxLagMs()).toBe(84);
  });

  it("keeps the worst frame in the interval, not the last", () => {
    const tracker = createLagTracker(16);
    tracker.frame(0);
    tracker.frame(200); // 184 ms of lag
    tracker.frame(216); // on budget
    expect(tracker.consumeMaxLagMs()).toBe(184);
  });

  it("resets after a consume", () => {
    const tracker = createLagTracker(16);
    tracker.frame(0);
    tracker.frame(100);
    tracker.consumeMaxLagMs();
    expect(tracker.consumeMaxLagMs()).toBe(0);
  });

  it("discards a stall — a suspended tab is not main-thread contention", () => {
    const tracker = createLagTracker(16);
    tracker.frame(0);
    tracker.frame(STALL_MS + 1000);
    expect(tracker.consumeMaxLagMs()).toBe(0);
    expect(tracker.frames()).toBe(0);
    // …and the clock re-bases, so the frame after a stall isn't lag either.
    tracker.frame(STALL_MS + 1016);
    expect(tracker.consumeMaxLagMs()).toBe(0);
  });

  it("a 120 Hz display reports less lag, never negative", () => {
    const tracker = createLagTracker();
    tracker.frame(0);
    tracker.frame(8.3);
    expect(tracker.consumeMaxLagMs()).toBe(0);
  });
});

describe("createLongTaskWatcher", () => {
  it("is unavailable with a reason where the entry type is not supported", () => {
    vi.stubGlobal(
      "PerformanceObserver",
      Object.assign(vi.fn(), { supportedEntryTypes: ["resource"] }),
    );
    const watcher = createLongTaskWatcher();
    expect(watcher.unavailableReason).toMatch(/Chromium/);
    expect(watcher.consume()).toBeNull();
    expect(() => watcher.stop()).not.toThrow();
  });

  it("is unavailable where PerformanceObserver itself is missing", () => {
    vi.stubGlobal("PerformanceObserver", undefined);
    expect(createLongTaskWatcher().unavailableReason).toMatch(
      /no PerformanceObserver/,
    );
  });

  it("collects entries and resets on consume when supported", () => {
    let emit: ((list: { getEntries: () => { duration: number }[] }) => void) | null =
      null;
    const observe = vi.fn();
    const disconnect = vi.fn();
    class FakeObserver {
      static supportedEntryTypes = ["longtask"];
      constructor(callback: (list: { getEntries: () => { duration: number }[] }) => void) {
        emit = callback;
      }
      observe = observe;
      disconnect = disconnect;
    }
    vi.stubGlobal("PerformanceObserver", FakeObserver);

    const watcher = createLongTaskWatcher();
    expect(watcher.unavailableReason).toBeNull();
    expect(observe).toHaveBeenCalledWith({ type: "longtask", buffered: false });

    emit!({ getEntries: () => [{ duration: 80 }, { duration: 120 }] });
    expect(watcher.consume()).toEqual({ count: 2, totalMs: 200 });
    expect(watcher.consume()).toEqual({ count: 0, totalMs: 0 });

    watcher.stop();
    expect(disconnect).toHaveBeenCalledOnce();
  });

  it("treats a refused observe as unsupported rather than throwing", () => {
    class ThrowingObserver {
      static supportedEntryTypes = ["longtask"];
      observe() {
        throw new Error("not allowed here");
      }
      disconnect() {}
    }
    vi.stubGlobal("PerformanceObserver", ThrowingObserver);
    expect(createLongTaskWatcher().consume()).toBeNull();
  });
});

describe("cores", () => {
  it("reports hardwareConcurrency", () => {
    vi.stubGlobal("navigator", { hardwareConcurrency: 16 });
    expect(cores()).toEqual({ status: "ok", value: 16 });
  });

  it("is unavailable when withheld", () => {
    vi.stubGlobal("navigator", {});
    expect(cores().status).toBe("unavailable");
  });
});

describe("busyFraction", () => {
  it("is a ratio of the interval", () => {
    expect(busyFraction(250, 1000)).toBe(0.25);
  });

  it("clamps to 1 — a run can span more than one interval", () => {
    expect(busyFraction(1500, 1000)).toBe(1);
  });

  it("is zero for a nonsense interval rather than Infinity", () => {
    expect(busyFraction(100, 0)).toBe(0);
    expect(busyFraction(NaN, 1000)).toBe(0);
  });
});
