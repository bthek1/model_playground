import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ProteinUnion } from "./proteins";
import {
  clearCachedUnion,
  loadCachedUnion,
  saveCachedUnion,
} from "./proteinsCache";

/** A minimal but internally consistent union: two nodes, one graph, one edge. */
function union(over: Partial<ProteinUnion> = {}): ProteinUnion {
  return {
    nGraphs: 1,
    nNodes: 2,
    nFeat: 3,
    nClasses: 2,
    rowPtr: Uint32Array.from([0, 1, 2]),
    colIdx: Uint32Array.from([1, 0]),
    degree: Uint32Array.from([1, 1]),
    features: new Float32Array(6),
    graphOf: Uint32Array.from([0, 0]),
    graphPtr: Uint32Array.from([0, 2]),
    labels: Uint8Array.from([1]),
    ...over,
  };
}

// happy-dom does not implement IndexedDB, so every helper must degrade to a
// no-op "no cache" mode rather than throwing — the page then re-downloads, which
// is slower but correct.
describe("proteinsCache without IndexedDB", () => {
  const original = (globalThis as { indexedDB?: unknown }).indexedDB;

  beforeEach(() => {
    Object.defineProperty(globalThis, "indexedDB", {
      value: undefined,
      configurable: true,
      writable: true,
    });
  });

  afterEach(() => {
    Object.defineProperty(globalThis, "indexedDB", {
      value: original,
      configurable: true,
      writable: true,
    });
    vi.restoreAllMocks();
  });

  it("loadCachedUnion resolves to null", async () => {
    await expect(loadCachedUnion()).resolves.toBeNull();
  });

  it("saveCachedUnion resolves without throwing", async () => {
    await expect(saveCachedUnion(union())).resolves.toBeUndefined();
  });

  it("clearCachedUnion resolves without throwing", async () => {
    await expect(clearCachedUnion()).resolves.toBeUndefined();
  });
});

// The validity check is the interesting half, because a union written by an older
// build would otherwise fail somewhere inside the training loop rather than here.
// It is exercised through a stub store, since happy-dom has no IndexedDB.
describe("proteinsCache validation", () => {
  const original = (globalThis as { indexedDB?: unknown }).indexedDB;
  let stored: unknown;

  beforeEach(() => {
    stored = undefined;
    const request = <T,>(result: () => T) => {
      const req: Record<string, unknown> = { result: undefined, error: null };
      queueMicrotask(() => {
        req.result = result();
        (req.onsuccess as (() => void) | undefined)?.();
      });
      return req;
    };
    Object.defineProperty(globalThis, "indexedDB", {
      value: {
        open: () =>
          request(() => ({
            objectStoreNames: { contains: () => true },
            transaction: () => {
              // `oncomplete` is assigned by the code under test, so it has to be
              // a setter that fires rather than a field that is merely stored.
              const tx = {
                objectStore: () => ({
                  get: () => request(() => stored),
                  put: (value: unknown) => {
                    stored = value;
                    return request(() => undefined);
                  },
                  delete: () => {
                    stored = undefined;
                    return request(() => undefined);
                  },
                }),
              };
              Object.defineProperty(tx, "oncomplete", {
                set: (fn: () => void) => queueMicrotask(fn),
                configurable: true,
              });
              return tx;
            },
            close: () => {},
          })),
      },
      configurable: true,
      writable: true,
    });
  });

  afterEach(() => {
    Object.defineProperty(globalThis, "indexedDB", {
      value: original,
      configurable: true,
      writable: true,
    });
  });

  it("returns a round-tripped union", async () => {
    await saveCachedUnion(union());
    const back = await loadCachedUnion();
    expect(back?.nGraphs).toBe(1);
    expect(Array.from(back!.colIdx)).toEqual([1, 0]);
  });

  it("rejects a union whose CSR does not match its node count", async () => {
    // The shape an older build's cache would have. Returning null re-downloads;
    // returning it would fail three layers down, inside the training loop.
    await saveCachedUnion(union({ rowPtr: Uint32Array.from([0, 1]) }));
    await expect(loadCachedUnion()).resolves.toBeNull();
  });

  it("rejects a union whose features are the wrong length", async () => {
    await saveCachedUnion(union({ features: new Float32Array(5) }));
    await expect(loadCachedUnion()).resolves.toBeNull();
  });

  it("rejects a union whose rowPtr disagrees with its colIdx", async () => {
    await saveCachedUnion(union({ colIdx: Uint32Array.from([1]) }));
    await expect(loadCachedUnion()).resolves.toBeNull();
  });

  it("forgets the union when cleared", async () => {
    await saveCachedUnion(union());
    await clearCachedUnion();
    await expect(loadCachedUnion()).resolves.toBeNull();
  });
});
