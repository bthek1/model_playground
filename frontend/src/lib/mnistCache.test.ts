import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { IMAGE_SIZE, type MnistPool } from "./mnist";
import { clearCachedPool, loadCachedPool, saveCachedPool } from "./mnistCache";

// happy-dom does not implement IndexedDB, so the helpers must degrade to a
// no-op "no cache" mode rather than throwing. When `indexedDB` is stubbed absent
// entirely, every helper resolves to its safe default.
describe("mnistCache without IndexedDB", () => {
  const original = (globalThis as { indexedDB?: unknown }).indexedDB;

  beforeEach(() => {
    // Force the unavailable path regardless of the test environment.
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

  const pool: MnistPool = {
    images: new Float32Array(2 * IMAGE_SIZE),
    labels: new Uint8Array(2),
    count: 2,
  };

  it("loadCachedPool resolves to null", async () => {
    await expect(loadCachedPool()).resolves.toBeNull();
  });

  it("saveCachedPool resolves without throwing", async () => {
    await expect(saveCachedPool(pool)).resolves.toBeUndefined();
  });

  it("clearCachedPool resolves without throwing", async () => {
    await expect(clearCachedPool()).resolves.toBeUndefined();
  });
});
