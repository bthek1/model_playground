// Persist the built PROTEINS union across page reloads.
//
// The same shape as mnistCache.ts, and best-effort in the same way: if IndexedDB
// is unavailable (SSR, tests, a locked-down browser) or a write hits quota, every
// helper degrades to "no cache" rather than throwing, so the page re-downloads
// instead of failing.
//
// Two decisions worth the lines:
//
//   - **Its own database, not a second store in `model-playground`.** Adding a
//     store to that database means bumping its version, and `mnistCache.ts` opens
//     it at version 1 — once a v2 exists, that open throws `VersionError` and the
//     training route loses its cache. A separate database costs nothing and
//     couples nothing.
//   - **The union is cached, not the 2 MB of JSON.** The download is one request;
//     parsing it and building the block-diagonal CSR is the slower half, and
//     IndexedDB structured-clones typed arrays natively, so the union goes in as
//     it stands.

import type { ProteinUnion } from "./proteins";

const DB_NAME = "model-playground-proteins";
const DB_VERSION = 1;
const STORE = "proteins";
const KEY = "union";

function hasIndexedDB(): boolean {
  return typeof indexedDB !== "undefined";
}

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () =>
      reject(req.error ?? new Error("Could not open the PROTEINS cache."));
  });
}

/**
 * Sanity-check a value read back before trusting it.
 *
 * A cache written by an older build could have a different field set, and a
 * half-recognised union would fail somewhere inside the training loop instead of
 * here. The internal consistency checks are cheap and catch exactly that.
 */
function isValidUnion(value: unknown): value is ProteinUnion {
  if (!value || typeof value !== "object") return false;
  const u = value as Partial<ProteinUnion>;
  return (
    typeof u.nGraphs === "number" &&
    typeof u.nNodes === "number" &&
    typeof u.nFeat === "number" &&
    u.nGraphs > 0 &&
    u.nNodes > 0 &&
    u.rowPtr instanceof Uint32Array &&
    u.colIdx instanceof Uint32Array &&
    u.degree instanceof Uint32Array &&
    u.graphOf instanceof Uint32Array &&
    u.graphPtr instanceof Uint32Array &&
    u.features instanceof Float32Array &&
    u.labels instanceof Uint8Array &&
    u.rowPtr.length === u.nNodes + 1 &&
    u.graphPtr.length === u.nGraphs + 1 &&
    u.graphOf.length === u.nNodes &&
    u.labels.length === u.nGraphs &&
    u.features.length === u.nNodes * u.nFeat &&
    u.rowPtr[u.nNodes] === u.colIdx.length
  );
}

/** Read the persisted union, or `null` if none / unavailable / invalid. */
export async function loadCachedUnion(): Promise<ProteinUnion | null> {
  if (!hasIndexedDB()) return null;
  try {
    const db = await openDB();
    const union = await new Promise<unknown>((resolve, reject) => {
      const tx = db.transaction(STORE, "readonly");
      const req = tx.objectStore(STORE).get(KEY);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
    db.close();
    return isValidUnion(union) ? union : null;
  } catch {
    return null;
  }
}

/** Persist the union (overwriting any previous one). Best-effort. */
export async function saveCachedUnion(union: ProteinUnion): Promise<void> {
  if (!hasIndexedDB()) return;
  try {
    const db = await openDB();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, "readwrite");
      tx.objectStore(STORE).put(union, KEY);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error);
    });
    db.close();
  } catch {
    // Quota or transaction failure just means no persistence this session.
  }
}

/** Drop the persisted union. Best-effort. */
export async function clearCachedUnion(): Promise<void> {
  if (!hasIndexedDB()) return;
  try {
    const db = await openDB();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, "readwrite");
      tx.objectStore(STORE).delete(KEY);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
    db.close();
  } catch {
    // ignore
  }
}
