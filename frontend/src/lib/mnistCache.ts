// Persist the decoded MNIST pool across page reloads. The pool is a large binary
// blob (a Float32Array of count×784 pixels plus labels — ~31 MB for 10k images),
// far too big for localStorage, so it lives in IndexedDB, which structured-clones
// typed arrays natively and has no practical size limit for this.
//
// Everything here is best-effort: if IndexedDB is unavailable (SSR, tests, a
// locked-down browser) or a write hits a quota error, the helpers degrade to
// "no cache" rather than throwing, so training still works — it just re-downloads.

import { IMAGE_SIZE, type MnistPool } from "./mnist";

const DB_NAME = "model-playground";
const DB_VERSION = 1;
const STORE = "mnist";
const KEY = "pool";

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
      reject(req.error ?? new Error("Could not open the MNIST cache database."));
  });
}

/** Sanity-check a value read back from the store before trusting it. */
function isValidPool(value: unknown): value is MnistPool {
  if (!value || typeof value !== "object") return false;
  const p = value as Partial<MnistPool>;
  return (
    p.images instanceof Float32Array &&
    p.labels instanceof Uint8Array &&
    typeof p.count === "number" &&
    p.count > 0 &&
    p.images.length === p.count * IMAGE_SIZE &&
    p.labels.length === p.count
  );
}

/** Read the persisted pool, or `null` if none / unavailable / invalid. */
export async function loadCachedPool(): Promise<MnistPool | null> {
  if (!hasIndexedDB()) return null;
  try {
    const db = await openDB();
    const pool = await new Promise<unknown>((resolve, reject) => {
      const tx = db.transaction(STORE, "readonly");
      const req = tx.objectStore(STORE).get(KEY);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
    db.close();
    return isValidPool(pool) ? pool : null;
  } catch {
    return null;
  }
}

/** Persist the pool (overwriting any previous one). Best-effort. */
export async function saveCachedPool(pool: MnistPool): Promise<void> {
  if (!hasIndexedDB()) return;
  try {
    const db = await openDB();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, "readwrite");
      tx.objectStore(STORE).put(pool, KEY);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error);
    });
    db.close();
  } catch {
    // Quota or transaction failure just means no persistence this session.
  }
}

/** Drop the persisted pool. Best-effort. */
export async function clearCachedPool(): Promise<void> {
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
