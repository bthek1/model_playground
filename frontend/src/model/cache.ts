// What is already on this machine.
//
// Weight downloads are the expensive, consented step (model-page-pattern.md §5),
// but the second load of a model costs nothing: `@huggingface/transformers`
// stores every file it fetches in a Cache Storage bucket that outlives the tab.
// Knowing whether a model is in there is what lets the picker say "already
// downloaded" and what lets a page resume after a refresh without re-asking for
// consent it doesn't need.
//
// Everything here is defensive. `caches` is absent on a non-secure origin and
// throws outright in some privacy modes, and the cache name is a Transformers.js
// implementation detail that an upgrade could rename. Every failure resolves to
// "not cached": the cost of being wrong that way is a lost badge and one extra
// click, and never a broken page.

/** The bucket `@huggingface/transformers` writes to. Their name, not ours. */
const CACHE_NAME = "transformers-cache";

/**
 * `https://huggingface.co/<org>/<name>/resolve/main/…` → `<org>/<name>`.
 * Anything not shaped like a Hub file is ignored rather than guessed at.
 */
function modelIdOf(url: string): string | null {
  try {
    const parts = new URL(url).pathname.split("/").filter(Boolean);
    const resolve = parts.indexOf("resolve");
    if (resolve < 2) return null;
    return parts.slice(resolve - 2, resolve).join("/");
  } catch {
    return null;
  }
}

async function openCache(): Promise<Cache | null> {
  try {
    if (typeof caches === "undefined") return null;
    return await caches.open(CACHE_NAME);
  } catch {
    return null;
  }
}

/** Every model id with at least one file in the cache. Never rejects. */
export async function cachedModels(): Promise<Set<string>> {
  const found = new Set<string>();
  const cache = await openCache();
  if (!cache) return found;
  try {
    for (const request of await cache.keys()) {
      const id = modelIdOf(request.url);
      if (id) found.add(id);
    }
  } catch {
    /* an unreadable cache is an empty one */
  }
  return found;
}

/**
 * Is this model's weights already downloaded? Strict on purpose — the resume
 * rule in §5 hangs off this answer, and a false positive spends the user's
 * bandwidth without asking. Matched on the model's own path segments, never a
 * substring.
 */
export async function isModelCached(modelId: string): Promise<boolean> {
  if (!modelId) return false;
  return (await cachedModels()).has(modelId);
}

/** Drop a model's files. For reclaiming space, and for testing a cold load. */
export async function evictModel(modelId: string): Promise<void> {
  const cache = await openCache();
  if (!cache) return;
  try {
    const keys = await cache.keys();
    await Promise.all(
      keys
        .filter((r) => modelIdOf(r.url) === modelId)
        .map((r) => cache.delete(r)),
    );
  } catch {
    /* nothing to reclaim */
  }
}
