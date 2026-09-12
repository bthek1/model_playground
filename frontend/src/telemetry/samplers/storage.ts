// Disk, which is the one resource a browser reports honestly.
//
// `navigator.storage.estimate()` gives usage and quota for the whole origin, and
// for this app the overwhelming majority of that is model weights: every file
// Transformers.js fetches lands in the `transformers-cache` bucket and outlives
// the tab. So this is the card that answers "how much of my disk have I spent on
// models, and how much more will the browser let me spend" — the counterpart to
// the size guardrail in `model/size.ts`, after the fact rather than before it.
//
// The model *count* comes from `model/cache.ts`, which already knows how to read
// that bucket. It walks every cache key, so it is sampled rarely rather than
// every tick — see `useTelemetry`.

import { cachedModels } from "@/model/cache";

import { ok, unavailable, type Metric, type StorageSample } from "../types";

const NO_ESTIMATE =
  "This browser doesn't expose a storage estimate (navigator.storage.estimate).";

export interface StorageOptions {
  /** Also count models in the cache — a full walk of the bucket's keys. */
  countModels?: boolean;
}

export async function sampleStorage(
  { countModels = false }: StorageOptions = {},
): Promise<Metric<StorageSample>> {
  if (typeof navigator === "undefined" || !navigator.storage?.estimate) {
    return unavailable(NO_ESTIMATE);
  }

  let usage: number | undefined;
  let quota: number | undefined;
  try {
    ({ usage, quota } = await navigator.storage.estimate());
  } catch {
    // Some privacy modes reject outright rather than reporting zero.
    return unavailable(
      "The browser refused a storage estimate — private browsing blocks it.",
    );
  }

  if (typeof usage !== "number" || typeof quota !== "number") {
    return unavailable(NO_ESTIMATE);
  }

  // `cachedModels` never rejects; it answers "none" on an unreadable cache,
  // which is indistinguishable from an empty one and treated the same way.
  const models = countModels ? (await cachedModels()).size : null;

  return ok({ usageBytes: usage, quotaBytes: quota, cachedModels: models });
}
