// Memory, to the very limited extent a page can see it.
//
// There is no API for the machine's RAM usage. There is `performance.memory`,
// which is Chrome/Edge-only, non-standard, quantised to 100 KB buckets, and
// reports the JS heap of **the realm that asks** — so the page's own heap, never
// the workers where Transformers.js holds the weights. On a 300 MB model load
// this number barely moves, and a card that implied otherwise would be worse
// than no card.
//
// The accurate cross-realm answer, `measureUserAgentSpecificMemory()`, needs
// cross-origin isolation (COOP/COEP). This app is not isolated, and turning that
// on changes how the Hugging Face CDN fetches and ORT's threading behave, so it
// is deliberately out of scope — see docs/explanations/telemetry-panel.md.

import { ok, unavailable, type Metric, type MemorySample } from "../types";

/** The non-standard Chrome shape. Absent everywhere else. */
interface ChromeMemory {
  usedJSHeapSize: number;
  totalJSHeapSize: number;
  jsHeapSizeLimit: number;
}

const NOT_IMPLEMENTED =
  "Only Chrome and Edge expose a JS heap size (performance.memory); this browser does not.";

export function sampleMemory(): Metric<MemorySample> {
  if (typeof performance === "undefined") return unavailable(NOT_IMPLEMENTED);

  const memory = (performance as Performance & { memory?: ChromeMemory }).memory;
  if (!memory || typeof memory.usedJSHeapSize !== "number") {
    return unavailable(NOT_IMPLEMENTED);
  }

  return ok({
    usedBytes: memory.usedJSHeapSize,
    totalBytes: memory.totalJSHeapSize,
    limitBytes: memory.jsHeapSizeLimit,
  });
}

/**
 * `navigator.deviceMemory` — the machine's RAM rounded to a power of two and
 * capped at 8, on purpose, to resist fingerprinting. Capacity, not usage.
 */
export function deviceMemoryGiB(): Metric<number> {
  if (typeof navigator === "undefined") {
    return unavailable("No navigator in this environment.");
  }
  const gib = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  if (typeof gib !== "number" || !Number.isFinite(gib) || gib <= 0) {
    return unavailable(
      "This browser doesn't report a device memory class (navigator.deviceMemory).",
    );
  }
  return ok(gib);
}
