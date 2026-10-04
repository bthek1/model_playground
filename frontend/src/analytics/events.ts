// What analytics may say (#60): what the app *did*, never what it was *given*.
//
// Several pages promise "nothing is uploaded", and an analytics call is an
// upload. So every property passes through a closed allowlist here before it
// can reach the SDK: an unknown key is dropped, a string must look like an
// identifier (a catalogue id, a route pattern, an enum) rather than prose, and
// an error is reduced to a category — its message can echo a prompt, a file
// name or a path, so the message itself never leaves.

export type AnalyticsEvent =
  | "$pageview"
  | "model_load_started"
  | "model_load_ready"
  | "model_load_failed"
  | "model_load_cancelled"
  | "model_run_completed"
  | "model_run_failed"
  | "webgpu_status"
  | "feature_used";

/** The only property keys an event may carry. */
export const ALLOWED_PROPS = [
  "route",
  "task",
  "modelId",
  "backend",
  "retry",
  "loadedInMs",
  "elapsedMs",
  "runMs",
  "errorKind",
  "webgpuStatus",
  "shaderF16",
  "feature",
  "family",
] as const;

export type AllowedProp = (typeof ALLOWED_PROPS)[number];
export type Props = Partial<Record<AllowedProp, string | number | boolean | null | undefined>>;

/**
 * An identifier, not a sentence: catalogue ids (`Xenova/distilbert-…`),
 * route patterns (`/text-classification`), enums (`webgpu`, `tabular_fit`).
 * No whitespace, so a typed sentence cannot pass even if a caller tried.
 */
const SAFE_STRING = /^[A-Za-z0-9_./:@-]{1,120}$/;

const allowed = new Set<string>(ALLOWED_PROPS);

/** Keep allowlisted keys with identifier-shaped values; drop everything else. */
export function sanitize(props: Record<string, unknown> = {}): Record<string, string | number | boolean> {
  const out: Record<string, string | number | boolean> = {};
  for (const [key, value] of Object.entries(props)) {
    if (!allowed.has(key)) continue;
    if (typeof value === "boolean") out[key] = value;
    else if (typeof value === "number" && Number.isFinite(value)) out[key] = Math.round(value);
    else if (typeof value === "string" && SAFE_STRING.test(value)) out[key] = value;
  }
  return out;
}

export type ErrorKind =
  | "network"
  | "not_found"
  | "out_of_memory"
  | "gpu_unavailable"
  | "gpu_feature_missing"
  | "session_create"
  | "worker_terminated"
  | "cancelled"
  | "unknown";

/**
 * Reduce an error to a category. Order matters: the more specific patterns go
 * first (a failed fetch of a missing file is `not_found`, not `network`).
 */
export function classifyError(error: unknown): ErrorKind {
  const message = (error instanceof Error ? error.message : String(error ?? "")).toLowerCase();
  if (/cancel/.test(message)) return "cancelled";
  if (/terminated/.test(message)) return "worker_terminated";
  if (/\b404\b|not found|could not locate|401\b/.test(message)) return "not_found";
  if (/out of memory|oom\b|allocation failed|array buffer allocation|exceeds.*memory/.test(message))
    return "out_of_memory";
  if (/shader-f16|requires f16|feature.*not (supported|enabled)/.test(message))
    return "gpu_feature_missing";
  if (/webgpu|adapter|gpu device|device (was )?lost|navigator\.gpu/.test(message))
    return "gpu_unavailable";
  if (/session|qdq_actions|missing required scale|onnx/.test(message)) return "session_create";
  if (/fetch|network|load failed|cors|err_/.test(message)) return "network";
  return "unknown";
}
