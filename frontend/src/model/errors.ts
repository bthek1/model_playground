// Load failures, translated.
//
// A model load fails through a long chain — fetch, Hub auth, ONNX Runtime,
// WebGPU device allocation — and every layer throws a string written for
// whoever wrote that layer. `Unauthorized access to file` means "that model id
// isn't on the Hub"; `qdq_actions.cc:137 … Missing required scale` means "this
// quantization can't open a session here". Neither tells the user what to do.
//
// So we classify: a plain sentence plus the one action worth offering. The raw
// message is never swallowed — `ErrorNote` keeps it in a `<details>`, because it
// is the only thing worth pasting into a bug report.

export type LoadErrorCause =
  | "offline"
  | "not-found"
  | "out-of-memory"
  | "device-lost"
  | "unknown";

export interface LoadErrorInfo {
  cause: LoadErrorCause;
  /** One sentence, in the user's terms. */
  message: string;
  /** What to try next. Omitted when Retry is the whole answer. */
  hint?: string;
  /** True when falling back to CPU is a plausible fix — see §6 of the pattern. */
  suggestsCpu: boolean;
  /** The original, verbatim. */
  raw: string;
}

export function classifyLoadError(error: unknown): LoadErrorInfo {
  const raw =
    error instanceof Error ? error.message : String(error ?? "Unknown error");
  const text = raw.toLowerCase();

  const offline =
    typeof navigator !== "undefined" && navigator.onLine === false;
  if (
    offline ||
    text.includes("failed to fetch") ||
    text.includes("networkerror") ||
    text.includes("err_internet_disconnected")
  ) {
    return {
      cause: "offline",
      message: offline
        ? "You appear to be offline, so the weights could not be downloaded."
        : "The weights could not be downloaded — the network request failed.",
      hint: "Models already downloaded still load from the browser cache.",
      suggestsCpu: false,
      raw,
    };
  }

  if (
    text.includes("unauthorized") ||
    text.includes("404") ||
    text.includes("401") ||
    text.includes("could not locate") ||
    text.includes("not found")
  ) {
    return {
      cause: "not-found",
      message: "This model could not be found on the Hugging Face Hub.",
      hint: "The repository may have moved or be private — try another model.",
      suggestsCpu: false,
      raw,
    };
  }

  if (
    text.includes("out of memory") ||
    // Word-bounded: "boom" in an ONNX stack trace is not an OOM.
    /\boom\b/.test(text) ||
    text.includes("allocation failed") ||
    text.includes("array buffer allocation")
  ) {
    return {
      cause: "out-of-memory",
      message: "The browser ran out of memory loading this model.",
      hint: "Close other tabs, or pick a smaller model.",
      suggestsCpu: false,
      raw,
    };
  }

  if (
    text.includes("device lost") ||
    text.includes("device is lost") ||
    text.includes("webgpu") ||
    text.includes("gpu adapter") ||
    text.includes("createshadermodule")
  ) {
    return {
      cause: "device-lost",
      message: "The GPU could not run this model.",
      hint: "Loading on the CPU is slower but works everywhere.",
      suggestsCpu: true,
      raw,
    };
  }

  return {
    cause: "unknown",
    message: "The model failed to load.",
    suggestsCpu: true,
    raw,
  };
}
