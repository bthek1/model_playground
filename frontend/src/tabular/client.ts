import type { ParseResult } from "./types";

// Main-thread factory for the fit worker. Isolated in its own module — as
// `text/client.ts` and `vision/client.ts` are — so `useTabularFit` can mock
// worker creation in tests without touching `import.meta.url` or `new Worker`,
// neither of which resolves under happy-dom.

export function createFitWorker(): Worker {
  return new Worker(new URL("./fit.worker.ts", import.meta.url), {
    type: "module",
  });
}

/**
 * Parse CSV text in a one-shot worker.
 *
 * Resolves with the same `ParseResult` `parseCsv` returns — the worker exists
 * to keep a ten-megabyte parse off the main thread, not to change the contract.
 * The worker is terminated as soon as it answers: a parse is a single event in
 * a session, and a resident worker per file would be a leak with a nicer name.
 */
export function parseCsvInWorker(
  text: string,
  options: { name?: string; maxRows?: number } = {},
  createWorker: () => Worker = createParseWorker,
): Promise<ParseResult> {
  return new Promise<ParseResult>((resolve, reject) => {
    const worker = createWorker();
    worker.onmessage = (event: MessageEvent<ParseWorkerResponse>) => {
      worker.terminate();
      if (event.data.ok) resolve(event.data.result);
      else reject(new Error(event.data.error));
    };
    worker.onerror = (event: ErrorEvent) => {
      worker.terminate();
      reject(new Error(event.message || "The parser worker failed."));
    };
    worker.postMessage({ id: 1, text, ...options });
  });
}

type ParseWorkerResponse =
  | { id: number; ok: true; result: ParseResult }
  | { id: number; ok: false; error: string };

export function createParseWorker(): Worker {
  return new Worker(new URL("./parse.worker.ts", import.meta.url), {
    type: "module",
  });
}
