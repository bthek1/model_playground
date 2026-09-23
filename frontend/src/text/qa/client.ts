// Main-thread factory for the QA worker. Isolated in its own module (as
// `text/client.ts` and `vision/zeroshot/client.ts` are) so `useQa` can mock
// worker creation in tests without touching `import.meta.url` / `new Worker`,
// neither of which resolves under happy-dom.

export function createQaWorker(): Worker {
  return new Worker(new URL("./worker.ts", import.meta.url), {
    type: "module",
  });
}
