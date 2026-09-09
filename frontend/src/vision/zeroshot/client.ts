// Main-thread factory for the zero-shot worker. Isolated in its own module (as
// `vision/client.ts` is) so `useZeroShotImage` can mock worker creation in tests
// without touching `import.meta.url` / `new Worker`, neither of which resolves
// under happy-dom.

export function createZeroShotWorker(): Worker {
  return new Worker(new URL("./worker.ts", import.meta.url), {
    type: "module",
  });
}
