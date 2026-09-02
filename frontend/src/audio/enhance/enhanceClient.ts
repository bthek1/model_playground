// Main-thread factory for the enhancement worker. Isolated in its own module
// (like `pipelineClient.ts`) so `useEnhance` can mock worker creation in tests
// without touching `import.meta.url` / `new Worker`, neither of which resolves
// under happy-dom.

export function createEnhanceWorker(): Worker {
  return new Worker(new URL("./enhance.worker.ts", import.meta.url), {
    type: "module",
  });
}
