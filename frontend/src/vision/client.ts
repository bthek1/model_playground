// Main-thread factory for the generic vision worker. Isolated in its own module
// (like `audio/pipelineClient.ts`) so `useVisionPipeline` can mock worker
// creation in tests without touching `import.meta.url` / `new Worker`, neither of
// which resolves under happy-dom.

export function createVisionWorker(): Worker {
  return new Worker(new URL("./vision.worker.ts", import.meta.url), {
    type: "module",
  });
}
