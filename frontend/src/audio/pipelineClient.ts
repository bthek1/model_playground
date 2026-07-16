// Main-thread factory for the generic pipeline worker. Isolated in its own module
// (like `webgpu/workerClient.ts`) so `usePipeline` can mock worker creation in
// tests without touching `import.meta.url`/`new Worker`, which don't resolve under
// happy-dom.

export function createPipelineWorker(): Worker {
  return new Worker(new URL("./pipeline.worker.ts", import.meta.url), {
    type: "module",
  });
}
