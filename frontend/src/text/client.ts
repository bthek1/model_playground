// Main-thread factory for the generic text worker. Isolated in its own module
// (like `vision/client.ts`) so `useTextPipeline` can mock worker creation in
// tests without touching `import.meta.url` / `new Worker`, neither of which
// resolves under happy-dom.

export function createTextWorker(): Worker {
  return new Worker(new URL("./pipeline.worker.ts", import.meta.url), {
    type: "module",
  });
}
