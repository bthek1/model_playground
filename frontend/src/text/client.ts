// Main-thread factory for the generic text worker. Isolated in its own module
// (like `vision/client.ts`) so `useTextPipeline` can mock worker creation in
// tests without touching `import.meta.url` / `new Worker`, neither of which
// resolves under happy-dom.

export function createTextWorker(): Worker {
  return new Worker(new URL("./pipeline.worker.ts", import.meta.url), {
    type: "module",
  });
}

/**
 * The **streaming** worker, for `/text-generation` alone.
 *
 * A second factory rather than a parameter on the one above: they load
 * different modules, and the generic worker's bundle has no business pulling in
 * `TextStreamer` and the generation machinery for the four pages that never
 * stream.
 */
export function createTextGenWorker(): Worker {
  return new Worker(new URL("./textgen.worker.ts", import.meta.url), {
    type: "module",
  });
}
