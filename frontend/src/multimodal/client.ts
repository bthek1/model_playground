// Main-thread factory for the image-text-to-text worker. Isolated in its own
// module (as the vision clients are) so `useVlm` can mock worker creation in
// tests without touching `import.meta.url` / `new Worker`, neither of which
// resolves under happy-dom.

export function createVlmWorker(): Worker {
  return new Worker(new URL("./vlm.worker.ts", import.meta.url), {
    type: "module",
  });
}
