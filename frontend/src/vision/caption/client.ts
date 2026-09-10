// Main-thread factory for the image-to-text worker. Isolated in its own module
// (as the other three vision clients are) so `useImageToText` can mock worker
// creation in tests without touching `import.meta.url` / `new Worker`, neither
// of which resolves under happy-dom.

export function createCaptionWorker(): Worker {
  return new Worker(new URL("./caption.worker.ts", import.meta.url), {
    type: "module",
  });
}
