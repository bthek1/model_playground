// Main-thread factory for the VAD worker. Isolated in its own module (like
// `enhanceClient.ts`) so `useVad` can mock worker creation in tests without
// touching `import.meta.url` / `new Worker`, neither of which resolves under
// happy-dom.

export function createVadWorker(): Worker {
  return new Worker(new URL("./vad.worker.ts", import.meta.url), {
    type: "module",
  });
}
