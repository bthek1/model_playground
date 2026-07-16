// Main-thread factory for the ASR worker. Isolated in its own module (like
// `webgpu/workerClient.ts`) so `useAsr` can mock worker creation in tests without
// touching `import.meta.url`/`new Worker`, which don't resolve under happy-dom.

export function createAsrWorker(): Worker {
  return new Worker(new URL("./asr.worker.ts", import.meta.url), {
    type: "module",
  });
}
