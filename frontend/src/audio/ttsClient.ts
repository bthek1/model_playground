// Main-thread factory for the TTS worker. Isolated in its own module (like
// `webgpu/workerClient.ts`) so `useTts` can mock worker creation in tests without
// touching `import.meta.url`/`new Worker`, which don't resolve under happy-dom.

export function createTtsWorker(): Worker {
  return new Worker(new URL("./tts.worker.ts", import.meta.url), {
    type: "module",
  });
}
