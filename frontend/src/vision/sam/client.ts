// Main-thread factory for the SAM worker. Isolated in its own module (as
// `vision/client.ts` and `vision/zeroshot/client.ts` are) so `useSam` can mock
// worker creation in tests without touching `import.meta.url` / `new Worker`,
// neither of which resolves under happy-dom.

export function createSamWorker(): Worker {
  return new Worker(new URL("./sam.worker.ts", import.meta.url), {
    type: "module",
  });
}
