// Main-thread factory for the pose worker. Isolated in its own module (as the
// other vision clients are) so `usePose` can mock worker creation in tests
// without touching `import.meta.url` / `new Worker`, neither of which resolves
// under happy-dom.

export function createPoseWorker(): Worker {
  return new Worker(new URL("./pose.worker.ts", import.meta.url), {
    type: "module",
  });
}
