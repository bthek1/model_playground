// Main-thread factory for the document-QA worker. Isolated in its own module so
// `useDocVqa` can mock worker creation in tests without touching
// `import.meta.url` / `new Worker`, neither of which resolves under happy-dom.

export function createDocVqaWorker(): Worker {
  return new Worker(new URL("./docvqa.worker.ts", import.meta.url), {
    type: "module",
  });
}
