// Seeded pseudo-randomness, shared by everything in the app that has to be
// reproducible: training runs, dataset splits and graph layouts. A layout that
// moved on every reload, or a split that changed between two architectures being
// compared, would make every head-to-head on the page meaningless.

/** Deterministic small-state PRNG. Same seed, same stream, everywhere. */
export function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** In-place Fisher–Yates shuffle driven by a supplied RNG. */
export function shuffle(
  arr: Int32Array | Uint32Array | number[],
  rand: () => number,
): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
}

/**
 * One draw from a standard normal, via Box–Muller. Used for weight init, where
 * the alternative — uniform noise — changes the variance the init was chosen for.
 */
export function gaussian(rand: () => number): number {
  // rand() can return 0, and log(0) is -Infinity.
  const u = 1 - rand();
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
