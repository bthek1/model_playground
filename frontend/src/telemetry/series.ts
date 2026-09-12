// A fixed-length window of samples.
//
// The panel shows the last couple of minutes and nothing older, so the natural
// structure is a ring buffer with a hard cap: memory is bounded whatever happens,
// and an hour with the panel open costs exactly the same as a minute. A plain
// array with `shift()` would also work, but it re-indexes the whole array on
// every sample and grows without a cap the moment someone forgets the slice.
//
// Pure, React-free and timer-free on purpose — `useTelemetry` owns the cadence,
// this owns the arithmetic.

export const DEFAULT_CAPACITY = 120; // ≈2 min at 1 Hz

export interface Series<T> {
  readonly capacity: number;
  push: (sample: T) => void;
  /** Oldest → newest. A copy; mutating it cannot corrupt the buffer. */
  toArray: () => T[];
  latest: () => T | null;
  /** Number of samples held, never more than `capacity`. */
  size: () => number;
  clear: () => void;
}

export function createSeries<T>(capacity = DEFAULT_CAPACITY): Series<T> {
  if (!Number.isInteger(capacity) || capacity < 1) {
    throw new Error(`Series capacity must be a positive integer, got ${capacity}`);
  }

  // `head` is the next write slot; `count` saturates at `capacity`.
  let slots: (T | undefined)[] = new Array(capacity);
  let head = 0;
  let count = 0;

  return {
    capacity,

    push(sample) {
      slots[head] = sample;
      head = (head + 1) % capacity;
      if (count < capacity) count++;
    },

    toArray() {
      const out: T[] = [];
      const start = count < capacity ? 0 : head;
      for (let i = 0; i < count; i++) {
        out.push(slots[(start + i) % capacity] as T);
      }
      return out;
    },

    latest() {
      if (count === 0) return null;
      return slots[(head - 1 + capacity) % capacity] as T;
    },

    size() {
      return count;
    },

    clear() {
      slots = new Array(capacity);
      head = 0;
      count = 0;
    },
  };
}

/** Largest value in a numeric series, or `fallback` when it is empty. */
export function maxOf(values: number[], fallback = 0): number {
  let max = fallback;
  let seen = false;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    if (!seen || v > max) {
      max = v;
      seen = true;
    }
  }
  return seen ? max : fallback;
}
