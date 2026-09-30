// Permutation importance: shuffle one column, re-score, restore.
//
// It is a loop with a shuffle in the middle, and it is the output non-technical
// users actually read — "which of my columns mattered" is the question a
// spreadsheet cannot answer and a fitted model can. It is model-agnostic by
// construction, so the same function serves all four families on the ladder.
//
// The two things that go wrong here are both silent. **Restoring the column**
// is the first: an in-place shuffle that forgets to put the values back poisons
// every column measured after it, and still returns a full, ordered, plausible
// ranking. The second is the shuffle's *stream* — see `permutation()` in
// `split.ts` for why each column draws its own rather than sharing one.

import { permutation } from "./split";
import type { DesignColumn } from "./design";
import type { FeatureImportance } from "./types";

export interface ImportanceOptions {
  /** Repeats per column. Averaging over a few shuffles steadies the ranking. */
  repeats?: number;
  seed: number;
  onProgress?: (done: number, total: number) => void;
  shouldStop?: () => boolean;
}

/**
 * Score every source column by how much shuffling it costs.
 *
 * `score` is handed the (possibly permuted) design matrix and returns a
 * higher-is-better number — accuracy for a classifier, R² for a regressor. The
 * drop is `baseline − permuted`, so a column the model ignores scores ~0 and a
 * column it depends on scores large and positive. A *negative* drop is real and
 * is left as it is: it means the column was actively misleading the model on
 * the held-out half, which is worth seeing rather than clamping away.
 *
 * Grouped by **source column**, not by design column: a one-hot encoded city
 * with eleven levels is one thing the user chose, and shuffling one of its
 * eleven indicators independently would break the encoding into an impossible
 * row rather than measure anything.
 */
export async function permutationImportance(
  x: Float32Array,
  rows: number,
  design: DesignColumn[],
  sourceNames: Map<number, string>,
  score: (matrix: Float32Array) => Promise<number>,
  options: ImportanceOptions,
): Promise<FeatureImportance[]> {
  const repeats = options.repeats ?? 3;
  const width = design.length;
  const baseline = await score(x);

  // Design columns grouped by the dataset column they came from.
  const groups = new Map<number, number[]>();
  for (let j = 0; j < width; j++) {
    const bucket = groups.get(design[j].source);
    if (bucket) bucket.push(j);
    else groups.set(design[j].source, [j]);
  }

  const out: FeatureImportance[] = [];
  const sources = [...groups.keys()];
  const saved = new Float32Array(rows);

  for (let g = 0; g < sources.length; g++) {
    if (options.shouldStop?.()) break;
    const source = sources[g];
    const cols = groups.get(source) as number[];
    let total = 0;
    for (let rep = 0; rep < repeats; rep++) {
      // One permutation for the whole group, so the row's city stays a city.
      const perm = permutation(rows, options.seed + source * 1013 + rep);
      for (const j of cols) {
        for (let i = 0; i < rows; i++) saved[i] = x[i * width + j];
        for (let i = 0; i < rows; i++) x[i * width + j] = saved[perm[i]];
      }
      total += baseline - (await score(x));
      // Restore — before the next repeat, and before the next column. This is
      // the line whose absence returns a complete and wrong ranking.
      for (const j of cols) {
        // `saved` holds the last column's originals only, so the restore reads
        // the permutation backwards instead: `x[perm[i]] = x_current[i]` undoes
        // `x_new[i] = x_old[perm[i]]` exactly, with no second buffer per column.
        for (let i = 0; i < rows; i++) saved[i] = x[i * width + j];
        for (let i = 0; i < rows; i++) x[perm[i] * width + j] = saved[i];
      }
    }
    out.push({
      name: sourceNames.get(source) ?? `column ${source}`,
      drop: total / repeats,
    });
    options.onProgress?.(g + 1, sources.length);
  }

  out.sort((a, b) => b.drop - a.drop);
  return out;
}
