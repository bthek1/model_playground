// The caps, and the measurement that set them.
//
// The Tabular category passes `adding-a-task-page.md` §0 question 2 vacuously —
// nothing is downloaded — so the feasibility bar here is not a download size,
// it is **whether the fit fits**. That had to be measured before the page was
// designed around it, because the answer could have cut a family off the ladder.
//
// Phase 0 of #48, on synthetic data of 20 columns, two classes, fitted in Node
// with the **CPU-reference matmul** (so the two GPU families below are upper
// bounds — on a real device they are faster; the trees never touch the GPU, so
// their numbers are what a browser sees):
//
//   rows     bin    forest 60×d8   boost 120×d4   boost 120×d6   logistic 30ep   MLP 40ep
//   10 000    21ms       1.0 s          1.7 s          3.0 s          0.14 s       1.8 s
//   50 000   106ms       4.6 s          7.7 s         12.2 s          0.71 s       9.2 s
//   200 000  467ms      16.9 s         31.4 s         51.0 s          3.20 s      38.0 s
//
// Three conclusions, and they are the reason this file exists rather than a
// round number appearing in the UI:
//
//   1. **Nothing is cut.** Every family fits 10 000 rows in three seconds or
//      less, so the ladder ships whole — including the MLP, which is on it
//      precisely because it loses on accuracy rather than on time.
//   2. **`MAX_ROWS` is 50 000, and gradient boosting at depth 6 set it.** At
//      200 000 rows that configuration is 51 seconds, which is not something a
//      page can offer behind a progress bar and a Stop button and still be
//      honest about; at 50 000 the worst case on the ladder is 12 seconds.
//   3. **Boosting's depth is capped at 6**, from the same table: depth 6 costs
//      1.6× depth 4 at every size, and depth 8 would put the worst case back
//      over twenty seconds at the row cap. The forest keeps a higher cap
//      because a bagged tree is fitted once rather than once per boosting round
//      per class.

/**
 * Rows kept from an uploaded file. Sampled evenly across it, never truncated to
 * a prefix, and the UI states both numbers — modelling the first 50 000 rows of
 * a file sorted by date is a different dataset from the one handed over.
 */
export const MAX_ROWS = 50_000;

/** Depth cap for gradient boosting. See conclusion 3 above. */
export const MAX_BOOST_DEPTH = 6;

/** Depth cap for a single tree or a bagged forest. */
export const MAX_TREE_DEPTH = 12;

/**
 * Roughly how long a fit will take, in milliseconds, so the page can warn
 * before the click rather than after it.
 *
 * Interpolated from the table above — linear in rows, and for the tree families
 * linear in `nTrees` and roughly 1.25× per level of depth. It is an estimate and
 * is rendered as one ("about 8 s"); the point is the order of magnitude, which
 * is what separates "press it" from "reduce something first".
 */
export function estimateFitMs(
  family: string,
  rows: number,
  hp: { nTrees: number; maxDepth: number; epochs: number; hidden: number },
  classes = 2,
): number {
  const scale = rows / 10_000;
  switch (family) {
    case "forest":
      return scale * 1000 * (hp.nTrees / 60) * Math.pow(1.25, hp.maxDepth - 8);
    case "boosting":
      return (
        scale * 1700 * (hp.nTrees / 120) * (classes / 2) * Math.pow(1.25, hp.maxDepth - 4)
      );
    case "logistic":
      return scale * 140 * (hp.epochs / 30);
    case "mlp":
      return scale * 1850 * (hp.epochs / 40) * (hp.hidden / 48);
    default:
      return scale * 1000;
  }
}

/**
 * A fit duration in words — "about 8 s".
 *
 * Rounded hard on purpose. The estimate above is an interpolation and quoting
 * it to the millisecond would claim a precision it does not have; what the user
 * needs is the difference between "press it" and "reduce something first".
 */
export function describeDuration(ms: number): string {
  if (ms < 900) return "under a second";
  if (ms < 60_000) return `about ${Math.round(ms / 1000)} s`;
  return `about ${Math.round(ms / 60_000)} min`;
}
