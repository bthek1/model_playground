// The exact optimal action values Q* for a finite MDP whose model is known —
// the CPU reference the learner is checked against.
//
// This is the most valuable file in the module, and the reason is the same one
// `/graph`'s finite-difference check exists for: **a wrong Bellman update still
// produces a rising return curve.** Drop the discount, bootstrap through a
// terminal state, or take the max over the wrong row, and the agent still finds
// the goal on a 4×4 grid; its values are simply wrong, and so is its policy
// wherever two routes nearly tie. Only comparing against the exact answer
// catches that, and on an MDP small enough to enumerate the exact answer is
// twenty lines:
//
//   Q*(s, a) = Σ_{s'} P(s' | s, a) · [ r + γ · max_a' Q*(s', a') · (1 − terminated) ]
//
// iterated to a fixed point. It converges for any γ < 1 because the Bellman
// operator is a γ-contraction in the max-norm.

import type { FiniteMdp } from "./types";

export interface ValueIterationResult {
  /** `nStates × nActions`, row-major — the same layout as `QTable.q`. */
  q: Float64Array;
  /** `max_a Q*(s, a)`. */
  v: Float64Array;
  iterations: number;
  /** Largest change in the final sweep. */
  residual: number;
}

export function valueIteration(
  mdp: FiniteMdp,
  gamma: number,
  { tolerance = 1e-10, maxIterations = 10_000 } = {},
): ValueIterationResult {
  if (!(gamma >= 0 && gamma < 1)) {
    throw new Error(`value iteration needs 0 ≤ γ < 1 to converge (got ${gamma})`);
  }
  const { nStates: S, nActions: A } = mdp;
  const q = new Float64Array(S * A);
  const v = new Float64Array(S);
  let residual = Infinity;
  let iterations = 0;

  while (residual > tolerance && iterations < maxIterations) {
    residual = 0;
    for (let s = 0; s < S; s++) {
      for (let a = 0; a < A; a++) {
        let value = 0;
        for (const t of mdp.transitions(s, a)) {
          value += t.p * (t.reward + (t.terminated ? 0 : gamma * v[t.next]));
        }
        residual = Math.max(residual, Math.abs(value - q[s * A + a]));
        q[s * A + a] = value;
      }
    }
    // Synchronous backup: every Q in a sweep reads the previous sweep's V.
    for (let s = 0; s < S; s++) {
      let best = -Infinity;
      for (let a = 0; a < A; a++) best = Math.max(best, q[s * A + a]);
      v[s] = best;
    }
    iterations++;
  }

  return { q, v, iterations, residual };
}

/**
 * Every action within `tolerance` of the best, per state. A policy check has to
 * accept any of them: from the 4×4 start, right-then-down and down-then-right
 * are exactly as long, and a test that demanded one would be pinning a tie.
 */
export function optimalActions(
  q: ArrayLike<number>,
  nStates: number,
  nActions: number,
  tolerance = 1e-6,
): number[][] {
  const out: number[][] = [];
  for (let s = 0; s < nStates; s++) {
    let best = -Infinity;
    for (let a = 0; a < nActions; a++) best = Math.max(best, q[s * nActions + a]);
    const set: number[] = [];
    for (let a = 0; a < nActions; a++) {
      if (best - q[s * nActions + a] <= tolerance) set.push(a);
    }
    out.push(set);
  }
  return out;
}
