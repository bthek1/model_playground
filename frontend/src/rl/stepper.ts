// The seam between a learner and the loop that drives it.
//
// Every algorithm in the category is advanced **one environment step at a
// time** through this interface, so `session.ts` can pace a run, draw it on a
// wall-clock interval and stop it between any two steps without knowing which
// algorithm it is holding. Q-learning updates on every step; REINFORCE only at
// the end of an episode; Actor-Critic on every step again — all of that stays
// inside `step()`, and the loop never learns the difference.

import type { RlRenderState } from "./types";

export interface Stepper {
  readonly totalEpisodes: number;
  /** Episodes finished so far. */
  readonly episode: number;
  /** Environment steps taken so far. */
  readonly steps: number;
  readonly done: boolean;
  /** The exploration rate in force, for the progress line; null if the learner has none. */
  readonly epsilon: number | null;
  /** Set by the live control; `null` returns the run to its own schedule. */
  epsilonOverride: number | null;
  /** One environment step. Returns the episode's return if it just ended, else null. */
  step(): number | null;
  /** A fresh copy of what the canvas draws — the loop keeps writing its own. */
  render(): RlRenderState;
}
