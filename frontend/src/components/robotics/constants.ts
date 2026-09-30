// Constants the cloning view and its tests share, kept out of the component
// files so fast refresh keeps working on them.

import type { ReachOutcome } from "@/rl/envs/reacher2d";

/** Side colours: fixed hues, because they encode meaning (model-visualization.md §3). */
export const SIDE_COLORS = { above: "rgba(59,130,246,0.55)", below: "rgba(245,158,11,0.6)" } as const;

/**
 * Measured, not inherited from CartPole: over seeds 1–6 the bimodal set's
 * policy heads 0–10° off the obstacle between the modes and collides 5 times in
 * 6; the unimodal one heads 16–39° off and reaches 6 of 6. ~0.4 s a run.
 */
export const CLONING_DEFAULTS = { demos: 20, obstacle: 0, hidden: 64, epochs: 100, lr: 0.01, seed: 1 };

export const VERDICT: Record<ReachOutcome, string> = {
  reached: "reached the target",
  collided: "drove into the obstacle",
  stalled: "stalled short of the target",
};
