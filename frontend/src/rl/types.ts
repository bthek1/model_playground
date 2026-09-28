// The shapes every Reinforcement Learning page shares: the environment
// interface, the training request, and the worker protocol.
//
// `src/rl/` is plain TypeScript and owns no GPU code, the way `src/tabular/`
// and `src/forecast/` do. Phase 0 of #51 measured why (see `limits.ts`): an RL
// step is thousands of *tiny* matmuls per second, not one big one, and every
// size this category offers is smaller than the round trip to the device.

/** What an observation or an action can be. Gymnasium's two spaces, and no more. */
export type Space =
  | { kind: "discrete"; n: number }
  | { kind: "box"; size: number; low: readonly number[]; high: readonly number[] };

/**
 * One environment step. `terminated` and `truncated` are separate on purpose,
 * as they are in Gymnasium: an episode that ends because the agent fell in a
 * hole has no future value, and one that ends because a time limit ran out
 * *does*. A learner that bootstraps on `terminated || truncated` treats the
 * clock as a wall, which is wrong, silent, and shows up only as values that are
 * a little too low near the start.
 */
export interface StepResult<O> {
  observation: O;
  reward: number;
  terminated: boolean;
  truncated: boolean;
}

/**
 * The interface everything in the category plugs into — the grid here,
 * CartPole and the reacher later. `reset(seed)` reseeds the environment's own
 * stream, and `reset()` without one continues it, which is Gymnasium's
 * convention: seed once per run, not once per episode, or every episode starts
 * identically on a stochastic environment.
 */
export interface Env<O, A> {
  readonly observationSpace: Space;
  readonly actionSpace: Space;
  reset(seed?: number): O;
  /** Throws once the episode has ended: stepping a finished episode is a bug. */
  step(action: A): StepResult<O>;
}

/** One outcome of taking an action, for a finite MDP whose model is known. */
export interface Transition {
  p: number;
  next: number;
  reward: number;
  terminated: boolean;
}

/**
 * A finite MDP with its transition model exposed — what value iteration needs.
 * Only environments small enough to enumerate implement it, which is exactly
 * the set where an exact reference exists.
 */
export interface FiniteMdp {
  readonly nStates: number;
  readonly nActions: number;
  transitions(state: number, action: number): readonly Transition[];
}

// ---------------------------------------------------------------------------
// The training request

export type GridMapId = "4x4" | "8x8";

export interface QLearningRequest {
  algorithm: "q-learning";
  map: GridMapId;
  slippery: boolean;
  /** Learning rate. */
  alpha: number;
  /** Discount. */
  gamma: number;
  /** Exploration rate at the start of the run; may be changed live. */
  epsilon: number;
  /**
   * Decay ε linearly to `EPSILON_FLOOR` over the first 80% of the episodes.
   * See `epsilonAt` in `qLearning.ts` for why a constant ε rarely works here.
   */
  decay: boolean;
  episodes: number;
  seed: number;
}

/**
 * REINFORCE or Actor-Critic on CartPole (#52). The two share every field the
 * head-to-head holds fixed; `criticLr` and `lambda` are read only by
 * Actor-Critic, `normalise` only by REINFORCE.
 */
export interface PolicyGradientRequest {
  algorithm: "reinforce" | "actor-critic";
  env: "cartpole";
  hidden: number;
  /** Actor (policy) learning rate, Adam. */
  lr: number;
  /** Critic learning rate, Adam. */
  criticLr: number;
  gamma: number;
  /** GAE's λ: 0 is one-step TD, 1 is the Monte Carlo return minus a baseline. */
  lambda: number;
  /** Standardise REINFORCE's returns per episode. Off by default — see policyGradient.ts. */
  normalise: boolean;
  episodes: number;
  seed: number;
}

export type RlTrainRequest = QLearningRequest | PolicyGradientRequest;
export type RlAlgorithm = RlTrainRequest["algorithm"];

/** What the canvas draws. Posted on a wall-clock interval, never per step. */
export interface GridRenderState {
  kind: "grid";
  map: GridMapId;
  /** `nStates × 4`, row-major — a copy, so the loop can keep writing its own. */
  q: Float32Array;
  /** Where the agent is right now. */
  agent: number;
}

/** Where the variance went: the weight each learner put on ∇log π, and the critic's loss. */
export interface PolicyGradientDiagnostics {
  /** Mean |Gₜ| for REINFORCE, mean |Aₜ| for Actor-Critic, over the last episode. */
  meanAbsWeight: number;
  /** The critic's ½·MSE on the last episode; null for REINFORCE. */
  criticLoss: number | null;
}

export interface CartPoleRenderState {
  kind: "cartpole";
  /** `[x, x_dot, theta, theta_dot]`. */
  state: Float32Array;
  diagnostics: PolicyGradientDiagnostics | null;
}

export type RlRenderState = GridRenderState | CartPoleRenderState;

/** Everything since the last post — so the chart gets every episode, unsmoothed. */
export interface RlProgress {
  /** Returns of the episodes finished since the previous progress message. */
  returns: number[];
  /** Episodes finished so far, including those above. */
  episode: number;
  totalEpisodes: number;
  /** Environment steps taken so far. */
  steps: number;
  /** The exploration rate in force; null for a learner that explores by sampling. */
  epsilon: number | null;
  render: RlRenderState;
}

export interface RlTrainResult {
  /** Episodes actually finished — fewer than requested when stopped. */
  episodes: number;
  steps: number;
  elapsedMs: number;
  stopped: boolean;
  /** True if ε was changed while the run was in progress. */
  epsilonChanged: boolean;
  /** Null when stopped before anything could be drawn. */
  render: RlRenderState | null;
}

/** Live controls: they change the behaviour of the run in progress, never restart it. */
export interface RlControl {
  epsilon?: number;
  /** Environment steps per second; `null` means as fast as the CPU goes. */
  speed?: number | null;
}

// ---------------------------------------------------------------------------
// The worker protocol — `workerClient.ts`'s envelope, so the client reads alike.

export type RlWorkerRequest =
  | {
      type: "train";
      id: number;
      req: RlTrainRequest;
      speed?: number | null;
      /** Override the render interval. Only Phase 0's measurement sets it (to 0). */
      renderIntervalMs?: number;
    }
  | { type: "cancel"; id: number }
  | { type: "control"; id: number; control: RlControl };

export type RlWorkerResponse =
  | { id: number; event: "progress"; progress: RlProgress }
  | { id: number; ok: true; result: RlTrainResult }
  | { id: number; ok: false; error: string };
