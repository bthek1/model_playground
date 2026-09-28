// REINFORCE and Actor-Critic on CartPole — one gradient, two estimates of how
// much to trust each step.
//
// Both learners push up log π(aₜ | sₜ) for every action taken, weighted:
//
//   loss = −(1/T) Σₜ wₜ · log π(aₜ | sₜ)       ∂loss/∂logitsₜ = (πₜ − onehot(aₜ)) · wₜ / T
//
// That second line is the whole policy gradient, and it is the cross-entropy
// gradient `linearModel.ts` already computes, scaled by the weight — which is
// why this file reuses its softmax rather than writing a second one.
//
//   - **REINFORCE** weights each step by its discounted return Gₜ. Unbiased,
//     and violently noisy: one lucky episode of 200 steps pushes every action
//     in it up by a lot, whether or not that action helped.
//   - **Actor-Critic** weights it by the *advantage* Aₜ — how much better the
//     step went than a learned critic V(s) expected. Same gradient, far less
//     variance, a little bias. The advantage is GAE(λ): a λ-weighted sum of
//     one-step TD errors δₜ = rₜ + γ V(sₜ₊₁) − V(sₜ).
//
// Both update once per episode, from the whole trajectory, with Adam. The
// variance is the page's subject, so REINFORCE's return normalisation — the
// usual thumb on the scale — exists but is **off by default**.
//
// Two things fail silently here and are pinned by tests:
//
//   - **A wrong policy gradient still produces a rising curve** — a sign error
//     in the weight, or a gradient scaled by the wrong T, is absorbed into the
//     learning rate. `policyGradient.test.ts` checks each learner's gradient
//     against finite differences of its own loss, at a generic point.
//   - **Truncation is not termination.** At CartPole's 500-step cap the pole
//     is still up; the critic's target must bootstrap from V(s₅₀₀) there, and
//     only a *fall* has zero future value.

import { mulberry32 } from "@/lib/random";

import { CartPole } from "./envs/cartPole";
import {
  Adam,
  backward,
  forward,
  initNet,
  type NetGrads,
  type NetParams,
  softmax,
} from "./policyNet";
import type { Stepper } from "./stepper";
import type {
  CartPoleRenderState,
  PolicyGradientDiagnostics,
  PolicyGradientRequest,
} from "./types";

const OBS = 4;
const ACTIONS = 2;

/**
 * Defaults per algorithm, **measured, never inherited** — from each other or
 * from the grid. 400 episodes of CartPole, hidden 32, γ = 0.99, mean return of
 * the last 50 episodes over seeds 1–6 (Node, CPU reference; a whole run is
 * 0.1–0.6 s):
 *
 *   REINFORCE     lr 0.01                          79 ± 56   (one seed ends at 9)
 *   REINFORCE     lr 0.005                        193 ± 117
 *   REINFORCE     lr 0.003                        258 ± 34   ← default
 *   REINFORCE     lr 0.002                        176 ± 17
 *   REINFORCE     lr 0.003, normalised returns    428 ± 20
 *   Actor-Critic  lr 0.01, critic 0.01, λ 0.95    479 ± 46
 *   Actor-Critic  lr 0.003, critic 0.03, λ 0.9    489 ± 9
 *   Actor-Critic  lr 0.01, critic 0.03, λ 0.9     500 ± 1    ← default
 *
 * Giving REINFORCE the Actor-Critic's learning rate would have made the
 * comparison look even better and been unfair: at 0.01 REINFORCE collapses. It
 * gets its own best rate. And the normalisation row is why that switch is off
 * by default — it is most of the way to the baseline Actor-Critic learns.
 */
export const PG_DEFAULTS = {
  reinforce: { lr: 0.003, criticLr: 0.03, lambda: 0.9 },
  "actor-critic": { lr: 0.01, criticLr: 0.03, lambda: 0.9 },
  hidden: 32,
  gamma: 0.99,
  episodes: 400,
} as const;

/** Gₜ = rₜ + γ Gₜ₊₁, backward from the end of the episode. */
export function discountedReturns(rewards: readonly number[], gamma: number): Float64Array {
  const out = new Float64Array(rewards.length);
  let g = 0;
  for (let t = rewards.length - 1; t >= 0; t--) {
    g = rewards[t] + gamma * g;
    out[t] = g;
  }
  return out;
}

/** Zero mean, unit variance — REINFORCE's optional normalisation. */
export function standardise(xs: Float64Array): Float64Array {
  const n = xs.length;
  if (n === 0) return xs;
  let mean = 0;
  for (const x of xs) mean += x / n;
  let v = 0;
  for (const x of xs) v += ((x - mean) * (x - mean)) / n;
  const sd = Math.sqrt(v) || 1;
  return xs.map((x) => (x - mean) / sd);
}

/**
 * GAE(λ) advantages and the critic's targets for one trajectory.
 *
 * `values[t]` is V(sₜ); `lastValue` is V of the state after the final step,
 * used only when the episode was *truncated* — a fall has no future. When the
 * critic is exact every TD error is zero, so every advantage is too: that is
 * the property a test asserts, because "the loss falls" cannot express it.
 */
export function advantages(
  rewards: readonly number[],
  values: ArrayLike<number>,
  lastValue: number,
  terminated: boolean,
  gamma: number,
  lambda: number,
): { advantages: Float64Array; targets: Float64Array } {
  const T = rewards.length;
  const adv = new Float64Array(T);
  const targets = new Float64Array(T);
  let next = terminated ? 0 : lastValue;
  let gae = 0;
  for (let t = T - 1; t >= 0; t--) {
    const delta = rewards[t] + gamma * next - values[t];
    gae = delta + gamma * lambda * gae;
    adv[t] = gae;
    targets[t] = gae + values[t];
    next = values[t];
  }
  return { advantages: adv, targets };
}

/** −(1/T) Σ wₜ log π(aₜ | sₜ) — the loss the policy gradient descends. */
export function policyLoss(
  net: NetParams,
  obs: Float32Array,
  actions: ArrayLike<number>,
  weights: ArrayLike<number>,
): number {
  const T = actions.length;
  const probs = softmax(forward(net, obs, T).out, T, ACTIONS);
  let loss = 0;
  for (let t = 0; t < T; t++) {
    loss -= (weights[t] * Math.log(Math.max(probs[t * ACTIONS + actions[t]], 1e-12))) / T;
  }
  return loss;
}

/** The gradient of `policyLoss`, by hand: the cross-entropy gradient, times the weight. */
export function policyGradient(
  net: NetParams,
  obs: Float32Array,
  actions: ArrayLike<number>,
  weights: ArrayLike<number>,
): NetGrads {
  const T = actions.length;
  const cache = forward(net, obs, T);
  const probs = softmax(cache.out, T, ACTIONS);
  const dOut = new Float32Array(T * ACTIONS);
  for (let t = 0; t < T; t++) {
    for (let k = 0; k < ACTIONS; k++) {
      dOut[t * ACTIONS + k] = ((probs[t * ACTIONS + k] - (k === actions[t] ? 1 : 0)) * weights[t]) / T;
    }
  }
  return backward(net, cache, dOut);
}

/** ½ mean (V(sₜ) − targetₜ)², the critic's own loss. */
export function valueLoss(net: NetParams, obs: Float32Array, targets: ArrayLike<number>): number {
  const T = targets.length;
  const v = forward(net, obs, T).out;
  let loss = 0;
  for (let t = 0; t < T; t++) loss += (0.5 * (v[t] - targets[t]) ** 2) / T;
  return loss;
}

export function valueGradient(net: NetParams, obs: Float32Array, targets: ArrayLike<number>): NetGrads {
  const T = targets.length;
  const cache = forward(net, obs, T);
  const dOut = new Float32Array(T);
  for (let t = 0; t < T; t++) dOut[t] = (cache.out[t] - targets[t]) / T;
  return backward(net, cache, dOut);
}

function sample(probs: Float32Array, rand: () => number): number {
  return rand() < probs[0] ? 0 : 1;
}

function meanAbs(xs: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += Math.abs(xs[i]);
  return xs.length ? s / xs.length : 0;
}

/**
 * The part both learners share: roll CartPole forward one step at a time,
 * sampling from the policy, and hand the finished trajectory to `learn`.
 */
export class PolicyGradientRun implements Stepper {
  readonly env = new CartPole();
  readonly policy: NetParams;
  readonly critic: NetParams | null;
  readonly totalEpisodes: number;
  episode = 0;
  steps = 0;
  /** Not used — a policy explores by sampling, not by ε — but the seam carries it. */
  epsilonOverride: number | null = null;
  readonly epsilon = null;
  /** The last update's diagnostics: where the variance went. */
  diagnostics: PolicyGradientDiagnostics | null = null;

  private readonly rand: () => number;
  private readonly actorOpt: Adam;
  private readonly criticOpt: Adam | null;
  private obs: Float32Array;
  private readonly trajObs: number[] = [];
  private readonly trajActions: number[] = [];
  private readonly trajRewards: number[] = [];

  constructor(readonly req: PolicyGradientRequest) {
    this.totalEpisodes = req.episodes;
    this.policy = initNet({ inputs: OBS, hidden: req.hidden, outputs: ACTIONS }, req.seed, 0.1);
    this.actorOpt = new Adam(this.policy, req.lr);
    if (req.algorithm === "actor-critic") {
      this.critic = initNet({ inputs: OBS, hidden: req.hidden, outputs: 1 }, req.seed ^ 0x7c1, 1);
      this.criticOpt = new Adam(this.critic, req.criticLr);
    } else {
      this.critic = null;
      this.criticOpt = null;
    }
    // As on the grid: the agent's stream is not the environment's.
    this.rand = mulberry32(req.seed ^ 0x51ed27);
    this.obs = this.env.reset(req.seed);
  }

  get done(): boolean {
    return this.episode >= this.totalEpisodes;
  }

  step(): number | null {
    const probs = softmax(forward(this.policy, this.obs, 1).out, 1, ACTIONS);
    const a = sample(probs, this.rand);
    const r = this.env.step(a);
    for (const v of this.obs) this.trajObs.push(v);
    this.trajActions.push(a);
    this.trajRewards.push(r.reward);
    this.steps++;
    if (!r.terminated && !r.truncated) {
      this.obs = r.observation;
      return null;
    }
    const ret = this.trajRewards.reduce((s, x) => s + x, 0);
    this.learn(r.observation, r.terminated);
    this.trajObs.length = 0;
    this.trajActions.length = 0;
    this.trajRewards.length = 0;
    this.episode++;
    // Keep drawing the fallen pole until the next step, as the grid keeps the
    // agent in the hole — then start the next episode.
    this.obs = this.done ? r.observation : this.env.reset();
    return ret;
  }

  private learn(finalObs: Float32Array, terminated: boolean): void {
    const obs = Float32Array.from(this.trajObs);
    const { gamma } = this.req;
    let weights: Float64Array;
    let criticLoss: number | null = null;

    if (this.critic && this.criticOpt) {
      const T = this.trajActions.length;
      const values = forward(this.critic, obs, T).out;
      const lastValue = forward(this.critic, finalObs, 1).out[0];
      const { advantages: adv, targets } = advantages(
        this.trajRewards,
        values,
        lastValue,
        terminated,
        gamma,
        this.req.lambda,
      );
      weights = adv;
      criticLoss = valueLoss(this.critic, obs, targets);
      this.criticOpt.step(valueGradient(this.critic, obs, targets));
    } else {
      weights = discountedReturns(this.trajRewards, gamma);
      if (this.req.normalise) weights = standardise(weights);
    }

    this.actorOpt.step(policyGradient(this.policy, obs, this.trajActions, weights));
    this.diagnostics = { meanAbsWeight: meanAbs(weights), criticLoss };
  }

  render(): CartPoleRenderState {
    return {
      kind: "cartpole",
      state: Float32Array.from(this.env.state),
      diagnostics: this.diagnostics,
    };
  }
}
