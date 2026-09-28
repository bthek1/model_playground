// CartPole, transcribed from Gymnasium rather than invented.
//
// Source: Gymnasium `gymnasium/envs/classic_control/cartpole.py` (CartPole-v1),
// https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py
// Every constant below was checked against that file (Gymnasium 1.3.0) and
// matches the values #52 listed.
//
// **The physics is pinned by a trajectory Gymnasium itself generated**
// (`fixtures/cartpole-trajectory.json`, from `scripts/make-cartpole-fixture.py`),
// because a CartPole with a sign error in the pole update trains perfectly
// happily: the agent learns whatever physics it is given, and the return curve
// rises either way. The classic mistake is `length` — it is **half** the pole's
// length, and upstream's own comment says so.
//
// Two things that are easy to get subtly wrong:
//
//   - **The terminating step still pays 1.** Upstream rewards the step on which
//     the pole falls (`steps_beyond_terminated is None` → 1.0), so a return of
//     N means the pole fell on step N.
//   - **The 500-step cap is a truncation, not a success.** `CartPole-v1` is
//     registered with `max_episode_steps=500`. A learner that treats the cap as
//     terminal would value a balanced pole at zero future reward — and a
//     termination test that was too lenient would produce a flat 500 that looks
//     like mastery.

import { mulberry32 } from "@/lib/random";

import type { Env, StepResult } from "../types";

export const GRAVITY = 9.8;
export const MASS_CART = 1.0;
export const MASS_POLE = 0.1;
export const TOTAL_MASS = MASS_POLE + MASS_CART;
/** Actually half the pole's length — upstream's own comment. */
export const HALF_LENGTH = 0.5;
export const POLE_MASS_LENGTH = MASS_POLE * HALF_LENGTH;
export const FORCE_MAG = 10.0;
/** Seconds between state updates. */
export const TAU = 0.02;
/** 12 degrees. */
export const THETA_THRESHOLD = (12 * 2 * Math.PI) / 360;
export const X_THRESHOLD = 2.4;
/** `CartPole-v1`'s registered `max_episode_steps`. */
export const TIME_LIMIT = 500;

/** `[x, x_dot, theta, theta_dot]`. */
export type CartPoleState = [number, number, number, number];

/**
 * One Euler step of upstream's equations of motion. Pure, so the fixture can
 * pin it. `force` overrides the action's ±10 N — only the zero-force physics
 * test passes it.
 */
export function cartPoleDynamics(
  state: CartPoleState,
  action: number,
  force: number = action === 1 ? FORCE_MAG : -FORCE_MAG,
): CartPoleState {
  const [x, xDot, theta, thetaDot] = state;
  const cos = Math.cos(theta);
  const sin = Math.sin(theta);
  const temp = (force + POLE_MASS_LENGTH * thetaDot * thetaDot * sin) / TOTAL_MASS;
  const thetaAcc =
    (GRAVITY * sin - cos * temp) /
    (HALF_LENGTH * (4.0 / 3.0 - (MASS_POLE * cos * cos) / TOTAL_MASS));
  const xAcc = temp - (POLE_MASS_LENGTH * thetaAcc * cos) / TOTAL_MASS;
  // "euler", upstream's default: positions advance on the *old* velocities.
  return [x + TAU * xDot, xDot + TAU * xAcc, theta + TAU * thetaDot, thetaDot + TAU * thetaAcc];
}

export function isFallen(state: CartPoleState): boolean {
  const [x, , theta] = state;
  return x < -X_THRESHOLD || x > X_THRESHOLD || theta < -THETA_THRESHOLD || theta > THETA_THRESHOLD;
}

export class CartPole implements Env<Float32Array, number> {
  readonly observationSpace = {
    kind: "box" as const,
    size: 4,
    low: [-2 * X_THRESHOLD, -Infinity, -2 * THETA_THRESHOLD, -Infinity],
    high: [2 * X_THRESHOLD, Infinity, 2 * THETA_THRESHOLD, Infinity],
  };
  readonly actionSpace = { kind: "discrete" as const, n: 2 };

  state: CartPoleState = [0, 0, 0, 0];
  private rand: () => number = mulberry32(0);
  private elapsed = 0;
  private finished = true;

  constructor(readonly timeLimit = TIME_LIMIT) {}

  /** Upstream draws each state variable uniformly from [−0.05, 0.05]. */
  reset(seed?: number): Float32Array {
    if (seed !== undefined) this.rand = mulberry32(seed);
    this.state = [0, 0, 0, 0].map(() => -0.05 + 0.1 * this.rand()) as CartPoleState;
    this.elapsed = 0;
    this.finished = false;
    return Float32Array.from(this.state);
  }

  /** Start from an exact state — for the fixture, and for nothing else. */
  resetTo(state: CartPoleState): Float32Array {
    this.state = [...state];
    this.elapsed = 0;
    this.finished = false;
    return Float32Array.from(this.state);
  }

  step(action: number): StepResult<Float32Array> {
    if (this.finished) {
      throw new Error("step() called on a finished episode — call reset() first");
    }
    if (action !== 0 && action !== 1) throw new Error(`invalid action ${action}`);
    this.state = cartPoleDynamics(this.state, action);
    this.elapsed++;
    const terminated = isFallen(this.state);
    const truncated = !terminated && this.elapsed >= this.timeLimit;
    this.finished = terminated || truncated;
    // The falling step pays 1 as well — see the header.
    return { observation: Float32Array.from(this.state), reward: 1, terminated, truncated };
  }
}
