// A toy 2-D reaching task: a two-link arm, a target, and a circular obstacle
// sitting on the straight line between where the hand starts and where it has
// to go. The environment behaviour cloning's multimodality failure is shown on
// (#54) — small enough to train on in a tab in seconds, and shaped so there are
// exactly two good ways round.
//
// Deliberately a toy, and the page says so: this is not `lerobot/pusht` and it
// is not a simulator. What it has to get right is geometry, so the geometry is
// pure and tested:
//
//   elbow = L₁ (cos q₁, sin q₁)
//   hand  = elbow + L₂ (cos(q₁ + q₂), sin(q₁ + q₂))
//
// Two simplifications, both stated on the page rather than hidden:
//
//   - **Only the hand collides.** The links pass through the obstacle. A full
//     link-collision check would change nothing about the lesson and would make
//     the "which way round" geometry depend on the elbow as well as the hand.
//   - **Actions are joint velocities, clipped**, integrated with one Euler step.
//     No dynamics, no inertia: the policy's output is where the arm goes.

import { gaussian, mulberry32 } from "@/lib/random";

import type { Env, StepResult } from "../types";

export const L1 = 1;
export const L2 = 1;
/** Seconds per step. */
export const DT = 0.05;
/** Joint speed limit, rad/s — a policy's output is clipped to it. */
export const MAX_JOINT_SPEED = 2.5;
/** The hand is "there" within this distance of the target. */
export const REACH_TOLERANCE = 0.08;
export const TIME_LIMIT = 200;

export interface Point {
  x: number;
  y: number;
}

/** Where the scene's pieces are. The obstacle's offset is a SELECT control. */
export interface ReacherScene {
  start: Point;
  target: Point;
  obstacle: Point;
  radius: number;
}

/**
 * The page's scene: the hand starts on the left, the target is on the right,
 * and the obstacle sits on the line between them — shifted perpendicular to it
 * by `offset`. At 0 the two ways round are equally long; near ±radius one way is
 * nearly straight, which is the near-degenerate case the page lets you try.
 */
export function makeScene(offset = 0): ReacherScene {
  return {
    start: { x: -1.2, y: 0.8 },
    target: { x: 1.2, y: 0.8 },
    obstacle: { x: 0, y: 0.9 + offset },
    radius: 0.3,
  };
}

export function forwardKinematics(q1: number, q2: number): { elbow: Point; hand: Point } {
  const elbow = { x: L1 * Math.cos(q1), y: L1 * Math.sin(q1) };
  const hand = {
    x: elbow.x + L2 * Math.cos(q1 + q2),
    y: elbow.y + L2 * Math.sin(q1 + q2),
  };
  return { elbow, hand };
}

/**
 * Inverse kinematics, one fixed branch (q₂ < 0) so a path never flips the
 * elbow mid-way. For this scene that branch puts the elbow *below* the base at
 * the start — which is right, just not "elbow-up". The *demonstrator* and the
 * drawing use it; the policy never sees it — learning that map from
 * demonstrations is part of what the policy has to do.
 */
export function inverseKinematics(p: Point): [number, number] | null {
  const d2 = p.x * p.x + p.y * p.y;
  const c2 = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2);
  if (c2 < -1 || c2 > 1) return null;
  const q2 = -Math.acos(c2);
  const q1 = Math.atan2(p.y, p.x) - Math.atan2(L2 * Math.sin(q2), L1 + L2 * Math.cos(q2));
  return [q1, q2];
}

/** The hand's velocity for a joint velocity: J · q̇. */
export function handVelocity(q1: number, q2: number, dq1: number, dq2: number): Point {
  const s1 = Math.sin(q1);
  const c1 = Math.cos(q1);
  const s12 = Math.sin(q1 + q2);
  const c12 = Math.cos(q1 + q2);
  return {
    x: (-L1 * s1 - L2 * s12) * dq1 - L2 * s12 * dq2,
    y: (L1 * c1 + L2 * c12) * dq1 + L2 * c12 * dq2,
  };
}

/** The joint velocity that moves the hand at `v`: J⁻¹ · v. Null near a singularity. */
export function jointVelocity(q1: number, q2: number, v: Point): [number, number] | null {
  const a = -L1 * Math.sin(q1) - L2 * Math.sin(q1 + q2);
  const b = -L2 * Math.sin(q1 + q2);
  const c = L1 * Math.cos(q1) + L2 * Math.cos(q1 + q2);
  const d = L2 * Math.cos(q1 + q2);
  const det = a * d - b * c;
  if (Math.abs(det) < 1e-6) return null;
  return [(d * v.x - b * v.y) / det, (-c * v.x + a * v.y) / det];
}

export type ReachOutcome = "reached" | "collided" | "stalled";

/** Observation: `[q1, q2, targetX, targetY]`. */
export class Reacher2D implements Env<Float32Array, Float32Array> {
  readonly observationSpace = {
    kind: "box" as const,
    size: 4,
    low: [-Math.PI, -Math.PI, -2, -2],
    high: [Math.PI, Math.PI, 2, 2],
  };
  readonly actionSpace = {
    kind: "box" as const,
    size: 2,
    low: [-MAX_JOINT_SPEED, -MAX_JOINT_SPEED],
    high: [MAX_JOINT_SPEED, MAX_JOINT_SPEED],
  };

  q: [number, number] = [0, 0];
  target: Point;
  outcome: ReachOutcome | null = null;
  private rand: () => number = mulberry32(0);
  private elapsed = 0;
  private finished = true;

  constructor(
    readonly scene: ReacherScene,
    readonly timeLimit = TIME_LIMIT,
  ) {
    this.target = scene.target;
  }

  get hand(): Point {
    return forwardKinematics(this.q[0], this.q[1]).hand;
  }

  /** Start near `scene.start`, jittered by `jitter`, with the target jittered too. */
  reset(seed?: number, jitter = 0.05): Float32Array {
    if (seed !== undefined) this.rand = mulberry32(seed);
    const s = {
      x: this.scene.start.x + jitter * gaussian(this.rand),
      y: this.scene.start.y + jitter * gaussian(this.rand),
    };
    this.target = {
      x: this.scene.target.x + jitter * gaussian(this.rand),
      y: this.scene.target.y + jitter * gaussian(this.rand),
    };
    const q = inverseKinematics(s);
    if (!q) throw new Error("start is out of reach");
    this.q = q;
    this.elapsed = 0;
    this.finished = false;
    this.outcome = null;
    return this.observe();
  }

  observe(): Float32Array {
    return Float32Array.from([this.q[0], this.q[1], this.target.x, this.target.y]);
  }

  step(action: Float32Array): StepResult<Float32Array> {
    if (this.finished) throw new Error("step() called on a finished episode — call reset() first");
    const clip = (v: number) => Math.max(-MAX_JOINT_SPEED, Math.min(MAX_JOINT_SPEED, v));
    this.q = [this.q[0] + DT * clip(action[0]), this.q[1] + DT * clip(action[1])];
    this.elapsed++;
    const hand = this.hand;
    const { obstacle, radius } = this.scene;
    const collided = Math.hypot(hand.x - obstacle.x, hand.y - obstacle.y) < radius;
    const reached = Math.hypot(hand.x - this.target.x, hand.y - this.target.y) < REACH_TOLERANCE;
    const terminated = collided || reached;
    const truncated = !terminated && this.elapsed >= this.timeLimit;
    this.finished = terminated || truncated;
    if (collided) this.outcome = "collided";
    else if (reached) this.outcome = "reached";
    else if (truncated) this.outcome = "stalled";
    return { observation: this.observe(), reward: reached ? 1 : 0, terminated, truncated };
  }
}
