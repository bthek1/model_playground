// The demonstrator: a scripted path generator, not a learned expert — and the
// page shows the script's paths before anything trains, because pretending
// otherwise would be the page's own hidden assumption.
//
// Each demonstration goes from the start round **one side** of the obstacle to
// the target: towards a waypoint just clear of the obstacle on the chosen side,
// then to the target, at a fixed hand speed. The hand velocity is turned into
// joint velocities with the Jacobian inverse — the expert may use inverse
// kinematics; the policy is never given it.
//
// **The mix is the whole experiment.** A bimodal set (both sides) and a
// unimodal set (one side) are generated from the same scene, the same noise
// and the same seed, so the only difference between the two policies trained
// on them is the multimodality. `demos.test.ts` checks the generator before any
// learning does: a `side: "above"` path never crosses below the obstacle's
// centre line, and the reverse — if the scripted paths were not genuinely
// bimodal, the page would demonstrate nothing and still look clean.

import { gaussian, mulberry32 } from "@/lib/random";

import {
  forwardKinematics,
  inverseKinematics,
  jointVelocity,
  MAX_JOINT_SPEED,
  type Point,
  Reacher2D,
  type ReacherScene,
} from "./envs/reacher2d";

export type Side = "above" | "below";
export type DemoMix = "both" | "one";

/** Hand speed the demonstrator moves at, m/s. */
export const DEMO_SPEED = 0.9;
/** How far clear of the obstacle's edge the waypoint sits. */
export const CLEARANCE = 0.12;

export interface Demonstration {
  side: Side;
  /** `T × 4` observations and `T × 2` actions, row-major. */
  obs: Float32Array;
  actions: Float32Array;
  /** The hand's path, for drawing. */
  path: Point[];
  reached: boolean;
}

/** The waypoint on `side` of the obstacle, perpendicular to the start→target line. */
export function waypoint(scene: ReacherScene, side: Side): Point {
  const dx = scene.target.x - scene.start.x;
  const dy = scene.target.y - scene.start.y;
  const len = Math.hypot(dx, dy);
  // The left-hand normal of start→target points "above" for this scene.
  const nx = -dy / len;
  const ny = dx / len;
  const sign = side === "above" ? 1 : -1;
  const d = scene.radius + CLEARANCE;
  return { x: scene.obstacle.x + sign * nx * d, y: scene.obstacle.y + sign * ny * d };
}

/** Which sides a mix of `n` demonstrations takes: alternating for "both". */
export function sidesFor(mix: DemoMix, n: number, oneSide: Side = "above"): Side[] {
  return Array.from({ length: n }, (_, i) =>
    mix === "one" ? oneSide : i % 2 === 0 ? "above" : "below",
  );
}

/** One scripted demonstration. */
export function demonstrate(
  scene: ReacherScene,
  side: Side,
  rand: () => number,
  noise = 0.05,
): Demonstration {
  const env = new Reacher2D(scene);
  env.reset(undefined, 0);
  // The start and target jitter come from the demonstrator's own stream, so
  // each demonstration in a set starts somewhere slightly different.
  const jitter = () => 0.05 * gaussian(rand);
  const start = { x: scene.start.x + jitter(), y: scene.start.y + jitter() };
  const q = inverseKinematics(start);
  if (q) env.q = q;
  env.target = { x: scene.target.x + jitter(), y: scene.target.y + jitter() };
  const obs: number[] = [];
  const actions: number[] = [];
  const path: Point[] = [env.hand];
  const via = waypoint(scene, side);
  let goal: Point = via;

  for (let t = 0; t < env.timeLimit; t++) {
    const hand = env.hand;
    // Turn for the target only once the hand is *at* the waypoint (or past it).
    // Turning 0.15 m early cut the corner across the top of the obstacle —
    // 0.29 m from its centre against a 0.30 radius — and three demonstrations
    // in twenty collided. From the waypoint itself the second leg clears the
    // obstacle by ~0.4 m on either side.
    if (goal === via && (Math.hypot(hand.x - via.x, hand.y - via.y) < 0.04 || hand.x > via.x)) {
      goal = env.target;
    }
    const dx = goal.x - hand.x;
    const dy = goal.y - hand.y;
    const dist = Math.hypot(dx, dy) || 1;
    const speed = Math.min(DEMO_SPEED, dist / 0.05);
    const v = { x: (dx / dist) * speed, y: (dy / dist) * speed };
    const dq = jointVelocity(env.q[0], env.q[1], v);
    if (!dq) break;
    const a = [dq[0] + noise * gaussian(rand), dq[1] + noise * gaussian(rand)].map((u) =>
      Math.max(-MAX_JOINT_SPEED, Math.min(MAX_JOINT_SPEED, u)),
    );
    obs.push(...env.observe());
    actions.push(...a);
    const r = env.step(Float32Array.from(a));
    path.push(env.hand);
    if (r.terminated || r.truncated) break;
  }
  return {
    side,
    obs: Float32Array.from(obs),
    actions: Float32Array.from(actions),
    path,
    reached: env.outcome === "reached",
  };
}

/** A seeded demonstration set. Pure: the page's preview and the worker build the same one. */
export function demonstrations(
  scene: ReacherScene,
  mix: DemoMix,
  count: number,
  seed: number,
): Demonstration[] {
  const rand = mulberry32(seed ^ 0xde705);
  return sidesFor(mix, count).map((side) => demonstrate(scene, side, rand));
}

/** Every (state, action) pair across a set, flattened for the regression. */
export function pairs(demos: readonly Demonstration[]): { x: Float32Array; y: Float32Array; n: number } {
  const n = demos.reduce((s, d) => s + d.actions.length / 2, 0);
  const x = new Float32Array(n * 4);
  const y = new Float32Array(n * 2);
  let i = 0;
  for (const d of demos) {
    x.set(d.obs, i * 4);
    y.set(d.actions, i * 2);
    i += d.actions.length / 2;
  }
  return { x, y, n };
}

/** The hand's position for an observation, for the drawing. */
export function handOf(obs: ArrayLike<number>): Point {
  return forwardKinematics(obs[0], obs[1]).hand;
}
