// Behaviour cloning: supervised regression from state to action over the
// demonstrations, and the failure that is the page's whole point.
//
// Train on demonstrations that go round the obstacle **both ways** and a
// mean-squared-error policy learns their *average* — and the average of "up
// and round" and "down and round" is "straight on", into the obstacle. That is
// the motivation for action chunking and diffusion policies, which model the
// whole distribution of actions instead of its mean; this page names them and
// does not demonstrate them.
//
// A failure is only evidence with its control beside it. A policy that drives
// into the obstacle could equally be undertrained, badly scaled or fed a broken
// environment, so the same network, budget and seed are trained on a
// **unimodal** set (one side only) too, and that one must reach the target.
// `behaviourCloning.test.ts` asserts both — at the level of the action field,
// not only the rollout, so a rollout failing for another reason cannot satisfy
// it.
//
// It reuses `policyNet.ts` with a linear head (the MSE gradient is one line)
// and runs through the same `Stepper` seam as the RL learners: one step is one
// minibatch, one "episode" is one epoch, and the "return" it reports is the
// epoch's mean loss.

import { mulberry32, shuffle } from "@/lib/random";

import {
  demonstrations,
  type Demonstration,
  pairs,
} from "./demos";
import {
  handVelocity,
  inverseKinematics,
  makeScene,
  type Point,
  type ReachOutcome,
  Reacher2D,
  type ReacherScene,
} from "./envs/reacher2d";
import { Adam, backward, forward, initNet, type NetGrads, type NetParams } from "./policyNet";
import type { Stepper } from "./stepper";
import type { CloningRenderState, CloningRequest } from "./types";

/**
 * Input and action scaling, **fitted on the demonstrations only** and carried
 * with the policy, so a rollout cannot normalise differently from the fit —
 * `/tabular-classification`'s `design.ts` rule in its cheapest form.
 */
export interface Normaliser {
  xMean: Float32Array;
  xStd: Float32Array;
  yMean: Float32Array;
  yStd: Float32Array;
}

function columnStats(data: Float32Array, n: number, d: number): { mean: Float32Array; std: Float32Array } {
  const mean = new Float32Array(d);
  const std = new Float32Array(d);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += data[i * d + j] / n;
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) std[j] += (data[i * d + j] - mean[j]) ** 2 / n;
  // A column that never varies (a fixed target) must not divide by zero.
  for (let j = 0; j < d; j++) std[j] = Math.sqrt(std[j]) || 1;
  return { mean, std };
}

export function fitNormaliser(x: Float32Array, y: Float32Array, n: number): Normaliser {
  const xs = columnStats(x, n, 4);
  const ys = columnStats(y, n, 2);
  return { xMean: xs.mean, xStd: xs.std, yMean: ys.mean, yStd: ys.std };
}

function scale(data: Float32Array, n: number, d: number, mean: Float32Array, std: Float32Array): Float32Array {
  const out = new Float32Array(n * d);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) out[i * d + j] = (data[i * d + j] - mean[j]) / std[j];
  return out;
}

/** ½ · mean over rows of the squared error, summed over the action's two dimensions. */
export function mseLoss(net: NetParams, x: Float32Array, y: Float32Array, n: number): number {
  const out = forward(net, x, n).out;
  let loss = 0;
  for (let i = 0; i < out.length; i++) loss += (0.5 * (out[i] - y[i]) ** 2) / n;
  return loss;
}

export function mseGradient(net: NetParams, x: Float32Array, y: Float32Array, n: number): NetGrads {
  const cache = forward(net, x, n);
  const dOut = new Float32Array(cache.out.length);
  for (let i = 0; i < dOut.length; i++) dOut[i] = (cache.out[i] - y[i]) / n;
  return backward(net, cache, dOut);
}

/** A trained policy: the network and the scaling it was fitted with, together. */
export interface ClonedPolicy {
  net: NetParams;
  norm: Normaliser;
}

/** The policy's action for one observation, in joint-velocity units. */
export function act(policy: ClonedPolicy, obs: ArrayLike<number>): [number, number] {
  const { norm } = policy;
  const x = new Float32Array(4);
  for (let j = 0; j < 4; j++) x[j] = (obs[j] - norm.xMean[j]) / norm.xStd[j];
  const out = forward(policy.net, x, 1).out;
  return [out[0] * norm.yStd[0] + norm.yMean[0], out[1] * norm.yStd[1] + norm.yMean[1]];
}

export interface Rollout {
  path: Point[];
  outcome: ReachOutcome;
}

/** Run the policy from the scene's own start, deterministically — its mean action. */
export function rollout(policy: ClonedPolicy, scene: ReacherScene): Rollout {
  const env = new Reacher2D(scene);
  let obs = env.reset(undefined, 0);
  const path: Point[] = [env.hand];
  for (;;) {
    const r = env.step(Float32Array.from(act(policy, obs)));
    path.push(env.hand);
    obs = r.observation;
    if (r.terminated || r.truncated) break;
  }
  return { path, outcome: env.outcome ?? "stalled" };
}

export interface FieldArrow {
  at: Point;
  /** The hand velocity the policy would produce here. */
  v: Point;
}

/**
 * The policy's predicted hand velocity over a grid of workspace points, with
 * the target fixed. This is where the two modes visibly cancel — the difference
 * between "the arm crashed" and "here is why it crashed".
 */
export function actionField(
  policy: ClonedPolicy,
  scene: ReacherScene,
  bounds = { x0: -1.6, x1: 1.6, y0: 0.2, y1: 1.6 },
  steps = { nx: 13, ny: 7 },
): FieldArrow[] {
  const out: FieldArrow[] = [];
  for (let iy = 0; iy < steps.ny; iy++) {
    for (let ix = 0; ix < steps.nx; ix++) {
      const at = {
        x: bounds.x0 + ((bounds.x1 - bounds.x0) * ix) / (steps.nx - 1),
        y: bounds.y0 + ((bounds.y1 - bounds.y0) * iy) / (steps.ny - 1),
      };
      if (Math.hypot(at.x - scene.obstacle.x, at.y - scene.obstacle.y) < scene.radius) continue;
      const q = inverseKinematics(at);
      if (!q) continue;
      const [dq1, dq2] = act(policy, [q[0], q[1], scene.target.x, scene.target.y]);
      out.push({ at, v: handVelocity(q[0], q[1], dq1, dq2) });
    }
  }
  return out;
}

/** The hand's predicted direction at `at`, as an angle away from the obstacle's centre, in degrees. */
export function headingOffObstacle(policy: ClonedPolicy, scene: ReacherScene, at: Point): number {
  const q = inverseKinematics(at);
  if (!q) throw new Error("point out of reach");
  const [dq1, dq2] = act(policy, [q[0], q[1], scene.target.x, scene.target.y]);
  const v = handVelocity(q[0], q[1], dq1, dq2);
  const toObstacle = { x: scene.obstacle.x - at.x, y: scene.obstacle.y - at.y };
  const cos =
    (v.x * toObstacle.x + v.y * toObstacle.y) /
    (Math.hypot(v.x, v.y) * Math.hypot(toObstacle.x, toObstacle.y) || 1);
  return (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI;
}

/** Flatten the network for the wire; `unpackPolicy` is its inverse. */
export function packPolicy(policy: ClonedPolicy): Float32Array {
  const { net, norm } = policy;
  const parts = [net.w1, net.b1, net.w2, net.b2, norm.xMean, norm.xStd, norm.yMean, norm.yStd];
  const out = new Float32Array(parts.reduce((s, p) => s + p.length, 0));
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

export function unpackPolicy(flat: Float32Array, hidden: number): ClonedPolicy {
  let o = 0;
  const take = (n: number) => flat.slice(o, (o += n));
  const net: NetParams = {
    inputs: 4,
    hidden,
    outputs: 2,
    w1: take(4 * hidden),
    b1: take(hidden),
    w2: take(hidden * 2),
    b2: take(2),
  };
  return { net, norm: { xMean: take(4), xStd: take(4), yMean: take(2), yStd: take(2) } };
}

export const BATCH = 64;

/** Behaviour cloning as a `Stepper`: one step is one minibatch; one episode is one epoch. */
export class CloningRun implements Stepper {
  readonly scene: ReacherScene;
  readonly demos: Demonstration[];
  readonly policy: ClonedPolicy;
  readonly totalEpisodes: number;
  episode = 0;
  steps = 0;
  epsilonOverride: number | null = null;
  readonly epsilon = null;

  private readonly x: Float32Array;
  private readonly y: Float32Array;
  private readonly n: number;
  private readonly order: Int32Array;
  private readonly opt: Adam;
  private readonly rand: () => number;
  private cursor = 0;
  private epochLoss = 0;

  constructor(readonly req: CloningRequest) {
    this.scene = makeScene(req.obstacle);
    this.demos = demonstrations(this.scene, req.mix, req.demos, req.seed);
    const data = pairs(this.demos);
    if (data.n === 0) throw new Error("the demonstrator produced no pairs");
    const norm = fitNormaliser(data.x, data.y, data.n);
    this.x = scale(data.x, data.n, 4, norm.xMean, norm.xStd);
    this.y = scale(data.y, data.n, 2, norm.yMean, norm.yStd);
    this.n = data.n;
    this.policy = { net: initNet({ inputs: 4, hidden: req.hidden, outputs: 2 }, req.seed), norm };
    this.opt = new Adam(this.policy.net, req.lr);
    this.totalEpisodes = req.epochs;
    this.rand = mulberry32(req.seed ^ 0xb0c1);
    this.order = Int32Array.from({ length: this.n }, (_, i) => i);
    shuffle(this.order, this.rand);
  }

  get done(): boolean {
    return this.episode >= this.totalEpisodes;
  }

  step(): number | null {
    const b = Math.min(BATCH, this.n - this.cursor);
    const xb = new Float32Array(b * 4);
    const yb = new Float32Array(b * 2);
    for (let i = 0; i < b; i++) {
      const k = this.order[this.cursor + i];
      xb.set(this.x.subarray(k * 4, k * 4 + 4), i * 4);
      yb.set(this.y.subarray(k * 2, k * 2 + 2), i * 2);
    }
    this.epochLoss += mseLoss(this.policy.net, xb, yb, b) * b;
    this.opt.step(mseGradient(this.policy.net, xb, yb, b));
    this.cursor += b;
    this.steps++;
    if (this.cursor < this.n) return null;
    const loss = this.epochLoss / this.n;
    this.cursor = 0;
    this.epochLoss = 0;
    shuffle(this.order, this.rand);
    this.episode++;
    return loss;
  }

  render(): CloningRenderState {
    return { kind: "cloning", policy: packPolicy(this.policy), hidden: this.req.hidden };
  }
}
