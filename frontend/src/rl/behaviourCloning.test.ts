import { describe, expect, it, vi } from "vitest";

import * as policyNet from "./policyNet";
import {
  act,
  CloningRun,
  fitNormaliser,
  headingOffObstacle,
  mseGradient,
  mseLoss,
  packPolicy,
  rollout,
  unpackPolicy,
} from "./behaviourCloning";
import { demonstrations, pairs } from "./demos";
import { makeScene } from "./envs/reacher2d";
import { initNet } from "./policyNet";
import type { CloningRequest } from "./types";

function request(over: Partial<CloningRequest> = {}): CloningRequest {
  return {
    algorithm: "behaviour-cloning",
    mix: "both",
    demos: 20,
    obstacle: 0,
    hidden: 64,
    epochs: 100,
    lr: 0.01,
    seed: 1,
    ...over,
  };
}

function trained(req: CloningRequest): CloningRun {
  const run = new CloningRun(req);
  while (!run.done) run.step();
  return run;
}

/** A point on the straight line from start to target, before the obstacle — between the modes. */
const BETWEEN = { x: -0.6, y: 0.85 };

describe("the regression", () => {
  it("has the gradient of its own loss, at a generic point", () => {
    const net = initNet({ inputs: 4, hidden: 5, outputs: 2 }, 3);
    for (let i = 0; i < net.b1.length; i++) net.b1[i] = 0.1 * (i - 2); // off zero
    net.b2[0] = 0.2;
    const x = Float32Array.from({ length: 12 }, (_, i) => Math.cos(i * 0.7));
    const y = Float32Array.from({ length: 6 }, (_, i) => Math.sin(i * 1.1));
    const g = mseGradient(net, x, y, 3);
    const h = 1e-3;
    for (const key of ["w1", "b1", "w2", "b2"] as const) {
      for (let i = 0; i < net[key].length; i++) {
        const orig = net[key][i];
        net[key][i] = orig + h;
        const up = mseLoss(net, x, y, 3);
        net[key][i] = orig - h;
        const down = mseLoss(net, x, y, 3);
        net[key][i] = orig;
        expect(g[key][i], `${key}[${i}]`).toBeCloseTo((up - down) / (2 * h), 3);
      }
    }
  });

  it("runs its arithmetic through policyNet's injected matmul", () => {
    const spy = vi.spyOn(policyNet, "forward");
    const run = new CloningRun(request({ epochs: 1 }));
    run.step();
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });

  it("fits its scaling on the demonstrations, and carries it with the policy", () => {
    // Assert the stored statistics, not just the outcome: a rollout that
    // normalised differently from the fit would still produce *an* action.
    const run = new CloningRun(request({ epochs: 1 }));
    const data = pairs(demonstrations(makeScene(0), "both", 20, 1));
    const fitted = fitNormaliser(data.x, data.y, data.n);
    expect(Array.from(run.policy.norm.xMean)).toEqual(Array.from(fitted.xMean));
    expect(Array.from(run.policy.norm.yStd)).toEqual(Array.from(fitted.yStd));
    // And it survives the trip to the page intact.
    const back = unpackPolicy(packPolicy(run.policy), 64);
    expect(Array.from(back.norm.xStd)).toEqual(Array.from(run.policy.norm.xStd));
    expect(act(back, [0.1, -1, 1.2, 0.8])).toEqual(act(run.policy, [0.1, -1, 1.2, 0.8]));
  });

  it("drives the loss down on the unimodal set and reaches the target", () => {
    const run = new CloningRun(request({ mix: "one" }));
    const losses: number[] = [];
    while (!run.done) {
      const l = run.step();
      if (l !== null) losses.push(l);
    }
    expect(losses[losses.length - 1]).toBeLessThan(losses[0] * 0.5);
    expect(rollout(run.policy, run.scene).outcome).toBe("reached");
  });
});

describe("the multimodality failure, and its control", () => {
  // The claim, at the level of the action field rather than the outcome, so it
  // cannot be satisfied by a rollout that fails for some other reason. Same
  // scene, network, budget and seed; only the demonstration mix differs.
  // Measured over seeds 1–6: bimodal heading 0–10° off the obstacle and 5 of 6
  // rollouts collide; unimodal 16–39° and 6 of 6 reach.

  it("points at the obstacle between the modes when both sides were shown, and not when one was", () => {
    for (const seed of [1, 2, 3, 4]) {
      const both = trained(request({ mix: "both", seed }));
      const one = trained(request({ mix: "one", seed }));
      expect(headingOffObstacle(both.policy, both.scene, BETWEEN), `bimodal, seed ${seed}`).toBeLessThan(12);
      expect(headingOffObstacle(one.policy, one.scene, BETWEEN), `unimodal, seed ${seed}`).toBeGreaterThan(14);
    }
  }, 30_000);

  it("collides when bimodal and reaches when unimodal, at seed 1", () => {
    const both = trained(request({ mix: "both" }));
    const one = trained(request({ mix: "one" }));
    expect(rollout(both.policy, both.scene).outcome).toBe("collided");
    expect(rollout(one.policy, one.scene).outcome).toBe("reached");
  });
});
