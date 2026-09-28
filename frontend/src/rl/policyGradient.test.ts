import { describe, expect, it } from "vitest";

import { mulberry32 } from "@/lib/random";

import {
  advantages,
  discountedReturns,
  PG_DEFAULTS,
  PolicyGradientRun,
  policyGradient,
  policyLoss,
  standardise,
  valueGradient,
  valueLoss,
} from "./policyGradient";
import { initNet, type NetGrads, type NetParams } from "./policyNet";
import type { PolicyGradientRequest } from "./types";

/** A net at a generic point — every bias nonzero, weights not tiny. */
function genericNet(outputs: number, seed: number): NetParams {
  const net = initNet({ inputs: 4, hidden: 6, outputs }, seed);
  const rand = mulberry32(seed + 100);
  for (const k of ["b1", "b2"] as const) {
    for (let i = 0; i < net[k].length; i++) net[k][i] = rand() - 0.5;
  }
  return net;
}

/** A short fixed "episode": 5 observations, actions, and weights of both signs. */
const OBS = Float32Array.from({ length: 20 }, (_, i) => Math.sin(i * 1.3) * 0.8);
const ACTIONS = [0, 1, 1, 0, 1];
const WEIGHTS = [3.1, -1.4, 0.7, 2.2, -0.5];

/** Central differences over every parameter against the analytic gradient. */
function checkGradient(net: NetParams, loss: () => number, grads: NetGrads) {
  const h = 1e-3;
  for (const key of ["w1", "b1", "w2", "b2"] as const) {
    const p = net[key];
    for (let i = 0; i < p.length; i++) {
      const orig = p[i];
      p[i] = orig + h;
      const up = loss();
      p[i] = orig - h;
      const down = loss();
      p[i] = orig;
      const numeric = (up - down) / (2 * h);
      expect(grads[key][i], `${key}[${i}]`).toBeCloseTo(numeric, 3);
    }
  }
}

describe("the policy gradient, by finite differences", () => {
  // A wrong policy gradient still produces a rising curve: a sign error in the
  // weight, or the wrong 1/T, is absorbed into the learning rate. So each
  // learner's gradient is checked against its own loss, at a generic point.

  it("REINFORCE: weighting by the return", () => {
    const net = genericNet(2, 1);
    const returns = discountedReturns([1, 1, 1, 1, 1], 0.9);
    const grads = policyGradient(net, OBS, ACTIONS, returns);
    checkGradient(net, () => policyLoss(net, OBS, ACTIONS, returns), grads);
  });

  it("Actor-Critic: weighting by a signed advantage", () => {
    const net = genericNet(2, 2);
    const grads = policyGradient(net, OBS, ACTIONS, WEIGHTS);
    checkGradient(net, () => policyLoss(net, OBS, ACTIONS, WEIGHTS), grads);
  });

  it("the critic's own gradient", () => {
    const net = genericNet(1, 3);
    const targets = [0.5, -1.2, 2.0, 0.1, 1.7];
    const grads = valueGradient(net, OBS, targets);
    checkGradient(net, () => valueLoss(net, OBS, targets), grads);
  });

  it("pushes the taken action up when its weight is positive, and down when negative", () => {
    // The sign the finite-difference check could only confirm, stated directly.
    const net = genericNet(2, 4);
    const obs = OBS.subarray(0, 4);
    const before = policyLoss(net, obs, [1], [1]);
    const g = policyGradient(net, obs, [1], [1]);
    for (const key of ["w1", "b1", "w2", "b2"] as const) {
      for (let i = 0; i < net[key].length; i++) net[key][i] -= 0.1 * g[key][i];
    }
    expect(policyLoss(net, obs, [1], [1])).toBeLessThan(before);
  });
});

describe("returns and advantages", () => {
  it("discounts backward from the end of the episode", () => {
    expect(Array.from(discountedReturns([1, 1, 1], 0.5))).toEqual([1.75, 1.5, 1]);
  });

  it("standardises to zero mean and unit variance", () => {
    const s = standardise(Float64Array.from([1, 2, 3, 4]));
    expect(s.reduce((a, b) => a + b)).toBeCloseTo(0, 12);
    expect(s.reduce((a, b) => a + b * b, 0) / 4).toBeCloseTo(1, 12);
  });

  it("gives zero advantage everywhere when the critic is exact", () => {
    // A property "the loss falls" cannot express. Along a fixed trajectory the
    // exact values are the discounted returns; every TD error is then zero, and
    // so is every GAE advantage, at any λ.
    const rewards = [1, 1, 1, 1, 1, 1];
    const gamma = 0.97;
    const exact = discountedReturns(rewards, gamma);
    for (const lambda of [0, 0.9, 1]) {
      const { advantages: adv, targets } = advantages(rewards, exact, 0, true, gamma, lambda);
      for (const a of adv) expect(a).toBeCloseTo(0, 12);
      expect(Array.from(targets)).toEqual(Array.from(exact));
    }
  });

  it("bootstraps through a truncation, and not through a fall", () => {
    const values = [0, 0];
    const fell = advantages([1, 1], values, 10, true, 0.5, 0);
    const cut = advantages([1, 1], values, 10, false, 0.5, 0);
    expect(fell.advantages[1]).toBe(1); // no future after a fall
    expect(cut.advantages[1]).toBe(1 + 0.5 * 10); // the pole was still up
  });

  it("reduces to the Monte Carlo return minus the baseline at λ = 1", () => {
    const rewards = [1, 0, 2];
    const values = [0.3, -0.2, 0.9];
    const gamma = 0.8;
    const g = discountedReturns(rewards, gamma);
    const { advantages: adv } = advantages(rewards, values, 0, true, gamma, 1);
    for (let t = 0; t < 3; t++) expect(adv[t]).toBeCloseTo(g[t] - values[t], 12);
  });
});

function request(algorithm: PolicyGradientRequest["algorithm"], seed: number, over = {}): PolicyGradientRequest {
  return {
    algorithm,
    env: "cartpole",
    hidden: PG_DEFAULTS.hidden,
    gamma: PG_DEFAULTS.gamma,
    episodes: PG_DEFAULTS.episodes,
    normalise: false,
    seed,
    ...PG_DEFAULTS[algorithm],
    ...over,
  };
}

function finalMean(req: PolicyGradientRequest): { final: number; run: PolicyGradientRun } {
  const run = new PolicyGradientRun(req);
  const returns: number[] = [];
  while (!run.done) {
    const r = run.step();
    if (r !== null) returns.push(r);
  }
  const tail = returns.slice(-50);
  return { final: tail.reduce((a, b) => a + b, 0) / tail.length, run };
}

describe("PolicyGradientRun", () => {
  it("is deterministic at a seed", () => {
    const a = finalMean(request("reinforce", 3, { episodes: 30 }));
    const b = finalMean(request("reinforce", 3, { episodes: 30 }));
    expect(a.final).toBe(b.final);
    expect(Array.from(a.run.policy.w1)).toEqual(Array.from(b.run.policy.w1));
  });

  it("reports where the variance went: |A| is far smaller than |G|", () => {
    const rf = finalMean(request("reinforce", 1, { episodes: 60 })).run.render();
    const ac = finalMean(request("actor-critic", 1, { episodes: 60 })).run.render();
    expect(rf.diagnostics?.criticLoss).toBeNull();
    expect(ac.diagnostics?.criticLoss).not.toBeNull();
    expect(ac.diagnostics!.meanAbsWeight).toBeLessThan(rf.diagnostics!.meanAbsWeight);
  });

  it("narrows the spread of final returns across seeds — without asserting a winner per seed", () => {
    // The claim the page makes is about variance across seeds, not that
    // Actor-Critic beats REINFORCE on any given one — that is not a property RL
    // promises, and pinning it would fail for reasons unrelated to the code.
    const seeds = [1, 2, 3, 4];
    const spread = (algorithm: PolicyGradientRequest["algorithm"]) => {
      const finals = seeds.map((s) => finalMean(request(algorithm, s)).final);
      const mean = finals.reduce((a, b) => a + b) / finals.length;
      return Math.sqrt(finals.reduce((a, b) => a + (b - mean) ** 2, 0) / finals.length);
    };
    expect(spread("actor-critic")).toBeLessThan(spread("reinforce"));
  }, 30_000);
});
