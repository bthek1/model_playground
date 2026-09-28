import { describe, expect, it } from "vitest";

import { mulberry32 } from "@/lib/random";

import { GridWorld, LEFT, RIGHT } from "./envs/gridWorld";
import {
  EPSILON_FLOOR,
  epsilonAt,
  greedyRollout,
  QLearningRun,
  QTable,
  qUpdate,
} from "./qLearning";
import type { QLearningRequest } from "./types";
import { optimalActions, valueIteration } from "./valueIteration";

function request(over: Partial<QLearningRequest> = {}): QLearningRequest {
  return {
    algorithm: "q-learning",
    map: "4x4",
    slippery: false,
    alpha: 0.5,
    gamma: 0.95,
    epsilon: 1,
    decay: true,
    episodes: 1000,
    seed: 1,
    ...over,
  };
}

function train(req: QLearningRequest): { run: QLearningRun; returns: number[] } {
  const run = new QLearningRun(req);
  const returns: number[] = [];
  while (!run.done) {
    const r = run.step();
    if (r !== null) returns.push(r);
  }
  return { run, returns };
}

describe("valueIteration", () => {
  it("finds the textbook value of the deterministic 4×4 start", () => {
    // Six steps to the goal, reward 1 on the sixth: γ⁵.
    const vi = valueIteration(new GridWorld({ slippery: false }), 0.95);
    expect(vi.v[0]).toBeCloseTo(0.95 ** 5, 9);
    expect(vi.residual).toBeLessThan(1e-10);
  });

  it("gives terminal states no future value", () => {
    const env = new GridWorld({ slippery: true });
    const vi = valueIteration(env, 0.99);
    for (let s = 0; s < env.nStates; s++) {
      if (env.isTerminal(s)) expect(vi.v[s]).toBe(0);
    }
  });

  it("refuses a discount that cannot converge", () => {
    expect(() => valueIteration(new GridWorld(), 1)).toThrow(/γ < 1/);
  });
});

describe("Q-learning against the exact Q*", () => {
  // The load-bearing test of the module. A wrong update rule — no discount, a
  // bootstrap through the terminal state, the max over the wrong row — still
  // finds the goal on a 4×4 grid and still draws a rising curve. Only the
  // exact answer can tell the difference.

  it("converges to Q* on the deterministic grid, entry for entry", () => {
    // ε = 1 throughout: Q-learning is off-policy, so a uniformly random
    // behaviour policy still learns the *optimal* values — and visits every
    // (state, action) often enough for every entry to converge.
    const req = request({ epsilon: 1, decay: false, episodes: 4000, alpha: 0.5 });
    const { run } = train(req);
    const vi = valueIteration(run.env, req.gamma);
    for (let i = 0; i < vi.q.length; i++) {
      expect(Math.abs(run.table.q[i] - vi.q[i])).toBeLessThan(1e-3);
    }
  });

  it("learns an optimal greedy policy under the page's defaults", () => {
    const req = request();
    const { run } = train(req);
    const vi = valueIteration(run.env, req.gamma);
    const optimal = optimalActions(vi.q, run.env.nStates, 4, 1e-6);
    for (let s = 0; s < run.env.nStates; s++) {
      if (run.env.isTerminal(s)) continue;
      expect(optimal[s], `state ${s}`).toContain(run.table.greedy(s));
    }
    expect(greedyRollout(new GridWorld({ slippery: false }), run.table, 0).return).toBe(1);
  });

  /** Random-behaviour Q-learning with α = 1/n^0.8 per pair; returns the worst error. */
  function slipperyError(agentSeed: number, envSeed: number): number {
    const env = new GridWorld({ slippery: true });
    const gamma = 0.9;
    const table = new QTable(env.nStates, 4);
    const visits = new Uint32Array(env.nStates * 4);
    const rand = mulberry32(agentSeed);
    env.reset(envSeed);
    for (let episode = 0; episode < 200000; episode++) {
      let s = episode === 0 ? 0 : env.reset();
      for (;;) {
        const a = Math.floor(rand() * 4);
        const r = env.step(a);
        const n = ++visits[s * 4 + a];
        qUpdate(table, s, a, r.reward, r.observation, r.terminated, 1 / n ** 0.8, gamma);
        if (r.terminated || r.truncated) break;
        s = r.observation;
      }
    }
    const vi = valueIteration(env, gamma);
    let worst = 0;
    for (let i = 0; i < vi.q.length; i++) worst = Math.max(worst, Math.abs(table.q[i] - vi.q[i]));
    return worst;
  }

  it("converges to Q* on the slippery grid under the Robbins–Monro step sizes", () => {
    // A constant α on a stochastic MDP leaves noise of order α on every entry,
    // and rarely-visited pairs are still climbing from zero — a tolerance loose
    // enough to pass that would pass a wrong update too. The convergence
    // theorem's own condition is a per-pair step size with Σα = ∞ and Σα² < ∞,
    // so this drives `qUpdate` with α = 1/n^0.8 and asks for the real answer.
    // The slowest entry is always state 14, beside the goal: rare under a
    // random walk, and its reward is a one-in-three slip.
    expect(slipperyError(7 ^ 0x51ed27, 7)).toBeLessThan(0.02);
  });

  it("learns the wrong values if the agent and the environment share a stream", () => {
    // Found by the test above, which first seeded both with 7. Each action draw
    // and the slip that follows it are then the *same number*, so which way the
    // agent slips depends on which action it chose — dynamics the model does
    // not have. Q-learning converged confidently to values four times too high
    // (Q(0, ·) ≈ 0.29 against Q* ≈ 0.07), with a rising curve and no error.
    // `QLearningRun` derives the agent's seed as `seed ^ 0x51ed27` for this.
    expect(slipperyError(7, 7)).toBeGreaterThan(0.15);
  });

  it("bootstraps through a truncation, and not through a termination", () => {
    const t = new QTable(2, 1);
    t.q[1] = 10; // Q(next) = 10
    qUpdate(t, 0, 0, 0, 1, false, 1, 0.5);
    expect(t.q[0]).toBe(5); // truncated or mid-episode: the future counts
    qUpdate(t, 0, 0, 0, 1, true, 1, 0.5);
    expect(t.q[0]).toBe(0); // terminated: it does not
  });
});

describe("exploration", () => {
  it("breaks ties toward the first action, as np.argmax does", () => {
    const t = new QTable(1, 4);
    expect(t.greedy(0)).toBe(LEFT);
    t.q[2] = 0.1;
    t.q[3] = 0.1;
    expect(t.greedy(0)).toBe(RIGHT);
  });

  it("never reaches the goal at ε = 0 from a cold table, and does at ε > 0", () => {
    // The page claims this beside the ε control. Ties go LEFT, into the wall,
    // and a greedy agent learns nothing from a wall it already valued at 0.
    const stuck = train(request({ epsilon: 0, episodes: 500 }));
    expect(stuck.returns.reduce((a, b) => a + b, 0)).toBe(0);
    expect(stuck.run.position).toBe(0);

    const exploring = train(request({ epsilon: 1, episodes: 500 }));
    expect(exploring.returns.reduce((a, b) => a + b, 0)).toBeGreaterThan(50);
  });

  it("gets stuck with a small constant ε too, which is why the schedule exists", () => {
    const constant = train(request({ epsilon: 0.2, decay: false, episodes: 2000 }));
    expect(constant.returns.reduce((a, b) => a + b, 0)).toBe(0);
  });

  it("decays linearly to the floor over 80% of the run, then holds", () => {
    expect(epsilonAt(0, 100, 1, true)).toBe(1);
    expect(epsilonAt(40, 100, 1, true)).toBeCloseTo(1 - 0.5 * (1 - EPSILON_FLOOR));
    expect(epsilonAt(80, 100, 1, true)).toBeCloseTo(EPSILON_FLOOR);
    expect(epsilonAt(99, 100, 1, true)).toBeCloseTo(EPSILON_FLOOR);
    expect(epsilonAt(50, 100, 0.3, false)).toBe(0.3);
    // A start below the floor is never raised to it: ε = 0 means zero.
    expect(epsilonAt(50, 100, 0, true)).toBe(0);
  });

  it("lets a live ε override the schedule without restarting the run", () => {
    const run = new QLearningRun(request({ episodes: 10 }));
    run.step();
    const steps = run.steps;
    run.epsilonOverride = 0;
    expect(run.epsilon).toBe(0);
    expect(run.steps).toBe(steps);
    run.epsilonOverride = null;
    expect(run.epsilon).toBeGreaterThan(0.5);
  });
});

describe("QLearningRun", () => {
  it("is deterministic at a seed", () => {
    const a = train(request({ slippery: true, seed: 5, episodes: 300 }));
    const b = train(request({ slippery: true, seed: 5, episodes: 300 }));
    expect(a.returns).toEqual(b.returns);
    expect(Array.from(a.run.table.q)).toEqual(Array.from(b.run.table.q));
  });

  it("hands the canvas a copy of the table, not the one it is still writing", () => {
    const run = new QLearningRun(request());
    const snapshot = run.render();
    for (let i = 0; i < 200; i++) run.step();
    expect(Array.from(snapshot.q).every((v) => v === 0)).toBe(true);
  });
});
