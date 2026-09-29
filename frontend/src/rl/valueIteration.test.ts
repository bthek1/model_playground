import { describe, expect, it } from "vitest";

import type { FiniteMdp, Transition } from "./types";
import { optimalActions, valueIteration } from "./valueIteration";

/**
 * A two-state MDP small enough to solve by hand. From state 0: action 0 stays
 * (reward 0), action 1 moves to state 1 (reward 1). State 1 is terminal.
 * So Q*(0,1) = 1 and Q*(0,0) = γ·V*(0) = γ.
 */
const chain: FiniteMdp = {
  nStates: 2,
  nActions: 2,
  transitions(s, a): Transition[] {
    if (s === 1) return [{ p: 1, next: 1, reward: 0, terminated: true }];
    return a === 1
      ? [{ p: 1, next: 1, reward: 1, terminated: true }]
      : [{ p: 1, next: 0, reward: 0, terminated: false }];
  },
};

/** A coin flip: action 0 pays 1 half the time and ends; the rest it pays 0 and ends. */
const coin: FiniteMdp = {
  nStates: 1,
  nActions: 1,
  transitions: () => [
    { p: 0.5, next: 0, reward: 1, terminated: true },
    { p: 0.5, next: 0, reward: 0, terminated: true },
  ],
};

describe("valueIteration", () => {
  it("matches a hand-solved MDP", () => {
    const vi = valueIteration(chain, 0.9);
    expect(vi.q[0 * 2 + 1]).toBeCloseTo(1, 12);
    expect(vi.q[0 * 2 + 0]).toBeCloseTo(0.9, 12);
    expect(vi.v[0]).toBeCloseTo(1, 12);
    expect(vi.v[1]).toBe(0);
  });

  it("takes the expectation over a stochastic transition", () => {
    expect(valueIteration(coin, 0.5).v[0]).toBeCloseTo(0.5, 12);
  });

  it("uses a synchronous backup and stops at its tolerance", () => {
    const vi = valueIteration(chain, 0.9, { tolerance: 1e-6 });
    expect(vi.residual).toBeLessThanOrEqual(1e-6);
    expect(vi.iterations).toBeGreaterThan(1);
  });

  it("stops at the iteration cap rather than looping", () => {
    const vi = valueIteration(chain, 0.9, { tolerance: 0, maxIterations: 3 });
    expect(vi.iterations).toBe(3);
  });

  it("refuses γ = 1 and a negative γ", () => {
    expect(() => valueIteration(chain, 1)).toThrow(/γ < 1/);
    expect(() => valueIteration(chain, -0.1)).toThrow();
  });
});

describe("optimalActions", () => {
  it("accepts every action within the tolerance, so a check never pins a tie", () => {
    const q = Float64Array.from([0.5, 0.5, 0.1, /* state 1 */ 0, 0.2, 0.2 - 1e-9]);
    expect(optimalActions(q, 2, 3, 1e-6)).toEqual([[0, 1], [1, 2]]);
    expect(optimalActions(q, 2, 3, 0)).toEqual([[0, 1], [1]]);
  });
});
