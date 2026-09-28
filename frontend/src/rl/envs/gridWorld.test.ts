import { describe, expect, it } from "vitest";

import { DOWN, GridWorld, LEFT, MAPS, RIGHT, SUCCESS_RATE, TIME_LIMITS, UP } from "./gridWorld";

describe("GridWorld — transcribed from Gymnasium's FrozenLake-v1", () => {
  it("carries upstream's maps and time limits verbatim", () => {
    // A map with one hole moved is a different MDP with different optimal
    // values, and nothing on screen would say so.
    expect(MAPS["4x4"]).toEqual(["SFFF", "FHFH", "FFFH", "HFFG"]);
    expect(MAPS["8x8"][5]).toBe("FHHFFFHF");
    expect(MAPS["8x8"][7]).toBe("FFFHFFFG");
    expect(TIME_LIMITS).toEqual({ "4x4": 100, "8x8": 200 });
    expect([LEFT, DOWN, RIGHT, UP]).toEqual([0, 1, 2, 3]);
  });

  it("gives a transition distribution that sums to 1 for every (state, action)", () => {
    for (const slippery of [false, true]) {
      for (const map of ["4x4", "8x8"] as const) {
        const env = new GridWorld({ map, slippery });
        for (let s = 0; s < env.nStates; s++) {
          for (let a = 0; a < 4; a++) {
            const total = env.transitions(s, a).reduce((sum, t) => sum + t.p, 0);
            expect(total).toBeCloseTo(1, 12);
          }
        }
      }
    }
  });

  it("slips perpendicular, never backwards", () => {
    // State 5 is a hole on the 4×4; state 9 has four open neighbours.
    const env = new GridWorld({ slippery: true });
    const outcomes = env.transitions(9, DOWN);
    const byNext = new Map(outcomes.map((t) => [t.next, t.p]));
    expect(byNext.get(13)).toBeCloseTo(SUCCESS_RATE); // down, as intended
    expect(byNext.get(8)).toBeCloseTo((1 - SUCCESS_RATE) / 2); // left
    expect(byNext.get(10)).toBeCloseTo((1 - SUCCESS_RATE) / 2); // right
    expect(byNext.has(5)).toBe(false); // up — backwards — is impossible
  });

  it("leaves the agent in place when it walks into a wall", () => {
    const env = new GridWorld({ slippery: false });
    env.reset(0);
    expect(env.step(LEFT).observation).toBe(0);
    expect(env.step(UP).observation).toBe(0);
  });

  it("terminates on a hole and on the goal, rewarding only the goal", () => {
    const env = new GridWorld({ slippery: false });
    env.reset(0);
    env.step(RIGHT); // 1
    const hole = env.step(DOWN); // 5 is H
    expect(hole).toMatchObject({ observation: 5, reward: 0, terminated: true, truncated: false });

    env.reset();
    let last = env.step(DOWN); // 4
    for (const a of [DOWN, RIGHT, DOWN, RIGHT, RIGHT]) last = env.step(a); // 8, 9, 13, 14, 15
    expect(last).toMatchObject({ observation: 15, reward: 1, terminated: true });
  });

  it("refuses to step a finished episode rather than silently continuing it", () => {
    const env = new GridWorld({ slippery: false });
    expect(() => env.step(LEFT)).toThrow(/reset/);
    env.reset(0);
    env.step(RIGHT);
    env.step(DOWN); // hole
    expect(() => env.step(LEFT)).toThrow(/finished episode/);
  });

  it("truncates at the time limit, which is not a termination", () => {
    const env = new GridWorld({ slippery: false, timeLimit: 3 });
    env.reset(0);
    env.step(LEFT);
    env.step(LEFT);
    expect(env.step(LEFT)).toMatchObject({ terminated: false, truncated: true });
  });

  it("gives the same trajectory for the same seed, and a different one for another", () => {
    const trace = (seed: number) => {
      const env = new GridWorld({ slippery: true, map: "8x8" });
      const states: number[] = [];
      for (let episode = 0; episode < 5; episode++) {
        env.reset(episode === 0 ? seed : undefined);
        for (;;) {
          const r = env.step(RIGHT);
          states.push(r.observation);
          if (r.terminated || r.truncated) break;
        }
      }
      return states;
    };
    expect(trace(3)).toEqual(trace(3));
    expect(trace(3)).not.toEqual(trace(4));
  });
});
