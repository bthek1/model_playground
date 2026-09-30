// A FrozenLake-shaped grid world, transcribed from Gymnasium rather than
// invented.
//
// Source: Gymnasium `gymnasium/envs/toy_text/frozen_lake.py` (FrozenLake-v1),
// https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/toy_text/frozen_lake.py
//
// What is transcribed, and why each matters for a comparison against a
// tutorial:
//
//   - **The maps.** `MAPS["4x4"]` and `MAPS["8x8"]`, letter for letter. A map
//     with one hole moved is a different MDP with different optimal values.
//   - **The action order.** LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3. A Q-table
//     printed from Python reads column-for-column against this one.
//   - **Walls.** Moving off the edge leaves the agent where it is (`inc()`
//     clamps), and costs a step.
//   - **The slip rule.** With `is_slippery`, the agent moves in one of
//     `(a − 1) % 4`, `a`, `(a + 1) % 4` — the intended direction or one of the
//     two perpendicular to it, **never backwards** — each with probability 1/3
//     (Gymnasium's `success_rate` default, with the remainder split evenly
//     between the two perpendiculars). The folk version "slip to any
//     neighbour" is a different environment and every published number
//     disagrees with it.
//   - **Rewards and termination.** 1 for entering G, 0 otherwise; entering H or
//     G terminates. Terminal states self-loop with no reward in the model.
//   - **The time limit.** `FrozenLake-v1` is registered with
//     `max_episode_steps=100`, and `FrozenLake8x8-v1` with 200. Hitting it
//     *truncates* — the state still has a future — which is why `StepResult`
//     keeps `truncated` apart from `terminated`.
//
// What is not transcribed: numpy's random stream. The same seed here and in
// Python gives different trajectories; the *distribution* is the same.

import { mulberry32 } from "@/lib/random";

import type { Env, FiniteMdp, GridMapId, StepResult, Transition } from "../types";

export const LEFT = 0;
export const DOWN = 1;
export const RIGHT = 2;
export const UP = 3;
export const N_ACTIONS = 4;

export const ACTION_NAMES = ["left", "down", "right", "up"] as const;

/** `MAPS` from frozen_lake.py, verbatim. */
export const MAPS: Record<GridMapId, readonly string[]> = {
  "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
  "8x8": [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
  ],
};

/** The registered `max_episode_steps` for each map. */
export const TIME_LIMITS: Record<GridMapId, number> = { "4x4": 100, "8x8": 200 };

/** Gymnasium's `success_rate` default: the intended move happens a third of the time. */
export const SUCCESS_RATE = 1 / 3;

export type Cell = "S" | "F" | "H" | "G";

export interface GridWorldOptions {
  map?: GridMapId;
  slippery?: boolean;
  /** Override the registered limit — tests use a short one. */
  timeLimit?: number;
}

export class GridWorld implements Env<number, number>, FiniteMdp {
  readonly rows: number;
  readonly cols: number;
  readonly nStates: number;
  readonly nActions = N_ACTIONS;
  readonly observationSpace;
  readonly actionSpace = { kind: "discrete" as const, n: N_ACTIONS };
  readonly map: GridMapId;
  readonly slippery: boolean;
  readonly timeLimit: number;
  readonly start: number;
  readonly goal: number;

  private readonly cells: Cell[];
  /** `P[s][a]` — the whole model, built once, exactly as the upstream constructor does. */
  private readonly model: Transition[][][];
  private rand: () => number = mulberry32(0);
  private state = 0;
  private elapsed = 0;
  private finished = true;

  constructor(options: GridWorldOptions = {}) {
    this.map = options.map ?? "4x4";
    this.slippery = options.slippery ?? true;
    this.timeLimit = options.timeLimit ?? TIME_LIMITS[this.map];
    const desc = MAPS[this.map];
    this.rows = desc.length;
    this.cols = desc[0].length;
    this.nStates = this.rows * this.cols;
    this.observationSpace = { kind: "discrete" as const, n: this.nStates };
    this.cells = desc.join("").split("") as Cell[];
    this.start = this.cells.indexOf("S");
    this.goal = this.cells.indexOf("G");
    this.model = this.buildModel();
  }

  cell(state: number): Cell {
    return this.cells[state];
  }

  isTerminal(state: number): boolean {
    const c = this.cells[state];
    return c === "H" || c === "G";
  }

  /** `inc()` from frozen_lake.py: move one cell, clamped at the walls. */
  move(state: number, action: number): number {
    let row = Math.floor(state / this.cols);
    let col = state % this.cols;
    if (action === LEFT) col = Math.max(col - 1, 0);
    else if (action === DOWN) row = Math.min(row + 1, this.rows - 1);
    else if (action === RIGHT) col = Math.min(col + 1, this.cols - 1);
    else if (action === UP) row = Math.max(row - 1, 0);
    return row * this.cols + col;
  }

  private buildModel(): Transition[][][] {
    const P: Transition[][][] = [];
    for (let s = 0; s < this.nStates; s++) {
      P.push([]);
      for (let a = 0; a < N_ACTIONS; a++) {
        const li: Transition[] = [];
        if (this.isTerminal(s)) {
          li.push({ p: 1, next: s, reward: 0, terminated: true });
        } else if (this.slippery) {
          for (const b of [(a + 3) % 4, a, (a + 1) % 4]) {
            const p = b === a ? SUCCESS_RATE : (1 - SUCCESS_RATE) / 2;
            li.push({ p, ...this.outcome(s, b) });
          }
        } else {
          li.push({ p: 1, ...this.outcome(s, a) });
        }
        P[s].push(li);
      }
    }
    return P;
  }

  /** `update_probability_matrix()` from frozen_lake.py. */
  private outcome(s: number, a: number): Omit<Transition, "p"> {
    const next = this.move(s, a);
    const c = this.cells[next];
    return { next, reward: c === "G" ? 1 : 0, terminated: c === "G" || c === "H" };
  }

  transitions(state: number, action: number): readonly Transition[] {
    return this.model[state][action];
  }

  get position(): number {
    return this.state;
  }

  reset(seed?: number): number {
    if (seed !== undefined) this.rand = mulberry32(seed);
    this.state = this.start;
    this.elapsed = 0;
    this.finished = false;
    return this.state;
  }

  step(action: number): StepResult<number> {
    if (this.finished) {
      throw new Error("step() called on a finished episode — call reset() first");
    }
    if (!Number.isInteger(action) || action < 0 || action >= N_ACTIONS) {
      throw new Error(`invalid action ${action}`);
    }
    const outcomes = this.model[this.state][action];
    // `categorical_sample` — one uniform draw against the cumulative sum.
    let t = outcomes[outcomes.length - 1];
    if (outcomes.length > 1) {
      const u = this.rand();
      let acc = 0;
      for (const o of outcomes) {
        acc += o.p;
        if (u < acc) {
          t = o;
          break;
        }
      }
    }
    this.state = t.next;
    this.elapsed++;
    const truncated = !t.terminated && this.elapsed >= this.timeLimit;
    this.finished = t.terminated || truncated;
    return {
      observation: t.next,
      reward: t.reward,
      terminated: t.terminated,
      truncated,
    };
  }
}
