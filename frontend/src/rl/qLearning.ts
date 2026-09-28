// Tabular Q-learning: the whole model is a table and the whole algorithm is one
// update rule.
//
//   Q(s, a) ← Q(s, a) + α · [ r + γ · max_a' Q(s', a') · (1 − terminated) − Q(s, a) ]
//
// Two details that are easy to get wrong without anything failing:
//
//   - **Bootstrap on `terminated`, never on `truncated`.** A time limit is not
//     the end of the world: the state the clock ran out in still has a future,
//     so its target keeps the `γ · max Q(s')` term. Masking it on `done` treats
//     the clock as a hole.
//   - **Ties go to the first action**, exactly as `np.argmax` does. That is not
//     an accident of implementation — it is what makes ε = 0 fail. From a cold
//     table every action is worth 0, so a greedy agent picks LEFT, walks into
//     the wall at the start, learns that the wall is worth 0 (which it already
//     thought), and does it again for ever. Break ties at random instead and a
//     cold table *explores by accident*, which hides the lesson this page is
//     for. `qLearning.test.ts` pins both halves.

import { mulberry32 } from "@/lib/random";

import { GridWorld } from "./envs/gridWorld";
import type { Stepper } from "./stepper";
import type { Env, GridRenderState, QLearningRequest } from "./types";

export class QTable {
  readonly q: Float32Array;

  constructor(
    readonly nStates: number,
    readonly nActions: number,
  ) {
    this.q = new Float32Array(nStates * nActions);
  }

  value(s: number, a: number): number {
    return this.q[s * this.nActions + a];
  }

  /** First index of the maximum — `np.argmax` semantics. See the header. */
  greedy(s: number): number {
    const base = s * this.nActions;
    let best = 0;
    let bestValue = this.q[base];
    for (let a = 1; a < this.nActions; a++) {
      if (this.q[base + a] > bestValue) {
        bestValue = this.q[base + a];
        best = a;
      }
    }
    return best;
  }

  max(s: number): number {
    return this.value(s, this.greedy(s));
  }
}

/** Where the decay schedule bottoms out. */
export const EPSILON_FLOOR = 0.05;
/** The fraction of the run the decay takes. */
export const DECAY_FRACTION = 0.8;

/**
 * The exploration rate for `episode` of `total`: linear from `start` to
 * `min(start, EPSILON_FLOOR)` over the first 80% of the run, then flat.
 *
 * It exists because of the tie-breaking rule above. With ties to the first
 * action, a small *constant* ε barely moves: the greedy action from the start
 * is LEFT, into the wall, and every exploratory step off the start is undone by
 * the next greedy one. Measured on the deterministic 4×4 at α = 0.5: ε = 0.2
 * held constant for 2000 episodes reaches the goal **zero** times, while this
 * schedule from ε = 1 finds it within ~170 episodes on every seed tried and
 * ends with the exact optimal policy. Random tie-breaking would also fix it —
 * and would make ε = 0 explore, which is the lesson the page is for.
 */
export function epsilonAt(
  episode: number,
  total: number,
  start: number,
  decay: boolean,
): number {
  if (!decay) return start;
  const floor = Math.min(start, EPSILON_FLOOR);
  const span = Math.max(1, DECAY_FRACTION * total);
  const t = Math.min(1, episode / span);
  return start + (floor - start) * t;
}

/** ε-greedy over a seeded stream: explore with probability ε, else the greedy action. */
export function epsilonGreedy(
  table: QTable,
  s: number,
  epsilon: number,
  rand: () => number,
): number {
  if (epsilon > 0 && rand() < epsilon) {
    return Math.floor(rand() * table.nActions);
  }
  return table.greedy(s);
}

/** The update rule, and nothing else. Returns the TD error. */
export function qUpdate(
  table: QTable,
  s: number,
  a: number,
  reward: number,
  next: number,
  terminated: boolean,
  alpha: number,
  gamma: number,
): number {
  const target = reward + (terminated ? 0 : gamma * table.max(next));
  const i = s * table.nActions + a;
  const td = target - table.q[i];
  table.q[i] += alpha * td;
  return td;
}

export interface EpisodeOutcome {
  return: number;
  steps: number;
  terminated: boolean;
}

/**
 * A whole Q-learning run, advanced one environment step at a time so the
 * session can pace it, draw it and stop it between any two steps. The learning
 * itself is `epsilonGreedy` + `qUpdate` and nothing else.
 */
export class QLearningRun implements Stepper {
  readonly env: GridWorld;
  readonly table: QTable;
  readonly totalEpisodes: number;
  episode = 0;
  steps = 0;
  /** A live ε, set by the RUN band's control. `null` follows the schedule. */
  epsilonOverride: number | null = null;

  private readonly rand: () => number;
  private state: number;
  private episodeReturn = 0;

  constructor(private readonly req: QLearningRequest) {
    this.env = new GridWorld({ map: req.map, slippery: req.slippery });
    this.table = new QTable(this.env.nStates, this.env.nActions);
    this.totalEpisodes = req.episodes;
    // The agent's stream and the environment's are separate, so toggling
    // slipperiness does not also change which exploratory actions are drawn.
    this.rand = mulberry32(req.seed ^ 0x51ed27);
    this.state = this.env.reset(req.seed);
  }

  get done(): boolean {
    return this.episode >= this.totalEpisodes;
  }

  get epsilon(): number {
    return (
      this.epsilonOverride ??
      epsilonAt(this.episode, this.totalEpisodes, this.req.epsilon, this.req.decay)
    );
  }

  get position(): number {
    return this.state;
  }

  step(): number | null {
    const s = this.state;
    const a = epsilonGreedy(this.table, s, this.epsilon, this.rand);
    const r = this.env.step(a);
    qUpdate(
      this.table,
      s,
      a,
      r.reward,
      r.observation,
      r.terminated,
      this.req.alpha,
      this.req.gamma,
    );
    this.steps++;
    this.episodeReturn += r.reward;
    if (!r.terminated && !r.truncated) {
      this.state = r.observation;
      return null;
    }
    const finished = this.episodeReturn;
    this.episode++;
    this.episodeReturn = 0;
    // Show the agent where it ended until the next episode's first step, rather
    // than teleporting it home before the frame that would have drawn the hole.
    this.state = this.done ? r.observation : this.env.reset();
    return finished;
  }

  render(): GridRenderState {
    return {
      kind: "grid",
      map: this.req.map,
      q: this.table.q.slice(),
      agent: this.state,
    };
  }
}

/** Run the greedy policy once, with no learning, and report whether it reached a reward. */
export function greedyRollout(
  env: Env<number, number>,
  table: QTable,
  seed?: number,
): EpisodeOutcome {
  let s = env.reset(seed);
  let total = 0;
  let steps = 0;
  for (;;) {
    const r = env.step(table.greedy(s));
    total += r.reward;
    steps++;
    s = r.observation;
    if (r.terminated || r.truncated) {
      return { return: total, steps, terminated: r.terminated };
    }
  }
}
