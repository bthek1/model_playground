// The worker-side owner of an RL training run: construct the environment and
// the learner, drive the loop, post the render state on a wall-clock interval,
// and honour Stop.
//
// A pure message handler over injected pieces — the same split as
// `webgpu/graphSession.ts` and `tabular/fitEngine.ts` — so every rule below is
// unit-tested without a Worker.
//
// **The step rate and the frame rate are different numbers.** The loop runs as
// fast as the CPU goes (or as fast as the speed dial allows) and the canvas is
// repainted at most every `renderIntervalMs`, never per step. `limits.ts`
// carries the measured cost of getting that wrong.
//
// **Control messages jump the queue.** `cancel` and `control` are handled
// synchronously the moment they arrive, and the loop yields a *macrotask*
// between slices so they can arrive at all. That second half is the trap: a
// CPU loop that only ever awaits resolved promises never lets the event loop
// deliver a message, so Stop would be queued behind the run it was meant to
// interrupt. `webgpu/worker.ts` gets away without this because its awaits are
// real GPU work, which resolves as a task; this loop has no GPU to lean on.

import { QLearningRun } from "./qLearning";
import type { Stepper } from "./stepper";
import type {
  RlProgress,
  RlRenderState,
  RlTrainRequest,
  RlTrainResult,
  RlWorkerRequest,
  RlWorkerResponse,
} from "./types";

/** ≈ 60 Hz. The render state is posted no more often than this. */
export const RENDER_INTERVAL_MS = 1000 / 60;
/** How long the loop computes before it yields to receive a Stop. */
export const SLICE_MS = 8;

export interface SessionDeps {
  now?: () => number;
  /** Yield one macrotask. The worker passes a MessageChannel; tests pass a no-op. */
  yieldToEvents?: () => Promise<void>;
  /** Wait while a throttled run is ahead of its budget. */
  sleep?: (ms: number) => Promise<void>;
  /** `0` posts after every step — only Phase 0's measurement asks for that. */
  renderIntervalMs?: number;
  /** Builds the learner for a request. Injected so a test can hold the stepper. */
  createStepper?: (req: RlTrainRequest) => Stepper;
}

export function createStepper(req: RlTrainRequest): Stepper {
  switch (req.algorithm) {
    case "q-learning":
      return new QLearningRun(req);
  }
}

interface Live {
  cancelled: boolean;
  epsilonChanged: boolean;
  speed: number | null;
  stepper: Stepper | null;
  /** Rate baseline: a speed change restarts the budget rather than bursting. */
  baseTime: number;
  baseSteps: number;
}

export function createRlSession(
  post: (message: RlWorkerResponse, transfer?: Transferable[]) => void,
  deps: SessionDeps = {},
) {
  const now = deps.now ?? (() => performance.now());
  const yieldToEvents = deps.yieldToEvents ?? defaultYield;
  const sleep = deps.sleep ?? ((ms: number) => new Promise<void>((r) => setTimeout(r, ms)));
  const defaultInterval = deps.renderIntervalMs ?? RENDER_INTERVAL_MS;
  const build = deps.createStepper ?? createStepper;

  const live = new Map<number, Live>();
  // A Stop that arrives before its run has started — pressed in the same frame
  // as Train. Remembered so the run resolves at once instead of ignoring it.
  const early = new Set<number>();

  function control(id: number, change: { epsilon?: number; speed?: number | null }) {
    const run = live.get(id);
    if (!run) return;
    if (change.epsilon !== undefined && run.stepper) {
      run.stepper.epsilonOverride = change.epsilon;
      run.epsilonChanged = true;
    }
    if (change.speed !== undefined) {
      run.speed = change.speed;
      run.baseTime = now();
      run.baseSteps = run.stepper?.steps ?? 0;
    }
  }

  async function train(
    id: number,
    req: RlTrainRequest,
    speed: number | null,
    interval: number,
  ) {
    const run: Live = {
      cancelled: early.delete(id),
      epsilonChanged: false,
      speed,
      stepper: null,
      baseTime: now(),
      baseSteps: 0,
    };
    live.set(id, run);
    const started = now();

    try {
      const stepper = build(req);
      run.stepper = stepper;
      let pending: number[] = [];
      let lastPost = -Infinity;

      const flush = () => {
        const render = stepper.render();
        const progress: RlProgress = {
          returns: pending,
          episode: stepper.episode,
          totalEpisodes: stepper.totalEpisodes,
          steps: stepper.steps,
          epsilon: stepper.epsilon,
          render,
        };
        pending = [];
        lastPost = now();
        post({ id, event: "progress", progress }, transferOf(render));
      };

      while (!stepper.done && !run.cancelled) {
        const sliceStart = now();
        // How many steps this slice may take: unlimited at full speed, or what
        // the wall clock has earned at the dial's rate.
        let budget = Infinity;
        if (run.speed != null) {
          const earned = Math.floor(((now() - run.baseTime) * run.speed) / 1000);
          budget = earned - (stepper.steps - run.baseSteps);
        }

        let took = 0;
        while (took < budget && !stepper.done) {
          const ret = stepper.step();
          took++;
          if (ret !== null) pending.push(ret);
          if (interval === 0) break;
          if ((took & 63) === 0 && now() - sliceStart >= SLICE_MS) break;
        }

        if (interval === 0 || now() - lastPost >= interval) flush();

        if (took === 0 && run.speed != null) {
          // Ahead of the dial: wait for the next step's worth of time, but no
          // longer than a frame, so a Stop is still honoured promptly.
          await sleep(Math.min(interval || 16, Math.max(1, 1000 / run.speed)));
        } else {
          await yieldToEvents();
        }
      }

      // The last episodes, and the final table, always reach the page.
      if (pending.length > 0 || !run.cancelled || stepper.episode > 0) flush();

      const result: RlTrainResult = {
        episodes: stepper.episode,
        steps: stepper.steps,
        elapsedMs: now() - started,
        stopped: run.cancelled,
        epsilonChanged: run.epsilonChanged,
        // A fresh copy, never the last progress post's: that one's buffer was
        // *transferred*, so it is detached here and cloning it throws — which
        // the page would see as a run that failed on its last line.
        render: stepper.steps > 0 ? stepper.render() : null,
      };
      post({ id, ok: true, result });
    } catch (error) {
      post({ id, ok: false, error: error instanceof Error ? error.message : String(error) });
    } finally {
      live.delete(id);
    }
  }

  return function handle(msg: RlWorkerRequest): Promise<void> | void {
    if (msg.type === "cancel") {
      const run = live.get(msg.id);
      if (run) run.cancelled = true;
      else early.add(msg.id);
      return;
    }
    if (msg.type === "control") {
      control(msg.id, msg.control);
      return;
    }
    return train(msg.id, msg.req, msg.speed ?? null, msg.renderIntervalMs ?? defaultInterval);
  };
}

function transferOf(render: RlRenderState): Transferable[] {
  return [render.q.buffer];
}

/**
 * One macrotask, without `setTimeout`'s 4 ms clamp — which a loop that yields
 * hundreds of times a second would otherwise pay on every slice.
 */
const defaultYield: () => Promise<void> = (() => {
  if (typeof MessageChannel === "undefined") {
    return () => new Promise<void>((r) => setTimeout(r, 0));
  }
  const channel = new MessageChannel();
  const queue: Array<() => void> = [];
  channel.port1.onmessage = () => queue.shift()?.();
  return () =>
    new Promise<void>((resolve) => {
      queue.push(resolve);
      channel.port2.postMessage(null);
    });
})();
