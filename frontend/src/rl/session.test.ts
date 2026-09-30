import { describe, expect, it, vi } from "vitest";

import { createRlSession } from "./session";
import type { QLearningRequest, RlProgress, RlTrainResult, RlWorkerResponse } from "./types";

function request(over: Partial<QLearningRequest> = {}): QLearningRequest {
  return {
    algorithm: "q-learning",
    map: "4x4",
    slippery: false,
    alpha: 0.5,
    gamma: 0.95,
    epsilon: 1,
    decay: true,
    episodes: 400,
    seed: 1,
    ...over,
  };
}

/** A clock the test owns: every read advances it, so slices and intervals are deterministic. */
function fakeClock(msPerRead: number) {
  let t = 0;
  return () => (t += msPerRead);
}

function harness(opts: { msPerRead?: number; renderIntervalMs?: number } = {}) {
  const messages: RlWorkerResponse[] = [];
  const onPost = vi.fn();
  const handle = createRlSession(
    (m, transfer) => {
      // What a real postMessage does: clone, and detach whatever was
      // transferred. Without this the harness passes a result that references
      // a buffer the previous post already gave away.
      const sent = structuredClone(m, { transfer });
      messages.push(sent);
      onPost(sent);
    },
    {
      now: fakeClock(opts.msPerRead ?? 0.01),
      yieldToEvents: async () => {},
      sleep: async () => {},
      renderIntervalMs: opts.renderIntervalMs,
    },
  );
  const progress = () =>
    messages.flatMap((m) => ("event" in m ? [m.progress as RlProgress] : []));
  const result = () => {
    const done = messages.find((m) => "ok" in m);
    if (!done || !("ok" in done) || !done.ok) throw new Error("no result");
    return done.result as RlTrainResult;
  };
  return { handle, messages, progress, result, onPost };
}

describe("createRlSession", () => {
  it("posts on the interval, not per step", async () => {
    // Phase 0's factor is only real if it cannot regress: count the posts.
    const h = harness({ msPerRead: 0.01 });
    await h.handle({ type: "train", id: 1, req: request() });
    const { steps } = h.result();
    const posts = h.progress().length;
    expect(steps).toBeGreaterThan(2000);
    expect(posts).toBeGreaterThan(0);
    expect(posts).toBeLessThan(steps / 50);
  });

  it("delivers every episode's return exactly once, across the posts", async () => {
    const h = harness();
    await h.handle({ type: "train", id: 1, req: request({ episodes: 250 }) });
    const all = h.progress().flatMap((p) => p.returns);
    expect(all).toHaveLength(250);
    expect(h.result().episodes).toBe(250);
    expect(h.progress()[h.progress().length - 1]?.episode).toBe(250);
    // The result's table must be readable after the last post transferred its
    // own — the bug a real browser found and a plain-object harness did not.
    const render = h.result().render;
    expect(render?.kind === "grid" && render.q).toHaveLength(64);
  });

  it("posts after every step when the interval is zero — the measurement's control arm", async () => {
    const h = harness({ renderIntervalMs: 0 });
    await h.handle({ type: "train", id: 1, req: request({ episodes: 5 }) });
    // One per step, plus the final flush.
    expect(h.progress().length).toBe(h.result().steps + 1);
  });

  it("stops within the episode it was in, and resolves rather than rejecting", async () => {
    const h = harness();
    let sent = false;
    h.onPost.mockImplementation((m: RlWorkerResponse) => {
      if (!sent && "event" in m) {
        sent = true;
        void h.handle({ type: "cancel", id: 1 });
      }
    });
    await h.handle({ type: "train", id: 1, req: request({ episodes: 100000 }) });
    const result = h.result();
    expect(result.stopped).toBe(true);
    const firstPost = h.progress()[0];
    expect(result.episodes).toBeLessThanOrEqual(firstPost.episode + 1);
    expect(h.messages.some((m) => "ok" in m && !m.ok)).toBe(false);
  });

  it("handles a Stop that arrives before the run has started", async () => {
    const h = harness();
    await h.handle({ type: "cancel", id: 9 });
    await h.handle({ type: "train", id: 9, req: request() });
    const result = h.result();
    expect(result).toMatchObject({ stopped: true, episodes: 0, steps: 0, render: null });
    expect(h.progress()).toEqual([]);
  });

  it("changes ε live without restarting, and says so in the result", async () => {
    const h = harness();
    let sent = false;
    h.onPost.mockImplementation((m: RlWorkerResponse) => {
      if (!sent && "event" in m) {
        sent = true;
        void h.handle({ type: "control", id: 1, control: { epsilon: 0.25 } });
      }
    });
    await h.handle({ type: "train", id: 1, req: request({ episodes: 3000 }) });
    const posts = h.progress();
    expect(posts[0].epsilon).not.toBe(0.25);
    expect(posts[posts.length - 1]?.epsilon).toBe(0.25);
    expect(h.result()).toMatchObject({ epsilonChanged: true, episodes: 3000, stopped: false });
  });

  it("throttles to the speed dial", async () => {
    // At 1000 steps/s on a clock advancing 0.01 ms per read, the run must not
    // get ahead of the wall time it has been given.
    const clock = fakeClock(0.01);
    let lastNow = 0;
    const now = () => (lastNow = clock());
    const messages: RlWorkerResponse[] = [];
    const handle = createRlSession((m) => messages.push(m), {
      now,
      yieldToEvents: async () => {},
      sleep: async () => {},
    });
    await handle({ type: "train", id: 1, req: request({ episodes: 20 }), speed: 1000 });
    const done = messages.find((m) => "ok" in m && m.ok) as { result: RlTrainResult };
    expect(done.result.steps).toBeLessThanOrEqual(Math.ceil(lastNow) + 1);
  });

  it("reports a failure in the error envelope", async () => {
    const messages: RlWorkerResponse[] = [];
    const handle = createRlSession((m) => messages.push(m), {
      yieldToEvents: async () => {},
      createStepper: () => {
        throw new Error("bad grid");
      },
    });
    await handle({ type: "train", id: 3, req: request() });
    expect(messages).toEqual([{ id: 3, ok: false, error: "bad grid" }]);
  });

  it("drives every learner through the same loop, and each render survives a real transfer", async () => {
    // The Stepper seam's promise: the session never learns which algorithm it
    // holds. The harness structured-clones with transfer, so a learner whose
    // render state reused a buffer the loop still writes — or referenced one
    // already given away — would fail here, as it did in a real browser.
    const cases = [
      {
        req: { algorithm: "reinforce", env: "cartpole", hidden: 8, lr: 0.003, criticLr: 0.03, gamma: 0.99, lambda: 0.9, normalise: false, episodes: 5, seed: 1 },
        kind: "cartpole",
      },
      {
        req: { algorithm: "actor-critic", env: "cartpole", hidden: 8, lr: 0.01, criticLr: 0.03, gamma: 0.99, lambda: 0.9, normalise: false, episodes: 5, seed: 1 },
        kind: "cartpole",
      },
      {
        req: { algorithm: "behaviour-cloning", mix: "both", demos: 4, obstacle: 0, hidden: 8, epochs: 3, lr: 0.01, seed: 1 },
        kind: "cloning",
      },
    ] as const;
    for (const c of cases) {
      const h = harness();
      await h.handle({ type: "train", id: 1, req: c.req });
      const result = h.result();
      expect(result.render?.kind, c.req.algorithm).toBe(c.kind);
      expect(h.progress().flatMap((p) => p.returns)).toHaveLength(result.episodes);
      // A policy gradient and a regression explore by sampling, not by ε.
      expect(h.progress()[0].epsilon).toBeNull();
    }
  });
});
