import { describe, expect, it } from "vitest";

import { trainRlInWorker } from "./client";
import type { QLearningRequest, RlProgress, RlWorkerRequest, RlWorkerResponse } from "./types";

/** A Worker stand-in: records what was posted and lets the test answer. */
class FakeWorker extends EventTarget {
  posted: RlWorkerRequest[] = [];
  postMessage(message: RlWorkerRequest) {
    this.posted.push(message);
  }
  reply(data: RlWorkerResponse) {
    this.dispatchEvent(new MessageEvent("message", { data }));
  }
}

const REQ: QLearningRequest = {
  algorithm: "q-learning",
  map: "4x4",
  slippery: false,
  alpha: 0.5,
  gamma: 0.95,
  epsilon: 1,
  decay: true,
  episodes: 10,
  seed: 1,
};

function progress(episode: number): RlProgress {
  return {
    returns: [1],
    episode,
    totalEpisodes: 10,
    steps: episode * 6,
    epsilon: 0.5,
    render: { kind: "grid", map: "4x4", q: new Float32Array(64), agent: 0 },
  };
}

const RESULT = { episodes: 10, steps: 60, elapsedMs: 2, stopped: false, epsilonChanged: false, render: null };

describe("trainRlInWorker", () => {
  it("posts the request with its speed, and forwards progress for its own id only", async () => {
    const worker = new FakeWorker();
    const seen: number[] = [];
    const handle = trainRlInWorker(worker as unknown as Worker, REQ, (p) => seen.push(p.episode), 300);
    const sent = worker.posted[0];
    expect(sent).toMatchObject({ type: "train", req: REQ, speed: 300 });
    const id = sent.id;

    worker.reply({ id: id + 999, event: "progress", progress: progress(9) }); // another run's
    worker.reply({ id, event: "progress", progress: progress(3) });
    worker.reply({ id, ok: true, result: RESULT });
    await expect(handle.promise).resolves.toEqual(RESULT);
    expect(seen).toEqual([3]);
  });

  it("rejects with the worker's error", async () => {
    const worker = new FakeWorker();
    const handle = trainRlInWorker(worker as unknown as Worker, REQ, () => {});
    worker.reply({ id: worker.posted[0].id, ok: false, error: "bad grid" });
    await expect(handle.promise).rejects.toThrow("bad grid");
  });

  it("sends Stop and live controls against the same id, without a new run", () => {
    const worker = new FakeWorker();
    const handle = trainRlInWorker(worker as unknown as Worker, REQ, () => {});
    const id = worker.posted[0].id;
    handle.control({ epsilon: 0 });
    handle.control({ speed: null });
    handle.cancel();
    expect(worker.posted.slice(1)).toEqual([
      { type: "control", id, control: { epsilon: 0 } },
      { type: "control", id, control: { speed: null } },
      { type: "cancel", id },
    ]);
  });

  it("gives every run its own id", () => {
    const worker = new FakeWorker();
    trainRlInWorker(worker as unknown as Worker, REQ, () => {});
    trainRlInWorker(worker as unknown as Worker, REQ, () => {});
    expect(worker.posted[0].id).not.toBe(worker.posted[1].id);
  });
});
