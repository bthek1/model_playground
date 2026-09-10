import { afterEach, describe, expect, it, vi } from "vitest";

import type { Detection } from "../draw";
import type { ImagePayload } from "../image";
import { createPoseHandler, type PoseStages, type RawPose } from "./poseEngine";
import type { PoseResponse } from "./types";

function clearGpu() {
  Object.defineProperty(navigator, "gpu", {
    value: undefined,
    configurable: true,
    writable: true,
  });
}

function image(width = 100, height = 100): ImagePayload {
  return {
    data: new Uint8ClampedArray(width * height * 3),
    width,
    height,
    channels: 3,
  };
}

const det = (
  label: string,
  score: number,
  box: [number, number, number, number],
): Detection => ({
  label,
  score,
  box: { xmin: box[0], ymin: box[1], xmax: box[2], ymax: box[3] },
});

const pose = (x: number, y: number): RawPose => ({
  keypoints: [[x, y]],
  scores: [0.9],
  labels: [0],
});

const LOAD = { type: "load", model: "dfine-n+vitpose-base" } as const;
const WASM = { device: "wasm", dtype: "q8" } as const;

/** `Array.prototype.at` is ES2022; the app's lib is ES2020. */
const last = <T,>(items: T[]): T => items[items.length - 1];

function stages(over: Partial<PoseStages> = {}): PoseStages {
  return {
    detect: vi.fn(async () => [] as Detection[]),
    estimate: vi.fn(async () => [] as RawPose[]),
    ...over,
  };
}

const RUN = { type: "run", id: 1, threshold: 0.4, maxPeople: 5 } as const;

describe("createPoseHandler", () => {
  afterEach(() => clearGpu());

  it("resolves the pair from its composite id and posts ready", async () => {
    clearGpu();
    const posted: PoseResponse[] = [];
    const factory = vi.fn(async (opts) => {
      opts.progress_callback?.({
        status: "progress",
        name: "onnx-community/dfine_n_coco-ONNX",
        file: "onnx/model_quantized.onnx",
      });
      return stages();
    });

    const handle = createPoseHandler((m) => posted.push(m), factory, {
      warmup: false,
    });
    await handle(LOAD);

    expect(factory).toHaveBeenCalledWith(
      expect.objectContaining({
        device: "wasm",
        entry: expect.objectContaining({ id: "dfine-n+vitpose-base" }),
      }),
    );
    expect(last(posted)).toMatchObject({
      type: "ready",
      model: "dfine-n+vitpose-base",
    });
  });

  it("warms up through both stages, and survives one that throws", async () => {
    // Both compile separately, and the pose model's compile is the expensive
    // half — a first frame that pays for it reads as "this is slow".
    const detect = vi.fn(async () => [] as Detection[]);
    const estimate = vi.fn().mockRejectedValue(new Error("shader compile blew up"));
    const posted: PoseResponse[] = [];
    const handle = createPoseHandler(
      (m) => posted.push(m),
      async () => stages({ detect, estimate }),
    );

    await handle({ ...LOAD, opts: WASM });
    expect(detect).toHaveBeenCalledTimes(1);
    expect(estimate).toHaveBeenCalledTimes(1);
    expect(posted).toContainEqual({
      type: "progress",
      progress: { status: "warmup" },
    });
    expect(last(posted)).toMatchObject({ type: "ready" });
  });

  it("crops only the people, and maps their joints back to source pixels", async () => {
    // The correctness surface of this route, end to end: a joint at the middle
    // of the crop must land at the middle of that person's box.
    const detect = vi.fn(async () => [
      det("car", 0.99, [0, 0, 20, 20]),
      det("person", 0.8, [40, 60, 60, 100]), // 20x40 crop at (40, 60)
    ]);
    const estimate = vi.fn(async () => [pose(10, 20)]);
    const posted: PoseResponse[] = [];
    const handle = createPoseHandler(
      (m) => posted.push(m),
      async () => stages({ detect, estimate }),
      { warmup: false },
    );

    await handle({ ...LOAD, opts: WASM });
    await handle({ ...RUN, image: image() });

    // One crop, and it is the person's — not the car's.
    expect(estimate).toHaveBeenCalledWith(expect.anything(), [[40, 60, 20, 40]]);

    const result = (last(posted) as { result: { people: { keypoints: unknown[] }[] } })
      .result;
    expect(result.people).toHaveLength(1);
    expect(result.people[0].keypoints[0]).toMatchObject({ x: 50, y: 80 });
  });

  it("reports how many people it found, before the cap", async () => {
    // "3 of 7" is the honest label; a cap that silently hides four people is
    // not.
    const detect = vi.fn(async () => [
      det("person", 0.9, [0, 0, 10, 10]),
      det("person", 0.8, [10, 0, 20, 10]),
      det("person", 0.7, [20, 0, 30, 10]),
    ]);
    const estimate = vi.fn(async () => [pose(1, 1)]);
    const posted: PoseResponse[] = [];
    const handle = createPoseHandler(
      (m) => posted.push(m),
      async () => stages({ detect, estimate }),
      { warmup: false },
    );

    await handle({ ...LOAD, opts: WASM });
    await handle({ ...RUN, maxPeople: 1, image: image() });

    expect(last(posted)).toMatchObject({
      result: { detected: 3, people: [expect.anything()] },
    });
  });

  it("never asks the pose model anything when nobody is above the threshold", async () => {
    // The pose pass is the expensive half; running it on an empty list is
    // wasted and would report a misleading `poseMs`.
    const estimate = vi.fn(async () => [] as RawPose[]);
    const posted: PoseResponse[] = [];
    const handle = createPoseHandler(
      (m) => posted.push(m),
      async () =>
        stages({
          detect: vi.fn(async () => [det("person", 0.1, [0, 0, 10, 10])]),
          estimate,
        }),
      { warmup: false },
    );

    await handle({ ...LOAD, opts: WASM });
    await handle({ ...RUN, image: image() });

    expect(estimate).not.toHaveBeenCalled();
    expect(last(posted)).toMatchObject({
      result: { people: [], detected: 0, poseMs: 0 },
    });
  });

  it("drops a person whose box clamps to nothing rather than shifting the rest", async () => {
    // Boxes and detections stay aligned: a degenerate crop must not put
    // person 2's skeleton on person 3's body.
    const detect = vi.fn(async () => [
      det("person", 0.9, [200, 200, 300, 300]), // entirely off a 100x100 image
      det("person", 0.8, [10, 10, 50, 90]),
    ]);
    const estimate = vi.fn(async () => [pose(5, 5)]);
    const posted: PoseResponse[] = [];
    const handle = createPoseHandler(
      (m) => posted.push(m),
      async () => stages({ detect, estimate }),
      { warmup: false },
    );

    await handle({ ...LOAD, opts: WASM });
    await handle({ ...RUN, image: image(100, 100) });

    expect(estimate).toHaveBeenCalledWith(expect.anything(), [[10, 10, 40, 80]]);
    const result = (
      last(posted) as { result: { people: { box: { xmin: number } }[] } }
    ).result;
    expect(result.people).toHaveLength(1);
    expect(result.people[0].box.xmin).toBe(10);
  });

  it("disposes both models, even when the first dispose throws", async () => {
    // The documented exception to "one model live at a time" is also the one
    // place a half-finished teardown can leak 170 MB.
    const disposeDetector = vi.fn().mockRejectedValue(new Error("boom"));
    const disposePose = vi.fn().mockResolvedValue(undefined);
    const posted: PoseResponse[] = [];
    const handle = createPoseHandler(
      (m) => posted.push(m),
      async () => stages({ disposeDetector, disposePose }),
      { warmup: false },
    );

    await handle({ ...LOAD, opts: WASM });
    await handle({ ...LOAD, model: "rtdetr-r50+vitpose-base", opts: WASM });

    expect(disposeDetector).toHaveBeenCalled();
    expect(disposePose).toHaveBeenCalled();
    expect(last(posted)).toMatchObject({ type: "ready" });
  });

  it("puts an inference failure in Machine B, with both models still loaded", async () => {
    const posted: PoseResponse[] = [];
    const handle = createPoseHandler(
      (m) => posted.push(m),
      async () =>
        stages({
          detect: vi.fn().mockRejectedValue(new Error("Non-zero status code")),
        }),
      { warmup: false },
    );

    await handle({ ...LOAD, opts: WASM });
    await handle({ ...RUN, image: image() });

    expect(last(posted)).toEqual({
      type: "error",
      id: 1,
      error: "Non-zero status code",
    });
  });

  it("refuses to run before a model is loaded", async () => {
    const posted: PoseResponse[] = [];
    const handle = createPoseHandler((m) => posted.push(m), async () => stages());
    await handle({ ...RUN, image: image() });
    expect(last(posted)).toEqual({
      type: "error",
      id: 1,
      error: "No model loaded",
    });
  });
});
