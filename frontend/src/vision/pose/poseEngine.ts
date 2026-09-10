// The pose worker's message-handling core, factored out of `pose.worker.ts` so
// it can be unit-tested with fake stages — no download, no real Worker, and no
// `@huggingface/transformers` import.
//
// It owes the same three behaviours as every engine here, with the one
// documented exception this route exists to make:
//
//  1. **Two models live at a time, and both disposed.** The exception to §5, and
//     the reason the teardown below nulls *both* references before disposing
//     either: a dispose that throws must not leave one of the pair live while
//     the other is replaced. `Promise.allSettled`, not `Promise.all` — a
//     detector whose teardown fails must not prevent the pose model's.
//  2. **Warm up before `ready`.** One throwaway pass through *both* stages, so
//     the first real frame pays for neither compile.
//  3. **Never block the main thread.**
//
// Both models are also *loaded* together rather than in sequence: their progress
// events then interleave, so `model/progress.ts` sees both denominators before
// either file finishes and the aggregate bar means something from the start.

import { loadOpts, pickBackend } from "@/model/backend";
import type { ModelProgress } from "@/model/types";

import type { Detection } from "../draw";
import type { ImagePayload } from "../image";
import { cropBox, personBoxes, toSourceKeypoints } from "./pose";
import type { Person } from "./skeleton";
import type { PosePairEntry, PoseRequest, PoseResponse } from "./types";
import { DETECTOR_THRESHOLD, POSE_MODELS } from "./types";

/** One person's raw pose answer, in the crop's own pixels. */
export interface RawPose {
  keypoints: [number, number][];
  scores: number[];
  labels: number[];
}

/** The two stages, as the worker supplies them. */
export interface PoseStages {
  /** Detect everything above `DETECTOR_THRESHOLD`, in source pixels. */
  detect: (image: ImagePayload) => Promise<Detection[]>;
  /** Run the pose model on each crop, in the order given. */
  estimate: (
    image: ImagePayload,
    boxes: readonly [number, number, number, number][],
  ) => Promise<RawPose[]>;
  disposeDetector?: () => Promise<void>;
  disposePose?: () => Promise<void>;
}

export interface PoseStageOpts {
  device: string;
  entry: PosePairEntry;
  progress_callback?: (p: ModelProgress) => void;
}

export type PoseStageFactory = (opts: PoseStageOpts) => Promise<PoseStages>;

const WARMUP_SIDE = 64;

function warmupImage(): ImagePayload {
  const data = new Uint8ClampedArray(WARMUP_SIDE * WARMUP_SIDE * 3);
  data.fill(128);
  return { data, width: WARMUP_SIDE, height: WARMUP_SIDE, channels: 3 };
}

export function createPoseHandler(
  post: (message: PoseResponse, transfer?: Transferable[]) => void,
  factory: PoseStageFactory,
  { warmup = true }: { warmup?: boolean } = {},
) {
  let stages: PoseStages | null = null;

  return async function handle(msg: PoseRequest): Promise<void> {
    if (msg.type === "load") {
      try {
        const previous = stages;
        stages = null;
        await disposeBoth(previous);

        const entry =
          POSE_MODELS.find((m) => m.id === msg.model) ?? POSE_MODELS[0];
        const opts = msg.opts ?? loadOpts(await pickBackend());
        stages = await factory({
          device: opts.device,
          entry,
          progress_callback: (progress) => post({ type: "progress", progress }),
        });

        if (warmup) {
          post({ type: "progress", progress: { status: "warmup" } });
          try {
            const image = warmupImage();
            await stages.detect(image);
            // A synthetic box, because the grey square has no people in it and
            // the pose model's compile is the expensive half.
            await stages.estimate(image, [[0, 0, WARMUP_SIDE, WARMUP_SIDE]]);
          } catch {
            /* the first real frame pays the compile cost instead */
          }
        }
        post({ type: "ready", model: entry.id, backend: opts.device });
      } catch (error) {
        post({ type: "error", error: errMessage(error) });
      }
      return;
    }

    try {
      if (!stages) throw new Error("No model loaded");

      const detectStarted = now();
      const detections = await stages.detect(msg.image);
      const detectMs = now() - detectStarted;

      const chosen = personBoxes(detections, {
        threshold: msg.threshold,
        maxPeople: msg.maxPeople,
      });
      // Boxes and detections are kept aligned: a degenerate crop drops the
      // person entirely rather than shifting every subsequent skeleton onto
      // the wrong body.
      const pairs: { detection: Detection; box: [number, number, number, number] }[] =
        [];
      for (const detection of chosen) {
        const box = cropBox(detection.box, msg.image.width, msg.image.height);
        if (box) pairs.push({ detection, box });
      }

      let poseMs = 0;
      let people: Person[] = [];
      if (pairs.length > 0) {
        const poseStarted = now();
        const poses = await stages.estimate(
          msg.image,
          pairs.map((p) => p.box),
        );
        poseMs = now() - poseStarted;
        people = pairs.map(({ detection, box }, i) => ({
          box: detection.box,
          score: detection.score,
          keypoints: poses[i]
            ? toSourceKeypoints(
                poses[i].keypoints,
                poses[i].scores,
                poses[i].labels,
                box,
              )
            : [],
        }));
      }

      post({
        type: "result",
        id: msg.id,
        result: {
          people,
          detected: personBoxes(detections, {
            threshold: msg.threshold,
            maxPeople: Number.MAX_SAFE_INTEGER,
          }).length,
          detectMs,
          poseMs,
        },
      });
    } catch (error) {
      post({ type: "error", id: msg.id, error: errMessage(error) });
    }
  };
}

/**
 * Free both models. `allSettled`, not `all`: a detector whose teardown throws
 * must not skip the pose model's, which is the larger of the two.
 */
async function disposeBoth(stages: PoseStages | null): Promise<void> {
  if (!stages) return;
  await Promise.allSettled([
    stages.disposeDetector?.(),
    stages.disposePose?.(),
  ]);
}

export { DETECTOR_THRESHOLD };

function now(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

function errMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
