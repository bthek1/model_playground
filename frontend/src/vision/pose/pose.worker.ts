// The pose worker: wires the engine to `self` and supplies the two real stages.
// The only file in `vision/pose/` that imports the runtime.
//
// Four details that are silent when wrong:
//
//   two loads, one bar    both models are fetched with `Promise.all`, so their
//                         progress events interleave and `model/progress.ts`
//                         sees both denominators before either file finishes.
//                         Loaded in sequence the bar reaches 100%, then starts
//                         again — the "two competing bars" the plan warns about.
//   percentage: false     the detector's boxes must be absolute pixels, for the
//                         same reason as `/object-detection`: `cropBox` and
//                         `drawBoxes` both want pixels, and 0–1 fractions
//                         collapse every crop into the top-left corner.
//   crops, not the image  `AutoProcessor` for ViTPose resizes its input to
//                         256x192. Handing it the whole frame makes every person
//                         a few pixels tall; the top-down design exists because
//                         a crop fills that box with one person.
//   the origin is ours    `post_process_pose_estimation` scales the heatmap peak
//                         by the box's *size* and never adds its *origin*. See
//                         `pose.ts` — that addition is the correctness surface
//                         of this route.

import {
  AutoModel,
  AutoProcessor,
  pipeline,
  RawImage,
} from "@huggingface/transformers";

import type { Detection } from "../draw";
import { fromPayload } from "../image";
import { createPoseHandler, DETECTOR_THRESHOLD, type RawPose } from "./poseEngine";
import type { PoseRequest, PoseResponse } from "./types";

const ctx = self as unknown as {
  onmessage: ((event: MessageEvent<PoseRequest>) => void) | null;
  postMessage: (message: PoseResponse, transfer?: Transferable[]) => void;
};

type PoseAnswer = {
  keypoints: [number, number][];
  scores: number[];
  labels: number[];
};

const handle = createPoseHandler(
  (message, transfer) => ctx.postMessage(message, transfer),
  async ({ device, entry, progress_callback }) => {
    const backend = device as "webgpu" | "wasm";
    const dtype = backend === "webgpu" ? "fp16" : "q8";

    // Together, not in sequence — see the header.
    const [detector, poseProcessor, poseModel] = await Promise.all([
      pipeline("object-detection", entry.detector.id, {
        device: backend,
        dtype: entry.detector.dtypes?.[backend] ?? dtype,
        progress_callback,
      }),
      AutoProcessor.from_pretrained(entry.pose.id),
      AutoModel.from_pretrained(entry.pose.id, {
        device: backend,
        dtype: entry.pose.dtypes?.[backend] ?? dtype,
        progress_callback,
      }),
    ]);

    const detect = detector as unknown as (
      image: unknown,
      opts: { threshold: number; percentage: boolean },
    ) => Promise<Detection[]>;
    const proc = poseProcessor as unknown as {
      (images: unknown): Promise<Record<string, unknown>>;
      post_process_pose_estimation: (
        heatmaps: unknown,
        boxes: number[][][],
      ) => PoseAnswer[][];
    };
    const net = poseModel as unknown as {
      (inputs: Record<string, unknown>): Promise<{ heatmaps: unknown }>;
      dispose: () => Promise<void>;
    };

    return {
      detect: async (payload) =>
        detect(fromPayload(payload), {
          threshold: DETECTOR_THRESHOLD,
          percentage: false,
        }),

      estimate: async (payload, boxes): Promise<RawPose[]> => {
        if (boxes.length === 0) return [];
        const image = fromPayload(payload);
        const crops = await Promise.all(
          boxes.map(([x, y, w, h]) =>
            (image as RawImage).crop([x, y, x + w, y + h]),
          ),
        );
        const inputs = await proc(crops);
        const { heatmaps } = await net(inputs);
        // One box per batch entry: each crop's heatmap belongs to exactly one
        // person, and `post_process_pose_estimation` would otherwise reuse the
        // same heatmap for every box in the list.
        const results = proc.post_process_pose_estimation(
          heatmaps,
          boxes.map((box) => [[...box]]),
        );
        return results.map((perImage) => {
          const first = perImage[0];
          return {
            keypoints: first?.keypoints ?? [],
            scores: first?.scores ?? [],
            labels: first?.labels ?? [],
          };
        });
      },

      disposeDetector: async () => {
        await (detector as unknown as { dispose: () => Promise<void> }).dispose();
      },
      disposePose: async () => {
        await net.dispose();
      },
    };
  },
);

ctx.onmessage = (event) => {
  void handle(event.data);
};
