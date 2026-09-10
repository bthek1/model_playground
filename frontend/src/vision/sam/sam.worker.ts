// The SAM worker: wires the engine to `self` and supplies the real two graphs.
// The only file in `vision/sam/` that imports the runtime, which is what keeps
// `sam.ts` and `samEngine.ts` cheap to unit-test.
//
// Four details that are easy to get subtly wrong, and silent when you do:
//
//   the split          `get_image_embeddings` runs `vision_encoder` alone;
//                      calling the model with `image_embeddings` already present
//                      makes `forward` skip the encoder and run only
//                      `prompt_encoder_mask_decoder`. That branch *is* the
//                      encode-once/decode-many optimisation — omit the
//                      embeddings and every click silently re-encodes at full
//                      cost while still returning correct masks.
//   point coordinates  clicks are in source-image pixels; the model wants them
//                      in the processor's resized space. `reshape_input_points`
//                      does that conversion using the sizes the *processor*
//                      reported, so the encode's `inputs` are kept alongside the
//                      embeddings rather than discarded.
//   label dtype        `input_labels` is **int64** (`BigInt64Array`), matching
//                      the `ones()` default the model falls back to. A
//                      Float32Array here fails inside ONNX Runtime with a shape
//                      error that names neither the tensor nor the cause.
//   mask ownership     `post_process_masks` returns a fresh tensor per call, so
//                      its buffer is ours to transfer. `.slice()` copies the
//                      per-candidate view out of the packed `[1, 3, H, W]`
//                      buffer, which is what makes each mask individually
//                      transferable.

import {
  AutoProcessor,
  SamModel,
  Tensor,
  type PreTrainedModel,
} from "@huggingface/transformers";

import { fromPayload } from "../image";
import { createSamHandler } from "./samEngine";
import type { SamMask, SamPoint, SamRequest, SamResponse } from "./types";

const ctx = self as unknown as {
  onmessage: ((event: MessageEvent<SamRequest>) => void) | null;
  postMessage: (message: SamResponse, transfer?: Transferable[]) => void;
};

/** What one encode leaves behind: the embeddings plus the sizes to map clicks. */
interface Encoded {
  embeddings: Record<string, unknown>;
  originalSizes: [number, number][];
  reshapedSizes: [number, number][];
}

type AnyTensor = { data: ArrayLike<number>; dims: number[] };

const handle = createSamHandler(
  (message, transfer) => ctx.postMessage(message, transfer),
  async (model, opts) => {
    const load = {
      device: opts.device as "webgpu" | "wasm",
      dtype: opts.dtype as "fp16" | "q8" | "fp32",
      progress_callback: opts.progress_callback,
    };
    const [processor, sam] = await Promise.all([
      AutoProcessor.from_pretrained(model),
      SamModel.from_pretrained(model, load),
    ]);
    const proc = processor as unknown as {
      (image: unknown): Promise<Record<string, unknown>>;
      reshape_input_points: (
        points: number[][][],
        originalSizes: [number, number][],
        reshapedSizes: [number, number][],
      ) => unknown;
      post_process_masks: (
        masks: unknown,
        originalSizes: [number, number][],
        reshapedSizes: [number, number][],
      ) => Promise<AnyTensor[]>;
    };
    const net = sam as unknown as PreTrainedModel & {
      get_image_embeddings: (
        inputs: Record<string, unknown>,
      ) => Promise<Record<string, unknown>>;
      (inputs: Record<string, unknown>): Promise<{
        pred_masks: AnyTensor;
        iou_scores: AnyTensor;
      }>;
    };

    return {
      encode: async (payload): Promise<Encoded> => {
        const inputs = await proc(fromPayload(payload));
        return {
          embeddings: await net.get_image_embeddings(inputs),
          originalSizes: inputs.original_sizes as [number, number][],
          reshapedSizes: inputs.reshaped_input_sizes as [number, number][],
        };
      },

      decode: async (encoded, points): Promise<SamMask[]> => {
        const { embeddings, originalSizes, reshapedSizes } = encoded as Encoded;

        // `[point_batch_size, nb_points, 2]` — one point batch, N clicks.
        const input_points = proc.reshape_input_points(
          [points.map((p: SamPoint) => [p.x, p.y])],
          originalSizes,
          reshapedSizes,
        );
        const input_labels = new Tensor(
          "int64",
          BigInt64Array.from(points.map((p: SamPoint) => (p.positive ? 1n : 0n))),
          [1, 1, points.length],
        );

        const { pred_masks, iou_scores } = await net({
          ...embeddings,
          input_points,
          input_labels,
        });
        const [masks] = await proc.post_process_masks(
          pred_masks,
          originalSizes,
          reshapedSizes,
        );

        // `[1, candidates, height, width]`, one byte per pixel.
        const [, candidates, height, width] = masks.dims;
        const data = masks.data as unknown as Uint8Array;
        const scores = iou_scores.data;
        const stride = height * width;

        const out: SamMask[] = [];
        for (let c = 0; c < candidates; c++) {
          out.push({
            data: data.slice(c * stride, (c + 1) * stride),
            width,
            height,
            score: Number(scores[c] ?? 0),
          });
        }
        // Best first: the page shows the winner and offers the other two, and
        // "the candidates are the interesting part" only reads that way if the
        // first one is the one the model would have chosen.
        out.sort((a, b) => b.score - a.score);
        return out;
      },

      dispose: async () => {
        await (sam as unknown as { dispose: () => Promise<void> }).dispose();
      },
    };
  },
);

ctx.onmessage = (event) => {
  void handle(event.data);
};
