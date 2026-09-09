// The zero-shot worker: wires the engine to `self` and supplies the real towers.
// Everything family-specific lives here, because this is the only file that
// imports the runtime.
//
// Three details that are not interchangeable between the two families, all taken
// from `@huggingface/transformers`' own `CLIPModel` / `SiglipModel` docs:
//
//   tokenizer padding   CLIP pads to the longest prompt; **SigLIP pads to
//                       `max_length`**, and gets different embeddings if it does
//                       not — its position embeddings assume the full width.
//   text output name    CLIP's projection head emits `text_embeds`; SigLIP has
//                       no projection and emits `pooler_output`.
//   image output name   the same split, `image_embeds` vs `pooler_output`.
//
// Both towers are loaded from the *same* repo — `text_model.onnx` and
// `vision_model.onnx`, which is why `zeroShot.ts` measures its `bytes` from the
// full-model files and the real download is close to it.

import {
  AutoProcessor,
  AutoTokenizer,
  CLIPTextModelWithProjection,
  CLIPVisionModelWithProjection,
  SiglipTextModel,
  SiglipVisionModel,
} from "@huggingface/transformers";

import { fromPayload } from "../image";
import { createZeroShotHandler, type Embeddings } from "./engine";
import type { ZeroShotRequest, ZeroShotResponse } from "./types";

const ctx = self as unknown as {
  onmessage: ((event: MessageEvent<ZeroShotRequest>) => void) | null;
  postMessage: (message: ZeroShotResponse, transfer?: Transferable[]) => void;
};

type AnyTensor = { data: ArrayLike<number>; dims: number[] };

/** Pull the embedding matrix out of whichever output name the family uses. */
function embeddingsOf(
  output: Record<string, unknown>,
  names: readonly string[],
): Embeddings {
  for (const name of names) {
    const tensor = output[name] as AnyTensor | undefined;
    if (tensor?.dims?.length === 2) {
      const [rows, dim] = tensor.dims;
      return { data: tensor.data, rows, dim };
    }
  }
  throw new Error(
    `Tower produced none of ${names.join(", ")} — got ${Object.keys(output).join(", ")}`,
  );
}

const handle = createZeroShotHandler(
  (message, transfer) => ctx.postMessage(message, transfer),
  async (model, opts) => {
    const isSiglip = opts.family === "siglip";
    const load = {
      device: opts.device as "webgpu" | "wasm",
      dtype: opts.dtype as "fp16" | "q8" | "fp32",
      progress_callback: opts.progress_callback,
    };

    const [tokenizer, processor, textModel, visionModel] = await Promise.all([
      AutoTokenizer.from_pretrained(model),
      AutoProcessor.from_pretrained(model),
      isSiglip
        ? SiglipTextModel.from_pretrained(model, load)
        : CLIPTextModelWithProjection.from_pretrained(model, load),
      isSiglip
        ? SiglipVisionModel.from_pretrained(model, load)
        : CLIPVisionModelWithProjection.from_pretrained(model, load),
    ]);

    const textNames = isSiglip
      ? (["pooler_output", "text_embeds"] as const)
      : (["text_embeds", "pooler_output"] as const);
    const imageNames = isSiglip
      ? (["pooler_output", "image_embeds"] as const)
      : (["image_embeds", "pooler_output"] as const);

    return {
      encodeText: async (prompts) => {
        const inputs = tokenizer(prompts, {
          // Not a stylistic choice — see the header.
          padding: isSiglip ? "max_length" : true,
          truncation: true,
        });
        const out = (await textModel(inputs)) as Record<string, unknown>;
        return embeddingsOf(out, textNames);
      },
      encodeImage: async (payload) => {
        const { pixel_values } = await processor(fromPayload(payload));
        const out = (await visionModel({ pixel_values })) as Record<
          string,
          unknown
        >;
        return embeddingsOf(out, imageNames);
      },
      dispose: async () => {
        await Promise.all([
          (textModel as unknown as { dispose: () => Promise<void> }).dispose(),
          (visionModel as unknown as { dispose: () => Promise<void> }).dispose(),
        ]);
      },
    };
  },
);

ctx.onmessage = (event) => {
  void handle(event.data);
};
