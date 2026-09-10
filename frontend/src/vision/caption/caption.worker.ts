// The image-to-text worker: wires the engine to `self` and supplies the two
// real captioners. The only file in `vision/caption/` that imports the runtime.
//
// Two families behind one `Captioner` interface, the same trade `tts.worker.ts`
// makes for Kokoro / MMS / MusicGen:
//
//   florence2                `Florence2ForConditionalGeneration` +
//                            `Florence2Processor`. The task token goes through
//                            `construct_prompts` into real text, and the answer
//                            comes back through `post_process_generation`.
//   vision-encoder-decoder   `AutoModelForVision2Seq` + `AutoProcessor`. One
//                            mode, `<CAPTION>`, and no task token at all.
//
// Four details that are silent when wrong:
//
//   skip_special_tokens      must be **false** for Florence-2. Its answers are
//                            special tokens — `<loc_412>` for every box
//                            coordinate — and stripping them returns a caption
//                            with the boxes silently deleted rather than an
//                            error.
//   image_size order         `post_process_generation` maps coordinates with
//                            `image_size[i % 2]` over an `x, y, x, y` sequence,
//                            so it wants **[width, height]** — `RawImage.size`.
//                            The method's own JSDoc says "height x width", and
//                            following it transposes every box on a
//                            non-square picture. `original_sizes` is
//                            [height, width]; passing it here is the trap.
//   the prompt is text       `construct_prompts('<OCR>')` resolves the token to
//                            "What is the text in the image?" from the model's
//                            own `preprocessor_config.json`. Sending the raw
//                            token as the prompt tokenizes it as literal
//                            characters and the model answers something fluent
//                            and unrelated.
//   box coordinates          come back in **source pixels**, which is what
//                            `drawBoxes` wants — no `percentage` flag here, and
//                            no scaling to undo.

import {
  AutoProcessor,
  AutoModelForVision2Seq,
  Florence2ForConditionalGeneration,
} from "@huggingface/transformers";

import { fromPayload } from "../image";
import { createCaptionHandler, type CaptionOutput } from "./engine";
import { toDetections } from "./grounding";
import { isBoxMode, type CaptionRequest, type CaptionResponse } from "./types";

const ctx = self as unknown as {
  onmessage: ((event: MessageEvent<CaptionRequest>) => void) | null;
  postMessage: (message: CaptionResponse, transfer?: Transferable[]) => void;
};

const handle = createCaptionHandler(
  (message, transfer) => ctx.postMessage(message, transfer),
  async (model, opts) => {
    const load = {
      device: opts.device as "webgpu" | "wasm",
      dtype: opts.dtype as "fp16" | "q8" | "fp32",
      progress_callback: opts.progress_callback,
    };

    if (opts.family === "florence2") {
      const [processor, net] = await Promise.all([
        AutoProcessor.from_pretrained(model),
        Florence2ForConditionalGeneration.from_pretrained(model, load),
      ]);
      const proc = processor as unknown as {
        (image: unknown, text?: unknown): Promise<Record<string, unknown>>;
        construct_prompts: (task: string) => string[];
        post_process_generation: (
          text: string,
          task: string,
          imageSize: [number, number],
        ) => Record<string, unknown>;
        batch_decode: (
          ids: unknown,
          opts: { skip_special_tokens: boolean },
        ) => string[];
      };
      const gen = net as unknown as {
        generate: (args: Record<string, unknown>) => Promise<unknown>;
        dispose: () => Promise<void>;
      };

      return {
        generate: async (payload, mode, maxNewTokens): Promise<CaptionOutput> => {
          const image = fromPayload(payload);
          const prompts = proc.construct_prompts(mode);
          const inputs = await proc(image, prompts);
          const ids = await gen.generate({ ...inputs, max_new_tokens: maxNewTokens });
          // `false`, not `true`: the `<loc_…>` tokens *are* the answer for the
          // box modes, and post-processing needs them.
          const [text] = proc.batch_decode(ids, { skip_special_tokens: false });
          const parsed = proc.post_process_generation(text, mode, [
            image.width,
            image.height,
          ]);
          const answer = parsed[mode];
          return isBoxMode(mode)
            ? { kind: "boxes", detections: toDetections(answer) }
            : { kind: "text", text: String(answer ?? "").trim() };
        },
        dispose: () => gen.dispose(),
      };
    }

    // vision-encoder-decoder: one mode, no task token, `pixel_values` only.
    const [processor, net] = await Promise.all([
      AutoProcessor.from_pretrained(model),
      AutoModelForVision2Seq.from_pretrained(model, load),
    ]);
    const proc = processor as unknown as {
      (image: unknown): Promise<{ pixel_values: unknown }>;
      batch_decode: (
        ids: unknown,
        opts: { skip_special_tokens: boolean },
      ) => string[];
    };
    const gen = net as unknown as {
      generate: (args: Record<string, unknown>) => Promise<unknown>;
      dispose: () => Promise<void>;
    };

    return {
      generate: async (payload, _mode, maxNewTokens): Promise<CaptionOutput> => {
        const { pixel_values } = await proc(fromPayload(payload));
        const ids = await gen.generate({
          inputs: pixel_values,
          max_new_tokens: maxNewTokens,
        });
        const [text] = proc.batch_decode(ids, { skip_special_tokens: true });
        return { kind: "text", text: (text ?? "").trim() };
      },
      dispose: () => gen.dispose(),
    };
  },
);

ctx.onmessage = (event) => {
  void handle(event.data);
};
