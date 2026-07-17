// Dedicated Web Worker for Text-to-Speech. Two engines behind the uniform
// `TtsSynthesizer` interface: kokoro-js (Kokoro-82M) and the Transformers.js
// `text-to-speech` pipeline (MMS-VITS, SpeechT5 with speaker embeddings). Kept
// separate from the generic pipeline worker so kokoro-js lands in its own bundle.
// Message-handling logic lives in `ttsEngine.ts` (unit-tested there); this file
// only wires it to `self` and supplies the real synthesizer factory.
//
// As in `webgpu/worker.ts`, we avoid `/// <reference lib="webworker" />` (it
// collides with the app's DOM lib) and narrow `self` to what we use.

import { pipeline } from "@huggingface/transformers";
import { KokoroTTS } from "kokoro-js";

import { TTS_MODELS, type TtsRequest, type TtsResponse } from "./tts";
import { createTtsHandler, type TtsSynthesizer } from "./ttsEngine";

const ctx = self as unknown as {
  onmessage: ((event: MessageEvent<TtsRequest>) => void) | null;
  postMessage: (message: TtsResponse, transfer?: Transferable[]) => void;
};

/** Flatten kokoro-js RawAudio chunks into one contiguous Float32Array. */
function flatten(audio: Float32Array | Float32Array[]): Float32Array {
  if (!Array.isArray(audio)) return audio;
  const out = new Float32Array(audio.reduce((n, c) => n + c.length, 0));
  let offset = 0;
  for (const chunk of audio) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

const handle = createTtsHandler(
  (message, transfer) => ctx.postMessage(message, transfer),
  async (model, opts) => {
    const meta = TTS_MODELS.find((m) => m.id === model);

    if (meta?.engine === "kokoro") {
      const tts = await KokoroTTS.from_pretrained(model, {
        dtype: opts.dtype as "fp16" | "q8",
        device: opts.device as "webgpu" | "wasm",
        progress_callback: opts.progress_callback,
      });
      const synth: TtsSynthesizer = async (text, runOpts) => {
        const out = await tts.generate(text, {
          voice: runOpts?.voice,
          speed: runOpts?.speed,
        } as Parameters<typeof tts.generate>[1]);
        return { audio: flatten(out.audio), sampleRate: out.sampling_rate };
      };
      synth.dispose = async () => {
        await tts.model.dispose();
      };
      return synth;
    }

    // Transformers.js `text-to-speech` pipeline (MMS-VITS, SpeechT5).
    const pipe = await pipeline("text-to-speech", model, {
      device: opts.device as "webgpu" | "wasm",
      dtype: opts.dtype as "fp16" | "q8",
      progress_callback: opts.progress_callback,
    });
    const synth: TtsSynthesizer = async (text) => {
      const out = (await pipe(
        text,
        meta?.speakerEmbeddings
          ? { speaker_embeddings: meta.speakerEmbeddings }
          : {},
      )) as { audio: Float32Array; sampling_rate: number };
      return { audio: out.audio, sampleRate: out.sampling_rate };
    };
    synth.dispose = () => pipe.dispose();
    return synth;
  },
);

ctx.onmessage = (event) => {
  void handle(event.data);
};
