// The ONNX Runtime side of Silero-VAD — the app's second direct ORT caller,
// after `enhance/session.ts`.
//
// The `/webgpu` subpath is the same deliberate choice made there: it is the exact
// entry point `@huggingface/transformers` imports, so Vite resolves both to one
// module and emits **one** ORT WASM asset instead of two.
//
// **This model runs on WASM, on purpose — it is not a fallback.** Three reasons,
// in the order they were checked:
//
//   1. It is already fast. Measured on the 11 s jfk.wav clip: 343 frames in
//      102 ms single-threaded, 0.30 ms per 32 ms frame — about 100x real time.
//      There is no user-visible latency left for a GPU to remove.
//   2. The work per call is tiny (a 576-sample window) and there are hundreds of
//      calls. A WebGPU dispatch plus the readback of a single float would cost
//      more per frame than the whole frame costs on CPU.
//   3. The graph is LSTM- and `If`-heavy — neither is covered by ORT's WebGPU
//      execution provider, so it would partition across both devices and copy
//      per frame anyway.
//
// So the picker never offers a device choice for this task, and `backend` is
// reported as `wasm` rather than claiming a GPU the model does not touch.

import * as ort from "onnxruntime-web/webgpu";

import type { Backend } from "../backend";
import { energyProbabilities } from "./energyVad";
import { ENERGY_VAD_MODEL, FRAME_SAMPLES, SAMPLE_RATE } from "./types";
import { frameProbabilities, STATE_SIZE, type FrameInfer } from "./vad";

const HF_BASE = "https://huggingface.co";

export interface LoadProgress {
  file: string;
  loaded: number;
  total: number;
  /** 0-100, to match Transformers.js's `progress_callback` payload. */
  progress: number;
}

export interface VadSession {
  /** Speech probability per frame for a whole clip. State is reset per call. */
  probabilities: (audio: Float32Array) => Promise<Float32Array>;
  backend: Backend;
  dispose: () => Promise<void>;
}

/** Fetch a file from the Hub, reporting bytes as they arrive. */
async function fetchWithProgress(
  url: string,
  file: string,
  onProgress?: (p: LoadProgress) => void,
): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${file}: ${response.status} ${response.statusText}`);
  }
  const total = Number(response.headers.get("content-length") ?? 0);
  if (!response.body || !onProgress || !total) return response.arrayBuffer();

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress({ file, loaded, total, progress: (loaded / total) * 100 });
  }
  const out = new Uint8Array(loaded);
  let at = 0;
  for (const chunk of chunks) {
    out.set(chunk, at);
    at += chunk.length;
  }
  return out.buffer;
}

/** The energy baseline as a session, so the engine has one shape to drive. */
function energySession(): VadSession {
  return {
    probabilities: (audio) =>
      Promise.resolve(energyProbabilities(audio, FRAME_SAMPLES)),
    backend: "wasm",
    dispose: () => Promise.resolve(),
  };
}

/**
 * Download the graph and open a session.
 *
 * The `sr` input is an int64 **scalar** (empty dims), not a 1-element vector —
 * ORT rejects the latter — and it is rebuilt per call because a tensor cannot be
 * reused across `run()`s once its buffer has been handed to the runtime.
 */
export async function loadVad(
  model: string,
  repo: string | null,
  file = "onnx/model.onnx",
  onProgress?: (p: LoadProgress) => void,
): Promise<VadSession> {
  if (model === ENERGY_VAD_MODEL || !repo) return energySession();

  const buffer = await fetchWithProgress(
    `${HF_BASE}/${repo}/resolve/main/${file}`,
    file,
    onProgress,
  );

  const session = await ort.InferenceSession.create(buffer, {
    executionProviders: ["wasm"],
  });

  // The graph's contract, asserted once at load rather than per frame. A repo
  // that re-exports v4 (separate `h`/`c` inputs) would otherwise fail 300 times
  // with an opaque ORT message on the first clip the user tries.
  for (const name of ["input", "state", "sr"]) {
    if (!session.inputNames.includes(name)) {
      throw new Error(
        `${repo} has inputs [${session.inputNames.join(", ")}] — expected a ` +
          `Silero v5 graph with [input, state, sr]`,
      );
    }
  }

  const infer: FrameInfer = async (window, state) => {
    const out = await session.run({
      input: new ort.Tensor("float32", window, [1, window.length]),
      state: new ort.Tensor("float32", state, [2, 1, 128]),
      sr: new ort.Tensor("int64", BigInt64Array.from([BigInt(SAMPLE_RATE)]), []),
    });
    const next = out.stateN.data as Float32Array;
    if (next.length !== STATE_SIZE) {
      throw new Error(
        `stateN has ${next.length} values, expected ${STATE_SIZE} — the graph's ` +
          `tensor contract changed`,
      );
    }
    return { probability: (out.output.data as Float32Array)[0], state: next };
  };

  return {
    probabilities: (audio) => frameProbabilities(audio, infer),
    backend: "wasm",
    dispose: async () => {
      await session.release();
    },
  };
}
