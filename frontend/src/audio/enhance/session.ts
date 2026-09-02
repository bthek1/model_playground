// The ONNX Runtime Web side of DeepFilterNet3 — the only place in the app that
// talks to `onnxruntime-web` directly.
//
// Why direct: DeepFilterNet3 has no Transformers.js task. The repo publishes a
// bare graph with hand-rolled feature tensors, so there is no pipeline to wrap.
// We import the copy `@huggingface/transformers` already depends on (pinned to
// the same version in package.json, so npm dedupes it) rather than a second
// runtime — two ORT builds would double the WASM payload for no benefit.
//
// The `/webgpu` subpath is deliberate — it is the exact entry point
// `@huggingface/transformers` imports, so Vite resolves both to one module and
// emits **one** ORT WASM asset instead of two (the bare `onnxruntime-web` entry
// pulls a different, 26 MB build). Its `.bundle.` variant locates its own WASM,
// so there is no `env.wasm.wasmPaths` to configure and nothing to copy into
// `public/`.

import * as ort from "onnxruntime-web/webgpu";

import type { Backend } from "../backend";
import { AUX_BYTES, ERB_BANDS, parseAux, type DeepFilterAux } from "./aux";
import { DF_ORDER, type DeepFilterInference, type DeepFilterOutputs } from "./deepFilterNet";
import { DF_BINS } from "./features";

const HF_BASE = "https://huggingface.co";

export interface LoadProgress {
  file: string;
  loaded: number;
  total: number;
  /** 0-100, to match Transformers.js's `progress_callback` payload. */
  progress: number;
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

export interface DeepFilterSession {
  aux: DeepFilterAux;
  infer: DeepFilterInference;
  backend: Backend;
  dispose: () => Promise<void>;
}

/**
 * Download the graph plus its auxiliary constants and open a session.
 *
 * The provider list is ordered, not a preference hint: ORT walks it and uses the
 * first that initialises, so a machine without WebGPU silently lands on WASM.
 * `backend` forces one (the E2E specs use it to exercise both paths).
 */
export async function loadDeepFilterNet(
  repo: string,
  onProgress?: (p: LoadProgress) => void,
  backend?: Backend,
): Promise<DeepFilterSession> {
  const base = `${HF_BASE}/${repo}/resolve/main`;
  const [auxBuffer, modelBuffer] = await Promise.all([
    fetchWithProgress(`${base}/deepfilter-auxiliary.bin`, "deepfilter-auxiliary.bin", onProgress),
    fetchWithProgress(`${base}/deepfilter.onnx`, "deepfilter.onnx", onProgress),
  ]);

  if (auxBuffer.byteLength !== AUX_BYTES) {
    throw new Error(
      `deepfilter-auxiliary.bin is ${auxBuffer.byteLength} bytes, expected ${AUX_BYTES} — ` +
        `refusing to run rather than emit noise`,
    );
  }
  const aux = parseAux(auxBuffer);

  const providers: Backend[] = backend ? [backend] : ["webgpu", "wasm"];
  const session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: providers,
  });
  // ORT does not report which provider it settled on, so with the default list
  // we can only say what was available. Ask the same question `pickBackend`
  // does rather than claim WebGPU on a machine that has none.
  const resolved: Backend =
    backend ??
    (typeof navigator !== "undefined" && "gpu" in navigator && navigator.gpu
      ? "webgpu"
      : "wasm");

  const infer: DeepFilterInference = async (featErb, featSpec, frames) => {
    const feeds = {
      feat_erb: new ort.Tensor("float32", featErb, [1, 1, frames, ERB_BANDS]),
      feat_spec: new ort.Tensor("float32", featSpec, [1, 2, frames, DF_BINS]),
    };
    const out = await session.run(feeds);
    const erbMask = out.erb_mask.data as Float32Array;
    const dfCoefs = out.df_coefs.data as Float32Array;
    const expected = DF_ORDER * frames * DF_BINS * 2;
    if (dfCoefs.length !== expected) {
      throw new Error(
        `df_coefs has ${dfCoefs.length} values, expected ${expected} — the graph's tensor ` +
          `contract changed`,
      );
    }
    return { erbMask, dfCoefs } satisfies DeepFilterOutputs;
  };

  return {
    aux,
    infer,
    backend: resolved,
    dispose: async () => {
      await session.release();
    },
  };
}
