// The ONNX Runtime boundary. ORT itself is mocked — the point here is
// everything *around* the session: what gets downloaded, what is refused, which
// providers are asked for, and the tensor contract the graph is held to.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import golden from "./__fixtures__/deepFilterNetFeatures.json";
import { AUX_BYTES, ERB_BANDS } from "./aux";
import { DF_ORDER } from "./deepFilterNet";
import { DF_BINS } from "./features";
import { FFT_BINS } from "./stft";

const createSession = vi.fn();
const release = vi.fn();

class FakeTensor {
  constructor(
    public type: string,
    public data: Float32Array,
    public dims: number[],
  ) {}
}

vi.mock("onnxruntime-web/webgpu", () => ({
  InferenceSession: {
    create: (...args: unknown[]) => createSession(...args),
  },
  Tensor: FakeTensor,
}));

const { loadDeepFilterNet } = await import("./session");

/** A byte-exact stand-in for `deepfilter-auxiliary.bin`. */
function auxBuffer(): ArrayBuffer {
  const buf = new ArrayBuffer(AUX_BYTES);
  const all = new Float32Array(buf);
  const matrix = FFT_BINS * ERB_BANDS;
  let bin = 0;
  for (let band = 0; band < ERB_BANDS; band++) {
    for (let j = 0; j < golden.widths[band]; j++, bin++) {
      all[bin * ERB_BANDS + band] = 1 / golden.widths[band];
      all[matrix + band * FFT_BINS + bin] = 1;
    }
  }
  all.set(golden.window, matrix * 2);
  return buf;
}

interface RunOutputs {
  erb_mask: { data: Float32Array };
  df_coefs: { data: Float32Array };
}

let run: ReturnType<typeof vi.fn>;

/** Serve the aux file (optionally corrupted) and a dummy graph. */
function stubFetch(aux: ArrayBuffer = auxBuffer(), { stream = false } = {}) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      const body = url.includes("auxiliary") ? aux : new ArrayBuffer(64);
      if (!stream) {
        return {
          ok: true,
          headers: { get: () => null },
          body: null,
          arrayBuffer: async () => body,
        };
      }
      // Two chunks, so progress is reported more than once.
      const bytes = new Uint8Array(body);
      const half = Math.ceil(bytes.length / 2);
      let sent = 0;
      return {
        ok: true,
        headers: { get: () => String(bytes.length) },
        body: {
          getReader: () => ({
            read: async () => {
              if (sent >= bytes.length) return { done: true, value: undefined };
              const value = bytes.slice(sent, sent + half);
              sent += half;
              return { done: false, value };
            },
          }),
        },
        arrayBuffer: async () => body,
      };
    }),
  );
}

describe("loadDeepFilterNet", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    run = vi.fn(
      async (): Promise<RunOutputs> => ({
        erb_mask: { data: new Float32Array(ERB_BANDS) },
        df_coefs: { data: new Float32Array(DF_ORDER * 1 * DF_BINS * 2) },
      }),
    );
    createSession.mockResolvedValue({ run, release });
    stubFetch();
  });

  afterEach(() => vi.unstubAllGlobals());

  it("fetches the graph and its constants from the model's Hub repo", async () => {
    await loadDeepFilterNet("soniqo/DeepFilterNet3-ONNX");
    const urls = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls.map(
      (c) => c[0],
    );
    expect(urls).toEqual(
      expect.arrayContaining([
        "https://huggingface.co/soniqo/DeepFilterNet3-ONNX/resolve/main/deepfilter.onnx",
        "https://huggingface.co/soniqo/DeepFilterNet3-ONNX/resolve/main/deepfilter-auxiliary.bin",
      ]),
    );
  });

  it("asks for WebGPU first and WASM second — the fallback is the provider list", async () => {
    await loadDeepFilterNet("repo");
    expect(createSession).toHaveBeenCalledWith(
      expect.anything(),
      { executionProviders: ["webgpu", "wasm"] },
    );
  });

  it("honours a forced backend, so a spec can exercise one path", async () => {
    const session = await loadDeepFilterNet("repo", undefined, "wasm");
    expect(createSession).toHaveBeenCalledWith(expect.anything(), {
      executionProviders: ["wasm"],
    });
    expect(session.backend).toBe("wasm");
  });

  it("reports download progress per file", async () => {
    stubFetch(auxBuffer(), { stream: true });
    const seen: string[] = [];
    await loadDeepFilterNet("repo", (p) => {
      seen.push(p.file);
      expect(p.progress).toBeGreaterThan(0);
      expect(p.progress).toBeLessThanOrEqual(100);
    });
    expect(new Set(seen)).toEqual(
      new Set(["deepfilter.onnx", "deepfilter-auxiliary.bin"]),
    );
  });

  it("refuses a wrong-sized constants file rather than emit noise", async () => {
    stubFetch(new ArrayBuffer(AUX_BYTES - 4));
    await expect(loadDeepFilterNet("repo")).rejects.toThrow(
      /refusing to run rather than emit noise/,
    );
    expect(createSession).not.toHaveBeenCalled();
  });

  it("propagates a failed download instead of loading half a model", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({ ok: false, status: 401, statusText: "Unauthorized" })),
    );
    await expect(loadDeepFilterNet("repo")).rejects.toThrow(/401 Unauthorized/);
  });

  it("feeds the graph the tensor shapes its contract declares", async () => {
    const session = await loadDeepFilterNet("repo");
    const frames = 12;
    run.mockResolvedValue({
      erb_mask: { data: new Float32Array(frames * ERB_BANDS) },
      df_coefs: { data: new Float32Array(DF_ORDER * frames * DF_BINS * 2) },
    });

    await session.infer(
      new Float32Array(frames * ERB_BANDS),
      new Float32Array(2 * frames * DF_BINS),
      frames,
    );

    const feeds = run.mock.calls[0][0] as Record<string, FakeTensor>;
    expect(feeds.feat_erb.dims).toEqual([1, 1, frames, ERB_BANDS]);
    expect(feeds.feat_spec.dims).toEqual([1, 2, frames, DF_BINS]);
    expect(feeds.feat_erb.type).toBe("float32");
  });

  it("catches a changed df_coefs contract instead of misreading the buffer", async () => {
    const session = await loadDeepFilterNet("repo");
    run.mockResolvedValue({
      erb_mask: { data: new Float32Array(4 * ERB_BANDS) },
      df_coefs: { data: new Float32Array(4 * DF_BINS * 2) }, // missing the order axis
    });
    await expect(
      session.infer(new Float32Array(4 * ERB_BANDS), new Float32Array(8 * DF_BINS), 4),
    ).rejects.toThrow(/tensor contract changed/);
  });

  it("releases the session on dispose", async () => {
    const session = await loadDeepFilterNet("repo");
    await session.dispose();
    expect(release).toHaveBeenCalledTimes(1);
  });
});
