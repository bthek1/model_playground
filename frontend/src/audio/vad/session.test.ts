// The ONNX Runtime boundary. ORT itself is mocked — the point here is everything
// *around* the session: what gets downloaded (and what doesn't), which providers
// are asked for, the tensor contract the graph is held to, and the fact that the
// energy baseline never touches the network at all.
//
// It cannot tell you the model works. That is `just fe-e2e-vad`'s job, and the
// distinction matters here more than anywhere else in the app: the frame window
// is the one thing a mocked session will happily agree with you about.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const createSession = vi.fn();
const release = vi.fn();

class FakeTensor {
  constructor(
    public type: string,
    public data: Float32Array | BigInt64Array,
    public dims: number[],
  ) {}
}

vi.mock("onnxruntime-web/webgpu", () => ({
  InferenceSession: {
    create: (...args: unknown[]) => createSession(...args),
  },
  Tensor: FakeTensor,
}));

const { loadVad } = await import("./session");
const { ENERGY_VAD_MODEL } = await import("./types");
const { FRAME_SAMPLES, STATE_SIZE, WINDOW_SAMPLES } = await import("./vad");

const SILERO = "onnx-community/silero-vad";

let run: ReturnType<typeof vi.fn>;

/** A session that answers with a fixed probability and a fresh state. */
function fakeGraph(probability = 0.8, stateSize = STATE_SIZE) {
  run = vi.fn(async () => ({
    output: { data: Float32Array.from([probability]) },
    stateN: { data: new Float32Array(stateSize) },
  }));
  return {
    run,
    release,
    inputNames: ["input", "state", "sr"],
    outputNames: ["output", "stateN"],
  };
}

function stubFetch({ ok = true, bytes = 2048 } = {}) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => ({
      ok,
      status: 404,
      statusText: "Not Found",
      headers: { get: () => null },
      body: null,
      arrayBuffer: async () => new ArrayBuffer(bytes),
    })),
  );
}

describe("loadVad", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    createSession.mockResolvedValue(fakeGraph());
    stubFetch();
  });

  afterEach(() => vi.unstubAllGlobals());

  it("fetches the graph from the model's Hub repo", async () => {
    await loadVad(SILERO, SILERO, "onnx/model.onnx");
    expect(fetch).toHaveBeenCalledWith(
      "https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx",
    );
  });

  it("asks for WASM only — the GPU is not a fallback here, it is declined", async () => {
    // A provider list of ["webgpu", "wasm"] would partition this LSTM graph and
    // copy per frame. See the comment at the top of session.ts.
    await loadVad(SILERO, SILERO);
    expect(createSession).toHaveBeenCalledWith(expect.anything(), {
      executionProviders: ["wasm"],
    });
    expect((await loadVad(SILERO, SILERO)).backend).toBe("wasm");
  });

  it("refuses a graph that is not Silero v5, naming what it found", async () => {
    // v4 exported separate `h`/`c` inputs. Failing here beats failing 300 times
    // with an opaque ORT message on the first clip the user tries.
    createSession.mockResolvedValue({
      run: vi.fn(),
      release,
      inputNames: ["input", "h", "c", "sr"],
      outputNames: ["output", "hn", "cn"],
    });
    await expect(loadVad(SILERO, SILERO)).rejects.toThrow(/expected a Silero v5 graph/);
  });

  it("surfaces a 404 as the file that failed", async () => {
    stubFetch({ ok: false });
    await expect(loadVad(SILERO, "onnx-community/nope")).rejects.toThrow(
      /onnx\/model\.onnx: 404 Not Found/,
    );
  });

  it("feeds the graph a 576-sample window and a [2,1,128] state", async () => {
    const session = await loadVad(SILERO, SILERO);
    await session.probabilities(new Float32Array(FRAME_SAMPLES));

    const feeds = run.mock.calls[0][0] as Record<string, FakeTensor>;
    expect(feeds.input.dims).toEqual([1, WINDOW_SAMPLES]);
    expect(feeds.state.dims).toEqual([2, 1, 128]);
    // An int64 *scalar*: empty dims. ORT rejects a 1-element vector here.
    expect(feeds.sr.dims).toEqual([]);
    expect(Array.from(feeds.sr.data as BigInt64Array)).toEqual([16000n]);
  });

  it("rejects a state tensor of the wrong size rather than drifting", async () => {
    createSession.mockResolvedValue(fakeGraph(0.8, 64));
    const session = await loadVad(SILERO, SILERO);
    await expect(session.probabilities(new Float32Array(FRAME_SAMPLES))).rejects.toThrow(
      /stateN has 64 values, expected 256/,
    );
  });

  it("scores one frame per 32 ms of audio", async () => {
    const session = await loadVad(SILERO, SILERO);
    const probs = await session.probabilities(new Float32Array(FRAME_SAMPLES * 3));
    expect(probs).toHaveLength(3);
    expect(run).toHaveBeenCalledTimes(3);
  });

  it("releases the session on dispose", async () => {
    const session = await loadVad(SILERO, SILERO);
    await session.dispose();
    expect(release).toHaveBeenCalledTimes(1);
  });

  describe("the energy baseline", () => {
    it("opens no session and fetches nothing", async () => {
      const session = await loadVad(ENERGY_VAD_MODEL, null);
      expect(fetch).not.toHaveBeenCalled();
      expect(createSession).not.toHaveBeenCalled();
      expect(session.backend).toBe("wasm");
    });

    it("still answers with one probability per frame", async () => {
      const session = await loadVad(ENERGY_VAD_MODEL, null);
      const probs = await session.probabilities(new Float32Array(FRAME_SAMPLES * 4));
      expect(probs).toHaveLength(4);
    });

    it("disposes without throwing", async () => {
      const session = await loadVad(ENERGY_VAD_MODEL, null);
      await expect(session.dispose()).resolves.toBeUndefined();
    });

    it("is chosen by a null repo too, not only by the model id", async () => {
      // The catalogue's `repo: null` is the real discriminator; the id is a
      // convenience. A future baseline must not accidentally hit the network.
      await loadVad("some-other-baseline", null);
      expect(fetch).not.toHaveBeenCalled();
    });
  });
});
