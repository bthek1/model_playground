import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { UseLiveAsrResult } from "@/hooks/useLiveAsr";

// No Web Audio in happy-dom — stub decode/playback/encode. decodeToMono returns
// a fixed clip so the sample-clip flow can be exercised without a real WAV;
// `play` returns a fake AudioContext whose transport calls we can assert on.
const decoded = new Float32Array([0.1, 0.2, 0.3]);
const playCtx = {
  state: "running",
  suspend: vi.fn(),
  resume: vi.fn(),
  close: vi.fn(),
};
const playMock = vi.fn<(...args: unknown[]) => typeof playCtx>(() => playCtx);
vi.mock("@/audio/io", () => ({
  decodeToMono: vi.fn().mockResolvedValue(decoded),
  play: (...args: unknown[]) => playMock(...args),
  toWavBlob: vi.fn(() => new Blob()),
}));

vi.mock("@tanstack/react-router", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    createFileRoute: vi
      .fn()
      .mockImplementation(
        (path: string) => (opts: Record<string, unknown>) => ({
          path,
          options: opts,
        }),
      ),
  };
});

const mockStart = vi.fn();
const mockStop = vi.fn();
const mockTranscribeClip = vi.fn();
const mockLoad = vi.fn();
const mockRetry = vi.fn();
const baseState: UseLiveAsrResult = {
  status: "idle",
  idle: true,
  loading: false,
  ready: false,
  progress: null,
  loadProgress: null,
  loadedInMs: null,
  backend: null,
  recording: false,
  running: false,
  stream: null,
  clip: null,
  sampleRate: 16000,
  text: "",
  chunks: [],
  error: null,
  start: mockStart,
  stop: mockStop,
  transcribeClip: mockTranscribeClip,
  load: mockLoad,
  retry: mockRetry,
  cancel: vi.fn(),
};
let mockState: UseLiveAsrResult = { ...baseState };

// Forwarded verbatim, arity included: the route passes the model and *nothing
// else*, and "there is no second argument" is the guarantee (the hook's own
// `autoLoad` default is `false`).
const useLiveAsrArgs = vi.fn<(...args: unknown[]) => void>();
vi.mock("@/hooks/useLiveAsr", () => ({
  useLiveAsr: (...args: unknown[]) => {
    useLiveAsrArgs(...args);
    return mockState;
  },
}));

// The browser cache probe. Empty unless a test seeds it.
let cached = new Set<string>();
vi.mock("@/model/cache", () => ({
  cachedModels: () => Promise.resolve(cached),
  evictModel: vi.fn(() => Promise.resolve()),
}));

const { Route } = await import("@/routes/asr");
const { useModelPrefs } = await import("@/store/models");
const { ASR_MODELS } = await import("@/audio/types");
const WHISPER = ASR_MODELS[0].id;
const MOONSHINE = ASR_MODELS[1].id;
const AsrPage = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!AsrPage) throw new Error("ASR route component not found");
  render(<AsrPage />);
}

describe("AsrPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockState = { ...baseState };
    cached = new Set();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders the heading and both model options", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /automatic speech recognition/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /whisper base/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /moonshine tiny/i }),
    ).toBeInTheDocument();
  });

  // Both RUN triggers wait for a model; the input sources do not. Getting audio
  // in is not running anything, and gating the upload forced a download before
  // the user was allowed to say what to run it on.
  it("gates both run triggers on the model, and neither input source", () => {
    renderPage();
    expect(
      screen.getByRole("button", { name: /start listening/i }),
    ).toBeDisabled();
    expect(screen.getByRole("button", { name: /^transcribe$/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /upload audio/i })).toBeEnabled();
    expect(
      screen.getByText(/load a model to transcribe/i),
    ).toBeInTheDocument();
  });

  it("downloads nothing on arrival and loads on request", () => {
    renderPage();
    expect(mockLoad).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /load model/i }));
    expect(mockLoad).toHaveBeenCalledOnce();
  });

  it("lets the user abandon a download in flight", () => {
    mockState = { ...baseState, status: "loading", idle: false, loading: true };
    renderPage();
    fireEvent.click(screen.getByTestId("load-cancel"));
    expect(baseState.cancel).toHaveBeenCalledOnce();
  });

  describe("after a page refresh", () => {
    it("comes back on the model the user had selected", () => {
      useModelPrefs.setState({ selected: { asr: MOONSHINE } });
      renderPage();
      expect(
        screen.getByRole("button", { name: /moonshine tiny/i }),
      ).toHaveAttribute("aria-pressed", "true");
    });

    // The refresh restores the *decision*, never the download. A revisit used
    // to resume a cached model on mount, which meant arriving at the page put
    // a model in GPU memory before the user had touched anything.
    it("does not load a cached model, even one loaded before the refresh", async () => {
      cached = new Set([WHISPER]);
      renderPage();

      await waitFor(() =>
        expect(
          screen.getByRole("button", { name: /load model \(cached\)/i }),
        ).toBeEnabled(),
      );
      expect(useLiveAsrArgs).toHaveBeenLastCalledWith(WHISPER);
      expect(mockLoad).not.toHaveBeenCalled();
    });

    it("says the weights are cached, so the one click is an informed one", async () => {
      cached = new Set([WHISPER]);
      renderPage();

      await waitFor(() =>
        expect(
          screen.getByRole("button", { name: /load model \(cached\)/i }),
        ).toBeInTheDocument(),
      );
      // Scoped to LOAD: the picker's own row carries a "cached" badge too, and
      // an unscoped query matches both.
      expect(
        within(screen.getByTestId("slot-2")).getByText(/already downloaded/i),
      ).toBeInTheDocument();
    });

    it("still asks before downloading an uncached model", async () => {
      renderPage();

      await waitFor(() =>
        expect(
          screen.getByRole("button", { name: /load model/i }),
        ).toBeEnabled(),
      );
      expect(mockLoad).not.toHaveBeenCalled();
    });
  });

  it("offers a retry when the load failed", () => {
    mockState = { ...baseState, status: "error", idle: false, error: "boom" };
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /^retry$/i }));
    expect(mockRetry).toHaveBeenCalledOnce();
  });

  it("renders the four slots with an empty transcript before any run", () => {
    renderPage();
    expect(screen.getAllByRole("region")).toHaveLength(4);
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("enables controls and shows the backend once ready", () => {
    mockState = {
      ...baseState,
      status: "ready",
      idle: false,
      loading: false,
      ready: true,
      backend: "webgpu",
    };
    renderPage();
    expect(
      screen.getByRole("button", { name: /start listening/i }),
    ).toBeEnabled();
    expect(screen.getByText(/running on/i)).toBeInTheDocument();
    // The intro copy also mentions "WebGPU"; the status badge holds the exact
    // lowercase backend value.
    expect(screen.getByText("webgpu")).toBeInTheDocument();
  });

  it("shows a Stop button and a live indicator while recording", () => {
    mockState = {
      ...baseState,
      status: "ready",
      idle: false,
      loading: false,
      ready: true,
      backend: "wasm",
      recording: true,
      text: "hello",
    };
    renderPage();
    expect(screen.getByRole("button", { name: /stop/i })).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /start listening/i }),
    ).not.toBeInTheDocument();
    expect(screen.getByText(/listening…/i)).toBeInTheDocument();
    expect(screen.getByText("hello")).toBeInTheDocument();
  });

  it("renders the final transcript text when present and idle", () => {
    mockState = {
      ...baseState,
      status: "ready",
      idle: false,
      loading: false,
      ready: true,
      backend: "wasm",
      text: "the quick brown fox",
    };
    renderPage();
    expect(screen.getByText("the quick brown fox")).toBeInTheDocument();
    expect(screen.getByText(/final transcript/i)).toBeInTheDocument();
  });

  it("hides the audio card until there is something to show", () => {
    mockState = { ...baseState, status: "ready", idle: false, ready: true };
    renderPage();
    expect(screen.queryByText(/^audio$/i)).not.toBeInTheDocument();
    expect(
      screen.queryByRole("img", { name: /waveform/i }),
    ).not.toBeInTheDocument();
  });

  it("shows the live waveform while recording", () => {
    mockState = {
      ...baseState,
      status: "ready",
      idle: false,
      loading: false,
      ready: true,
      recording: true,
      stream: {} as MediaStream,
    };
    renderPage();
    expect(
      screen.getByRole("img", { name: /live microphone waveform/i }),
    ).toBeInTheDocument();
    expect(screen.getByText(/live microphone signal/i)).toBeInTheDocument();
  });

  it("shows the retained take with play / download, and no run control", () => {
    const clip = new Float32Array(16000 * 2); // 2 s of silence
    mockState = {
      ...baseState,
      status: "ready",
      idle: false,
      loading: false,
      ready: true,
      clip,
      text: "hello",
    };
    renderPage();
    expect(
      screen.getByRole("img", { name: /audio waveform/i }),
    ).toBeInTheDocument();
    expect(screen.getByText(/0:02\.0/)).toBeInTheDocument(); // duration chip
    expect(screen.getByRole("button", { name: /play/i })).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /download wav/i }),
    ).toBeInTheDocument();

    // The take's own "Transcribe clip" button is gone: a page with two RUN
    // triggers has no single answer to "what runs the model?". The take feeds
    // the transport's one Transcribe button instead.
    fireEvent.click(screen.getByRole("button", { name: /^transcribe$/i }));
    expect(mockTranscribeClip).toHaveBeenCalledWith(clip);
  });

  it("toggles play → pause → resume and stops playback", () => {
    mockState = {
      ...baseState,
      status: "ready",
      idle: false,
      loading: false,
      ready: true,
      clip: new Float32Array(16000),
    };
    renderPage();

    // Play — starts a fresh context; Pause + Stop appear.
    fireEvent.click(screen.getByRole("button", { name: /^play$/i }));
    expect(playMock).toHaveBeenCalledTimes(1);
    const pauseBtn = screen.getByRole("button", { name: /^pause$/i });
    expect(screen.getByRole("button", { name: /^stop$/i })).toBeInTheDocument();

    // Pause → suspend; the toggle now offers Resume.
    fireEvent.click(pauseBtn);
    expect(playCtx.suspend).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: /^resume$/i }));
    expect(playCtx.resume).toHaveBeenCalledTimes(1);

    // Stop → closes the context and returns to the idle Play state.
    fireEvent.click(screen.getByRole("button", { name: /^stop$/i }));
    expect(playCtx.close).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("button", { name: /^play$/i })).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /^stop$/i }),
    ).not.toBeInTheDocument();
  });

  it("shows a spinner and blocks re-runs while a transcription is in flight", () => {
    mockState = {
      ...baseState,
      status: "ready",
      idle: false,
      loading: false,
      ready: true,
      running: true,
      clip: new Float32Array(16000),
    };
    renderPage();
    expect(
      screen.getByRole("button", { name: /transcribing…/i }),
    ).toBeDisabled();
  });

  // Picking a clip needs no model, so the clips are live from the start. The
  // 60 s TED sample is why: gating them forced a download first, and running
  // them on click charged an inference for a browse.
  it("offers sample clips, choosable before a model is loaded", () => {
    renderPage();
    expect(screen.getByText(/test with a sample clip/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^jfk$/i })).toBeEnabled();
    expect(
      screen.getByRole("button", { name: /^transcribe$/i }),
    ).toBeDisabled();
  });

  it("loads a sample clip and its reference, and transcribes only on request", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        arrayBuffer: () => Promise.resolve(new ArrayBuffer(8)),
      }),
    );
    mockState = { ...baseState, status: "ready", idle: false, ready: true };
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: /^jfk$/i }));

    // The reference transcript is on screen so the output can be compared to
    // it — before a model has been spent, not after.
    await waitFor(() =>
      expect(
        screen.getByText(/ask not what your country/i),
      ).toBeInTheDocument(),
    );
    expect(fetch).toHaveBeenCalledWith(expect.stringMatching(/jfk\.wav$/));
    expect(mockTranscribeClip).not.toHaveBeenCalled();

    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /^transcribe$/i }),
      ).toBeEnabled(),
    );
    fireEvent.click(screen.getByRole("button", { name: /^transcribe$/i }));
    await waitFor(() => expect(mockTranscribeClip).toHaveBeenCalledTimes(1));
    expect(mockTranscribeClip.mock.calls[0][0]).toBeInstanceOf(Float32Array);
  });

  it("surfaces a failed sample fetch as an error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 404 }),
    );
    mockState = { ...baseState, status: "ready", idle: false, ready: true };
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: /^jfk$/i }));

    await waitFor(() =>
      expect(screen.getByText(/couldn't fetch jfk/i)).toBeInTheDocument(),
    );
    expect(mockTranscribeClip).not.toHaveBeenCalled();
  });

  it("surfaces a load error from the hook", () => {
    mockState = {
      ...baseState,
      status: "error",
      idle: false,
      error: "download failed",
    };
    renderPage();
    expect(screen.getByText(/download failed/i)).toBeInTheDocument();
  });

  it("renders a timestamped transcript when the model returns segments", () => {
    mockState = {
      ...baseState,
      status: "ready",
      idle: false,
      loading: false,
      ready: true,
      backend: "wasm",
      text: "And so my fellow Americans ask not",
      chunks: [
        { text: " And so my fellow Americans", timestamp: [0, 4] },
        { text: " ask not", timestamp: [65, 68] },
      ],
    };
    renderPage();

    expect(screen.getByText(/And so my fellow Americans/)).toBeInTheDocument();
    expect(screen.getByText(/ask not/)).toBeInTheDocument();
    // m:ss, take-relative — 65 s is 1:05, not 0:65.
    expect(screen.getByText("0:00")).toBeInTheDocument();
    expect(screen.getByText("1:05")).toBeInTheDocument();
  });

  it("falls back to plain text when the model returns no segments", () => {
    mockState = {
      ...baseState,
      status: "ready",
      idle: false,
      loading: false,
      ready: true,
      backend: "wasm",
      text: "no segmentation here",
      chunks: [],
    };
    renderPage();

    expect(screen.getByText("no segmentation here")).toBeInTheDocument();
    expect(screen.queryByText(/^\d+:\d\d$/)).not.toBeInTheDocument();
  });
});
