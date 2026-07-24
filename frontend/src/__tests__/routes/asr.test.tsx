import { fireEvent, render, screen, waitFor } from "@testing-library/react";
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
const playMock = vi.fn((..._args: unknown[]) => playCtx);
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
        (path: string) => (opts: Record<string, unknown>) => ({ path, options: opts }),
      ),
  };
});

const mockStart = vi.fn();
const mockStop = vi.fn();
const mockTranscribeClip = vi.fn();
const baseState: UseLiveAsrResult = {
  status: "loading",
  loading: true,
  ready: false,
  progress: null,
  backend: null,
  recording: false,
  running: false,
  stream: null,
  clip: null,
  sampleRate: 16000,
  text: "",
  error: null,
  start: mockStart,
  stop: mockStop,
  transcribeClip: mockTranscribeClip,
};
let mockState: UseLiveAsrResult = { ...baseState };

vi.mock("@/hooks/useLiveAsr", () => ({
  useLiveAsr: () => mockState,
}));

const { Route } = await import("@/routes/asr");
const AsrPage = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!AsrPage) throw new Error("ASR route component not found");
  render(<AsrPage />);
}

describe("AsrPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockState = { ...baseState };
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders the heading and both model options", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /automatic speech recognition/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /whisper base/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /moonshine tiny/i })).toBeInTheDocument();
  });

  it("disables the listen/upload controls until the model is ready", () => {
    renderPage();
    expect(screen.getByRole("button", { name: /start listening/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /upload audio/i })).toBeDisabled();
    expect(screen.getByText(/loading model/i)).toBeInTheDocument();
  });

  it("enables controls and shows the backend once ready", () => {
    mockState = {
      ...baseState,
      status: "ready",
      loading: false,
      ready: true,
      backend: "webgpu",
    };
    renderPage();
    expect(screen.getByRole("button", { name: /start listening/i })).toBeEnabled();
    expect(screen.getByText(/running on/i)).toBeInTheDocument();
    // The intro copy also mentions "WebGPU"; the status badge holds the exact
    // lowercase backend value.
    expect(screen.getByText("webgpu")).toBeInTheDocument();
  });

  it("shows a Stop button and a live indicator while recording", () => {
    mockState = {
      ...baseState,
      status: "ready",
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
    mockState = { ...baseState, status: "ready", loading: false, ready: true };
    renderPage();
    expect(screen.queryByText(/^audio$/i)).not.toBeInTheDocument();
    expect(screen.queryByRole("img", { name: /waveform/i })).not.toBeInTheDocument();
  });

  it("shows the live waveform while recording", () => {
    mockState = {
      ...baseState,
      status: "ready",
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

  it("shows the retained take with play / download / re-transcribe actions", () => {
    const clip = new Float32Array(16000 * 2); // 2 s of silence
    mockState = {
      ...baseState,
      status: "ready",
      loading: false,
      ready: true,
      clip,
      text: "hello",
    };
    renderPage();
    expect(screen.getByRole("img", { name: /audio waveform/i })).toBeInTheDocument();
    expect(screen.getByText(/0:02\.0/)).toBeInTheDocument(); // duration chip
    expect(screen.getByRole("button", { name: /play/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /download wav/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /transcribe clip/i }));
    expect(mockTranscribeClip).toHaveBeenCalledWith(clip);
  });

  it("toggles play → pause → resume and stops playback", () => {
    mockState = {
      ...baseState,
      status: "ready",
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
    expect(screen.queryByRole("button", { name: /^stop$/i })).not.toBeInTheDocument();
  });

  it("shows a spinner and blocks re-runs while a transcription is in flight", () => {
    mockState = {
      ...baseState,
      status: "ready",
      loading: false,
      ready: true,
      running: true,
      clip: new Float32Array(16000),
    };
    renderPage();
    expect(screen.getByRole("button", { name: /transcribing…/i })).toBeDisabled();
  });

  it("offers sample clips and disables them until the model is ready", () => {
    renderPage();
    expect(screen.getByText(/test with a sample clip/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^jfk$/i })).toBeDisabled();
  });

  it("runs a sample clip through the model and shows its reference", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        arrayBuffer: () => Promise.resolve(new ArrayBuffer(8)),
      }),
    );
    mockState = { ...baseState, status: "ready", loading: false, ready: true };
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: /^jfk$/i }));

    await waitFor(() => expect(mockTranscribeClip).toHaveBeenCalledTimes(1));
    expect(fetch).toHaveBeenCalledWith(expect.stringMatching(/jfk\.wav$/));
    expect(mockTranscribeClip.mock.calls[0][0]).toBeInstanceOf(Float32Array);
    // The reference transcript is shown so the output can be compared to it.
    expect(screen.getByText(/ask not what your country/i)).toBeInTheDocument();
  });

  it("surfaces a failed sample fetch as an error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 404 }),
    );
    mockState = { ...baseState, status: "ready", loading: false, ready: true };
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: /^jfk$/i }));

    await waitFor(() =>
      expect(screen.getByText(/couldn't fetch jfk/i)).toBeInTheDocument(),
    );
    expect(mockTranscribeClip).not.toHaveBeenCalled();
  });

  it("surfaces a load error from the hook", () => {
    mockState = { ...baseState, status: "error", loading: false, error: "download failed" };
    renderPage();
    expect(screen.getByText(/download failed/i)).toBeInTheDocument();
  });
});
