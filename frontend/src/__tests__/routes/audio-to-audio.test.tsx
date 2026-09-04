import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { EnhanceResult } from "@/audio/enhance/types";
import type { ModelTask } from "@/model/types";

// No Web Audio in the test env: decoding, playback and WAV encoding are mocked.
const play = vi.fn<(...args: unknown[]) => { close: () => void }>(() => ({
  close: vi.fn(),
}));
const decodeToMono = vi.fn(async () => new Float32Array([0.1, 0.2, 0.3]));
const recordMic = vi.fn(async () => new Float32Array([0.4, 0.5]));
vi.mock("@/audio/io", () => ({
  play: (...args: unknown[]) => play(...args),
  decodeToMono: (...args: unknown[]) => decodeToMono(...(args as [])),
  recordMic: (...args: unknown[]) => recordMic(...(args as [])),
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

const mockRun = vi.fn<(audio: Float32Array) => Promise<EnhanceResult>>();
const mockLoad = vi.fn();
const mockRetry = vi.fn();

type EnhanceTask = ModelTask<Float32Array, EnhanceResult>;

const baseState: EnhanceTask = {
  status: "idle",
  idle: true,
  loading: false,
  ready: false,
  progress: null,
  loadProgress: null,
  loadedInMs: null,
  backend: null,
  load: mockLoad,
  retry: mockRetry,
  cancel: vi.fn(),
  run: mockRun,
  running: false,
  result: null,
  error: null,
};
let mockState: EnhanceTask = { ...baseState };

vi.mock("@/hooks/useEnhance", () => ({
  useEnhance: () => mockState,
}));

const { Route } = await import("@/routes/audio-to-audio");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Audio-to-audio route component not found");
  render(<Page />);
}

const readyState: EnhanceTask = {
  ...baseState,
  status: "ready",
  idle: false,
  ready: true,
  backend: "wasm",
};

describe("AudioToAudioPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockState = { ...baseState };
  });

  it("renders the heading and the single model option", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /audio to audio/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /deepfilternet3/i }),
    ).toBeInTheDocument();
  });

  it("states the download cost and offers Load before anything is fetched", () => {
    renderPage();
    expect(screen.getByTestId("model-size-note")).toHaveTextContent(/MB on WebGPU/);
    // 8 MB is nowhere near LARGE_MODEL_BYTES, so the guardrail must stay quiet.
    expect(screen.queryByTestId("model-size-warning")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /load model/i }));
    expect(mockLoad).toHaveBeenCalledTimes(1);
  });

  it("keeps the transport disabled until the model is ready", () => {
    renderPage();
    expect(screen.getByRole("button", { name: /record/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /upload audio/i })).toBeDisabled();
  });

  it("decodes and records at 48 kHz — not the 16 kHz every other route uses", async () => {
    mockState = readyState;
    mockRun.mockResolvedValue({
      audio: new Float32Array([0.9]),
      sampleRate: 48000,
    });
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: /record/i }));
    await waitFor(() => expect(recordMic).toHaveBeenCalled());
    expect(recordMic).toHaveBeenCalledWith(expect.any(Number), 48000);
  });

  it("enhances an uploaded file and keeps a copy of the input for playback", async () => {
    mockState = readyState;
    const enhanced = new Float32Array([0.9, 0.8]);
    mockRun.mockResolvedValue({ audio: enhanced, sampleRate: 48000 });
    renderPage();

    const file = new File(["x"], "noisy.wav", { type: "audio/wav" });
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(input, { target: { files: [file] } });

    await waitFor(() => expect(decodeToMono).toHaveBeenCalled());
    expect(decodeToMono).toHaveBeenCalledWith(expect.anything(), 48000);
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));

    // `run` transfers (and detaches) its argument, so the route must have kept
    // its own copy — otherwise the "before" row would render an empty buffer.
    const passed = mockRun.mock.calls[0][0];
    expect(passed).not.toBe(decodeToMono.mock.results[0].value);
    await waitFor(() =>
      expect(screen.getByText(/noisy input/i)).toBeInTheDocument(),
    );
  });

  it("shows both rows once a result arrives, and plays each one", async () => {
    mockState = {
      ...readyState,
      result: { audio: new Float32Array([0.5, 0.6]), sampleRate: 48000 },
    };
    mockRun.mockResolvedValue(mockState.result!);
    renderPage();

    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(input, {
      target: { files: [new File(["x"], "a.wav", { type: "audio/wav" })] },
    });
    await waitFor(() => expect(screen.getByText(/enhanced/i)).toBeInTheDocument());

    const plays = screen.getAllByRole("button", { name: /^play$/i });
    expect(plays).toHaveLength(2);
    fireEvent.click(plays[1]);
    await waitFor(() =>
      expect(play).toHaveBeenCalledWith(mockState.result!.audio, 48000),
    );
    expect(screen.getByRole("button", { name: /wav/i })).toBeInTheDocument();
  });

  it("surfaces a load error and offers a retry", () => {
    mockState = {
      ...baseState,
      status: "error",
      idle: false,
      error: "404 on deepfilter.onnx",
    };
    renderPage();
    expect(screen.getByText(/404 on deepfilter\.onnx/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /retry/i }));
    expect(mockRetry).toHaveBeenCalledTimes(1);
  });

  it("shows a decode failure without blaming the model", async () => {
    mockState = readyState;
    decodeToMono.mockRejectedValueOnce(new Error("Unsupported audio format"));
    renderPage();

    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(input, {
      target: { files: [new File(["x"], "a.txt", { type: "text/plain" })] },
    });

    await waitFor(() =>
      expect(screen.getByText(/unsupported audio format/i)).toBeInTheDocument(),
    );
    expect(mockRun).not.toHaveBeenCalled();
  });
});
