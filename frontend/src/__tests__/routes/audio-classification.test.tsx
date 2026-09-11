import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { UseAudioClassifierResult } from "@/hooks/useAudioClassifier";

// Mock the mic/decode helpers so the record button resolves without real
// AudioContext/getUserMedia (absent in the test env).
const recordMic = vi.fn().mockResolvedValue(new Float32Array([0.1]));
const decodeToMono = vi.fn().mockResolvedValue(new Float32Array([0.2]));
vi.mock("@/audio/io", () => ({
  recordMic: (...args: unknown[]) => recordMic(...args),
  decodeToMono: (...args: unknown[]) => decodeToMono(...args),
}));

// happy-dom has no canvas 2D context; Waveform renders its own guarded
// fallback and is covered by its own tests.
vi.mock("@/components/audio/Waveform", () => ({
  Waveform: () => <div data-testid="waveform" />,
  LiveWaveform: () => <div data-testid="live-waveform" />,
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

const mockClassify = vi.fn();
const baseState: UseAudioClassifierResult = {
  status: "idle",
  idle: true,
  loading: false,
  ready: false,
  progress: null,
  loadProgress: null,
  loadedInMs: null,
  backend: null,
  running: false,
  error: null,
  run: vi.fn(),
  isZeroShot: false,
  result: null,
  classify: mockClassify,
  // Machine A actions — additive in the useModelWorker refactor.
  load: vi.fn(),
  retry: vi.fn(),
  cancel: vi.fn(),
};
let mockState: UseAudioClassifierResult = { ...baseState };

vi.mock("@/hooks/useAudioClassifier", () => ({
  useAudioClassifier: () => mockState,
}));

const { Route } = await import("@/routes/audio-classification");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Audio classification route component not found");
  render(<Page />);
}

describe("AudioClassificationPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockState = { ...baseState };
  });

  it("renders the heading and every model option", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /audio classification/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /ast \(audioset\)/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /keyword spotting/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /clap \(zero-shot\)/i })).toBeInTheDocument();
  });

  // Getting audio in and running a model on it are different things, so the
  // sources are *not* gated on a loaded model — picking a clip first is a
  // sensible order to work in. Only the Classify button is gated.
  it("lets a clip be chosen before a model exists, and gates only Classify", () => {
    renderPage();
    expect(screen.getByRole("button", { name: /record 5s/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /upload audio/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /^classify$/i })).toBeDisabled();
    expect(screen.getByText(/load a model to classify/i)).toBeInTheDocument();
  });

  it("keeps Classify disabled until a clip is held, however ready the model", () => {
    mockState = { ...baseState, status: "ready", idle: false, ready: true };
    renderPage();
    expect(screen.getByRole("button", { name: /^classify$/i })).toBeDisabled();
    expect(screen.getByTestId("audio-input-empty")).toBeInTheDocument();
  });

  it("downloads nothing on arrival and loads on request", () => {
    renderPage();
    expect(baseState.load).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /load model/i }));
    expect(baseState.load).toHaveBeenCalledOnce();
  });

  it("renders the four slots with an empty OUTPUT before any run", () => {
    renderPage();
    expect(screen.getAllByRole("region")).toHaveLength(4);
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("hides the labels textarea for a fixed-label model", () => {
    mockState = { ...baseState, status: "ready", idle: false, ready: true, isZeroShot: false };
    renderPage();
    expect(screen.queryByLabelText(/labels to score against/i)).not.toBeInTheDocument();
  });

  it("shows the labels textarea for a zero-shot model", () => {
    mockState = { ...baseState, status: "ready", idle: false, ready: true, isZeroShot: true };
    renderPage();
    expect(screen.getByLabelText(/labels to score against/i)).toBeInTheDocument();
  });

  it("renders ranked predictions with percentages", () => {
    mockState = {
      ...baseState,
      status: "ready",
      idle: false,
      ready: true,
      backend: "wasm",
      result: [
        { label: "Speech", score: 0.82 },
        { label: "Music", score: 0.11 },
      ],
    };
    renderPage();
    expect(screen.getByText("Speech")).toBeInTheDocument();
    expect(screen.getByText("82%")).toBeInTheDocument();
    expect(screen.getByText("Music")).toBeInTheDocument();
    expect(screen.getByText("11%")).toBeInTheDocument();
  });

  it("surfaces an error from the hook", () => {
    mockState = { ...baseState, status: "ready", idle: false, ready: true, error: "add at least one label" };
    renderPage();
    expect(screen.getByText(/add at least one label/i)).toBeInTheDocument();
  });

  // Recording captures the clip and stops. It used to classify it too, which
  // meant editing CLAP's prompts required recording all over again.
  it("records into the input without classifying, then classifies on request", async () => {
    mockState = { ...baseState, status: "ready", idle: false, ready: true, isZeroShot: true };
    renderPage();

    fireEvent.change(screen.getByLabelText(/labels to score against/i), {
      target: { value: " cat \ndog, bird\n\n" },
    });
    fireEvent.click(screen.getByRole("button", { name: /record 5s/i }));

    await waitFor(() => expect(recordMic).toHaveBeenCalledWith(5, 16_000));
    await waitFor(() =>
      expect(screen.getByTestId("waveform")).toBeInTheDocument(),
    );
    expect(mockClassify).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /^classify$/i }));
    await waitFor(() => expect(mockClassify).toHaveBeenCalledTimes(1));
    const [audio, labels] = mockClassify.mock.calls[0];
    expect(audio).toBeInstanceOf(Float32Array);
    expect(labels).toEqual(["cat", "dog", "bird"]);
  });

  // The point of holding the clip rather than consuming it: the prompts are
  // the interesting variable on a zero-shot page, and re-scoring must not cost
  // a second recording.
  it("re-scores the same clip after the prompts change", async () => {
    mockState = { ...baseState, status: "ready", idle: false, ready: true, isZeroShot: true };
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: /record 5s/i }));
    await waitFor(() =>
      expect(screen.getByTestId("waveform")).toBeInTheDocument(),
    );

    fireEvent.click(screen.getByRole("button", { name: /^classify$/i }));
    await waitFor(() => expect(mockClassify).toHaveBeenCalledTimes(1));

    fireEvent.change(screen.getByLabelText(/labels to score against/i), {
      target: { value: "rain\nthunder" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^classify$/i }));

    await waitFor(() => expect(mockClassify).toHaveBeenCalledTimes(2));
    expect(mockClassify.mock.calls[1][1]).toEqual(["rain", "thunder"]);
    expect(recordMic).toHaveBeenCalledTimes(1); // the mic was used once
  });
});
