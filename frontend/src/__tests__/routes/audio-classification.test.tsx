import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { UseAudioClassifierResult } from "@/hooks/useAudioClassifier";

// Mock the mic/decode helpers so the record button resolves without real
// AudioContext/getUserMedia (absent in the test env).
const recordMic = vi.fn().mockResolvedValue(new Float32Array([0.1]));
vi.mock("@/audio/io", () => ({
  recordMic: (...args: unknown[]) => recordMic(...args),
  decodeToMono: vi.fn(),
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
  status: "loading",
  loading: true,
  ready: false,
  progress: null,
  backend: null,
  running: false,
  error: null,
  run: vi.fn(),
  isZeroShot: false,
  result: null,
  classify: mockClassify,
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

  it("disables the record/upload controls until the model is ready", () => {
    renderPage();
    expect(screen.getByRole("button", { name: /record 5s/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /upload audio/i })).toBeDisabled();
    expect(screen.getByText(/loading model/i)).toBeInTheDocument();
  });

  it("hides the labels textarea for a fixed-label model", () => {
    mockState = { ...baseState, loading: false, ready: true, isZeroShot: false };
    renderPage();
    expect(screen.queryByLabelText(/labels to score against/i)).not.toBeInTheDocument();
  });

  it("shows the labels textarea for a zero-shot model", () => {
    mockState = { ...baseState, loading: false, ready: true, isZeroShot: true };
    renderPage();
    expect(screen.getByLabelText(/labels to score against/i)).toBeInTheDocument();
  });

  it("renders ranked predictions with percentages", () => {
    mockState = {
      ...baseState,
      loading: false,
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
    mockState = { ...baseState, loading: false, ready: true, error: "add at least one label" };
    renderPage();
    expect(screen.getByText(/add at least one label/i)).toBeInTheDocument();
  });

  it("parses newline/comma-separated prompts and passes them to classify on record", async () => {
    mockState = { ...baseState, status: "ready", loading: false, ready: true, isZeroShot: true };
    renderPage();

    fireEvent.change(screen.getByLabelText(/labels to score against/i), {
      target: { value: " cat \ndog, bird\n\n" },
    });
    fireEvent.click(screen.getByRole("button", { name: /record 5s/i }));

    await waitFor(() => expect(mockClassify).toHaveBeenCalledTimes(1));
    expect(recordMic).toHaveBeenCalledWith(5);
    const [audio, labels] = mockClassify.mock.calls[0];
    expect(audio).toBeInstanceOf(Float32Array);
    expect(labels).toEqual(["cat", "dog", "bird"]);
  });
});
