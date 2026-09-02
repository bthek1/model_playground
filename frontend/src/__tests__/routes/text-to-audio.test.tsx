import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { TtsAudio } from "@/audio/tts";
import { TOKENS_PER_SECOND, DEFAULT_SECONDS } from "@/audio/textToAudio";
import type { UseTtsResult } from "@/hooks/useTts";

// Mock playback/encoding — no Web Audio in the test env.
const play = vi.fn<(...args: unknown[]) => { close: () => void }>(() => ({
  close: vi.fn(),
}));
vi.mock("@/audio/io", () => ({
  play: (...args: unknown[]) => play(...args),
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

const mockSynthesize = vi.fn<(text: string, opts?: unknown) => Promise<TtsAudio>>();
const baseState: UseTtsResult = {
  status: "ready",
  loading: false,
  ready: true,
  progress: null,
  backend: "wasm",
  result: null,
  running: false,
  error: null,
  synthesize: mockSynthesize,
};
let mockState: UseTtsResult = { ...baseState };

// The hook is what triggers the 571 MB download, so the spy also proves the
// gate works: it must not be called until the user opts in.
const useTts = vi.fn(() => mockState);
vi.mock("@/hooks/useTts", () => ({ useTts: () => useTts() }));

const { Route } = await import("@/routes/text-to-audio");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Text-to-audio route component not found");
  render(<Page />);
}

function enable() {
  fireEvent.click(screen.getByRole("button", { name: /Download the model/i }));
}

describe("TextToAudioPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockState = { ...baseState };
  });

  it("gates the model behind an explicit opt-in", () => {
    renderPage();

    expect(
      screen.getByRole("heading", { name: /Text to Audio/i }),
    ).toBeInTheDocument();
    expect(screen.getByText(/Experimental — and slow/i)).toBeInTheDocument();
    // Nothing may load before the user accepts the cost.
    expect(useTts).not.toHaveBeenCalled();
    expect(
      screen.queryByRole("button", { name: /^Generate/ }),
    ).not.toBeInTheDocument();
  });

  it("states the download size and the speed caveat before downloading", () => {
    renderPage();

    // The two things that would surprise a user, both up front.
    expect(screen.getByText(/on WebGPU · .* on WASM/)).toBeInTheDocument();
    expect(screen.getByText(/autoregressive/i)).toBeInTheDocument();
  });

  it("mounts the generator only after opting in", () => {
    renderPage();
    enable();

    expect(useTts).toHaveBeenCalled();
    expect(screen.getByRole("button", { name: /^Generate/ })).toBeEnabled();
    expect(screen.getByLabelText(/Prompt/i)).toBeInTheDocument();
  });

  it("generates from the prompt and plays the result", async () => {
    const audio = { audio: new Float32Array([0.1, 0.2]), sampleRate: 32000 };
    mockSynthesize.mockResolvedValue(audio);
    renderPage();
    enable();

    fireEvent.change(screen.getByLabelText(/Prompt/i), {
      target: { value: "8-bit chiptune" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^Generate/ }));

    await waitFor(() => expect(mockSynthesize).toHaveBeenCalled());
    // Duration is converted to a token budget for the worker.
    expect(mockSynthesize).toHaveBeenCalledWith("8-bit chiptune", {
      maxNewTokens: DEFAULT_SECONDS * TOKENS_PER_SECOND,
    });
    await waitFor(() =>
      expect(play).toHaveBeenCalledWith(audio.audio, audio.sampleRate),
    );
  });

  it("sends a bigger token budget for a longer clip", async () => {
    mockSynthesize.mockResolvedValue({
      audio: new Float32Array([0]),
      sampleRate: 32000,
    });
    renderPage();
    enable();

    fireEvent.change(screen.getByLabelText(/Length/i), {
      target: { value: "10" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^Generate/ }));

    await waitFor(() =>
      expect(mockSynthesize).toHaveBeenCalledWith(expect.any(String), {
        maxNewTokens: 10 * TOKENS_PER_SECOND,
      }),
    );
  });

  it("disables Generate while a clip is being generated", () => {
    mockState = { ...baseState, running: true };
    renderPage();
    enable();

    expect(screen.getByRole("button", { name: /Generating/i })).toBeDisabled();
  });

  it("surfaces a load or generation error", () => {
    mockState = { ...baseState, status: "error", ready: false, error: "boom" };
    renderPage();
    enable();

    expect(screen.getByText("boom")).toBeInTheDocument();
  });

  it("offers replay and WAV download once a clip exists", () => {
    mockState = {
      ...baseState,
      result: { audio: new Float32Array([0.1]), sampleRate: 32000 },
    };
    renderPage();
    enable();

    expect(screen.getByText(/Generated audio/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Play/ })).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Download WAV/ }),
    ).toBeInTheDocument();
  });
});
