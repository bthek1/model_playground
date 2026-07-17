import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { TtsAudio } from "@/audio/tts";
import type { UseTtsResult } from "@/hooks/useTts";

// Mock playback/encoding — no Web Audio in the test env.
const play = vi.fn(() => ({ close: vi.fn() }));
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
  status: "loading",
  loading: true,
  ready: false,
  progress: null,
  backend: null,
  result: null,
  running: false,
  error: null,
  synthesize: mockSynthesize,
};
let mockState: UseTtsResult = { ...baseState };

vi.mock("@/hooks/useTts", () => ({
  useTts: () => mockState,
}));

const { Route } = await import("@/routes/text-to-speech");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Text-to-speech route component not found");
  render(<Page />);
}

describe("TextToSpeechPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockState = { ...baseState };
  });

  it("renders the heading and every model option", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /text to speech/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /kokoro 82m/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /mms english/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /speecht5/i })).toBeInTheDocument();
  });

  it("disables Speak until the model is ready", () => {
    renderPage();
    expect(screen.getByRole("button", { name: /speak/i })).toBeDisabled();
    expect(screen.getByText(/loading model/i)).toBeInTheDocument();
  });

  it("shows the voice picker for Kokoro and hides it for pipeline models", () => {
    mockState = { ...baseState, status: "ready", loading: false, ready: true };
    renderPage();
    // Kokoro is the default model → voice picker present.
    expect(screen.getByLabelText(/voice/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /mms english/i }));
    expect(screen.queryByLabelText(/voice/i)).not.toBeInTheDocument();
  });

  it("synthesises the text with the selected voice and plays it", async () => {
    mockState = { ...baseState, status: "ready", loading: false, ready: true, backend: "wasm" };
    const audio = new Float32Array([0.1, 0.2]);
    mockSynthesize.mockResolvedValue({ audio, sampleRate: 24000 });
    renderPage();

    fireEvent.change(screen.getByLabelText(/text/i), {
      target: { value: "hello world" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^speak$/i }));

    await waitFor(() => expect(mockSynthesize).toHaveBeenCalledTimes(1));
    expect(mockSynthesize).toHaveBeenCalledWith("hello world", { voice: "af_heart" });
    await waitFor(() => expect(play).toHaveBeenCalledWith(audio, 24000));
  });

  it("shows the result card with duration, play, and download", () => {
    mockState = {
      ...baseState,
      status: "ready",
      loading: false,
      ready: true,
      result: { audio: new Float32Array(48000), sampleRate: 24000 }, // 2 s
    };
    renderPage();
    expect(screen.getByText(/2\.0s · 24 kHz/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^play$/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /download wav/i })).toBeInTheDocument();
  });

  it("surfaces a load error from the hook", () => {
    mockState = { ...baseState, status: "error", loading: false, error: "download failed" };
    renderPage();
    expect(screen.getByText(/download failed/i)).toBeInTheDocument();
  });
});
