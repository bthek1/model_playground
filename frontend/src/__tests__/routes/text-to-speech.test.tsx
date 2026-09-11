import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { TtsAudio } from "@/audio/tts";
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
  status: "idle",
  idle: true,
  loading: false,
  ready: false,
  progress: null,
  loadProgress: null,
  loadedInMs: null,
  backend: null,
  result: null,
  running: false,
  error: null,
  synthesize: mockSynthesize,
  // Machine A actions — additive in the useModelWorker refactor.
  load: vi.fn(),
  retry: vi.fn(),
  cancel: vi.fn(),
};
let mockState: UseTtsResult = { ...baseState };

// What the page asks the worker layer for — the second argument is the resolved
// auto-load decision, which is the whole refresh story (see useModelSelection).
// Forwarded verbatim, arity included: the route passes the model and *nothing
// else*, and "there is no second argument" is the guarantee.
const useTtsArgs = vi.fn<(...args: unknown[]) => void>();
vi.mock("@/hooks/useTts", () => ({
  useTts: (...args: unknown[]) => {
    useTtsArgs(...args);
    return mockState;
  },
}));

// The browser cache probe. Empty unless a test says otherwise.
let cached = new Set<string>();
vi.mock("@/model/cache", () => ({
  cachedModels: () => Promise.resolve(cached),
  evictModel: vi.fn(() => Promise.resolve()),
}));

const { Route } = await import("@/routes/text-to-speech");
const { useModelPrefs } = await import("@/store/models");
const { TTS_MODELS } = await import("@/audio/tts");
const KOKORO = TTS_MODELS[0].id;
const MMS = TTS_MODELS[1].id;
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Text-to-speech route component not found");
  render(<Page />);
}

describe("TextToSpeechPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockState = { ...baseState };
    cached = new Set();
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
    expect(screen.getByRole("button", { name: /^speak$/i })).toBeDisabled();
    expect(screen.getByText(/load a model to synthesise/i)).toBeInTheDocument();
  });

  it("downloads nothing on arrival — the LOAD slot offers the action instead", () => {
    renderPage();
    expect(baseState.load).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: /load model/i })).toBeEnabled();
  });

  it("starts the download only when the user asks", () => {
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /load model/i }));
    expect(baseState.load).toHaveBeenCalledOnce();
  });

  it("shows aggregate download progress in the LOAD slot", () => {
    mockState = {
      ...baseState,
      status: "loading",
      idle: false,
      loading: true,
      loadProgress: {
        phase: "downloading",
        percent: 30,
        loaded: 30 * 1024 * 1024,
        total: 100 * 1024 * 1024,
        files: { done: 1, count: 4 },
        current: "model.onnx",
        elapsedMs: 5000,
      },
    };
    renderPage();
    expect(screen.getByText(/model\.onnx/)).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).toHaveAttribute(
      "aria-valuenow",
      "30",
    );
    expect(screen.getByText("30 MB / 100 MB")).toBeInTheDocument();
  });

  it("lets the user abandon a download in flight", () => {
    mockState = { ...baseState, status: "loading", idle: false, loading: true };
    renderPage();
    fireEvent.click(screen.getByTestId("load-cancel"));
    expect(baseState.cancel).toHaveBeenCalledOnce();
  });

  describe("surviving a page refresh", () => {
    it("restores the model the user had selected", () => {
      useModelPrefs.setState({ selected: { tts: MMS } });
      renderPage();
      // The picker starts where the user left it, not on the default.
      expect(screen.getByRole("button", { name: /mms english/i })).toHaveAttribute(
        "aria-pressed",
        "true",
      );
    });

    // A cached model is one click, not zero. The page used to resume it on
    // mount, which put a model in memory before the user had touched anything.
    it("does not load a cached model — it says the click is cheap instead", async () => {
      cached = new Set([KOKORO]);
      renderPage();

      await waitFor(() =>
        expect(screen.getByTestId("model-size-note")).toHaveTextContent(
          /already downloaded/i,
        ),
      );
      expect(useTtsArgs).toHaveBeenLastCalledWith(KOKORO);
      expect(baseState.load).not.toHaveBeenCalled();
    });

    it("does NOT download a model that is not cached either", async () => {
      renderPage();

      await waitFor(() =>
        expect(screen.getByRole("button", { name: /load model/i })).toBeEnabled(),
      );
      // The guardrail: consent to a *download* is never inferred.
      expect(baseState.load).not.toHaveBeenCalled();
    });

    it("records nothing about the load, so the next visit asks again", () => {
      renderPage();
      fireEvent.click(screen.getByRole("button", { name: /load model/i }));
      expect(baseState.load).toHaveBeenCalledOnce();
      expect(JSON.stringify(useModelPrefs.getState())).not.toContain("Resume");
    });
  });

  it("renders the four slots, output included, before any result exists", () => {
    renderPage();
    expect(screen.getAllByRole("region")).toHaveLength(4);
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("offers a retry when the load failed", () => {
    mockState = {
      ...baseState,
      status: "error",
      idle: false,
      error: "download failed",
    };
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /^retry$/i }));
    expect(baseState.retry).toHaveBeenCalledOnce();
  });

  it("shows the voice picker for Kokoro and hides it for pipeline models", () => {
    mockState = { ...baseState, status: "ready", idle: false, ready: true };
    renderPage();
    // Kokoro is the default model → voice picker present.
    expect(screen.getByLabelText(/voice/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /mms english/i }));
    expect(screen.queryByLabelText(/voice/i)).not.toBeInTheDocument();
  });

  it("synthesises the text with the selected voice and plays it", async () => {
    mockState = { ...baseState, status: "ready", idle: false, ready: true, backend: "wasm" };
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
      idle: false,
      ready: true,
      result: { audio: new Float32Array(48000), sampleRate: 24000 }, // 2 s
    };
    renderPage();
    expect(screen.getByText(/2\.0s · 24 kHz/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^play$/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /download wav/i })).toBeInTheDocument();
  });

  it("surfaces a load error from the hook", () => {
    mockState = { ...baseState, status: "error", idle: false, error: "download failed" };
    renderPage();
    expect(screen.getByText(/download failed/i)).toBeInTheDocument();
  });
});
