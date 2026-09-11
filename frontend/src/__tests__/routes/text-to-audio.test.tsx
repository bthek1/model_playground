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
  idle: false,
  loading: false,
  ready: true,
  progress: null,
  loadProgress: null,
  loadedInMs: null,
  backend: "wasm",
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

// The gate is not "don't mount the hook" — the hook is always mounted and
// simply starts `idle`. The spy records its arguments forwarded verbatim,
// arity included: the route passes the model and *nothing else*, and "there is
// no second argument" is what guarantees nothing downloads on arrival.
const useTts = vi.fn((...args: unknown[]) => {
  void args;
  return mockState;
});
vi.mock("@/hooks/useTts", () => ({
  useTts: (...args: unknown[]) => useTts(...args),
}));

const { Route } = await import("@/routes/text-to-audio");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Text-to-audio route component not found");
  render(<Page />);
}

/** The generator is always mounted now; only its load state changes. */
function ready(extra: Partial<UseTtsResult> = {}) {
  mockState = { ...baseState, ...extra };
}

describe("TextToAudioPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockState = { ...baseState };
  });

  it("gates the model behind an explicit opt-in", () => {
    mockState = { ...baseState, status: "idle", idle: true, ready: false };
    renderPage();

    expect(
      screen.getByRole("heading", { name: /Text to Audio/i }),
    ).toBeInTheDocument();
    expect(screen.getByText(/Experimental — and slow/i)).toBeInTheDocument();
    // Nothing may download before the user accepts the cost. The hook is
    // mounted — that is free — and it is handed no auto-load option at all,
    // because its default is `idle`.
    expect(useTts).toHaveBeenCalledWith(expect.any(String));
    expect(baseState.load).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: /^Generate/ })).toBeDisabled();
  });

  it("states the download size and the speed caveat before downloading", () => {
    mockState = { ...baseState, status: "idle", idle: true, ready: false };
    renderPage();

    // The two things that would surprise a user, both up front. The size appears
    // twice — once in the picker line, once in the notice — and for a 571 MB
    // download that restatement is deliberate.
    expect(screen.getAllByText(/on WebGPU · .* on WASM/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/autoregressive/i).length).toBeGreaterThan(0);
  });

  it("starts the download from the LOAD slot when the user asks", () => {
    mockState = { ...baseState, status: "idle", idle: true, ready: false };
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: /load model/i }));
    expect(baseState.load).toHaveBeenCalledOnce();
  });

  it("drops the warning once the model is loaded", () => {
    ready();
    renderPage();

    expect(screen.queryByText(/Experimental — and slow/i)).toBeNull();
    expect(screen.getByRole("button", { name: /^Generate/ })).toBeEnabled();
    expect(screen.getByLabelText(/Prompt/i)).toBeInTheDocument();
  });

  it("generates from the prompt and plays the result", async () => {
    const audio = { audio: new Float32Array([0.1, 0.2]), sampleRate: 32000 };
    mockSynthesize.mockResolvedValue(audio);
    renderPage();

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

    expect(screen.getByRole("button", { name: /Generating/i })).toBeDisabled();
  });

  it("surfaces a load or generation error", () => {
    mockState = { ...baseState, status: "error", ready: false, error: "boom" };
    renderPage();

    expect(screen.getByText("boom")).toBeInTheDocument();
  });

  it("offers replay and WAV download once a clip exists", () => {
    mockState = {
      ...baseState,
      result: { audio: new Float32Array([0.1]), sampleRate: 32000 },
    };
    renderPage();

    expect(screen.getByText(/Generated audio/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Play/ })).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Download WAV/ }),
    ).toBeInTheDocument();
  });
});
