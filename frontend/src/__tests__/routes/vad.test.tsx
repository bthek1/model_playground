import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { VadResult } from "@/audio/vad/types";
import type { ModelTask } from "@/model/types";

// No Web Audio in the test env: decoding and capture are mocked.
const decodeToMono = vi.fn(async () => new Float32Array([0.1, 0.2, 0.3]));
const recordMic = vi.fn(async () => new Float32Array([0.4, 0.5]));
vi.mock("@/audio/io", () => ({
  decodeToMono: (...args: unknown[]) => decodeToMono(...(args as [])),
  recordMic: (...args: unknown[]) => recordMic(...(args as [])),
  play: vi.fn(),
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

const mockRun = vi.fn<(audio: Float32Array) => Promise<VadResult>>();
const mockLoad = vi.fn();
const mockRetry = vi.fn();
const useVadSpy = vi.fn();

type VadTask = ModelTask<Float32Array, VadResult>;

const baseState: VadTask = {
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
let mockState: VadTask = { ...baseState };

vi.mock("@/hooks/useVad", () => ({
  useVad: (...args: unknown[]) => {
    useVadSpy(...args);
    return mockState;
  },
}));

const { Route } = await import("@/routes/vad");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("VAD route component not found");
  render(<Page />);
}

const readyState: VadTask = {
  ...baseState,
  status: "ready",
  idle: false,
  ready: true,
  backend: "wasm",
};

/** 12 frames: quiet, then loud, then quiet — one segment in the middle. */
function detection(): VadResult {
  const probabilities = Float32Array.from([
    0.01, 0.02, 0.9, 0.95, 0.97, 0.93, 0.91, 0.94, 0.9, 0.92, 0.03, 0.01,
  ]);
  return { probabilities, frameSamples: 512, sampleRate: 16000, samples: 12 * 512 };
}

describe("VadPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockState = { ...baseState };
  });

  it("renders all four slots, with the output empty before any run", () => {
    renderPage();
    for (const slot of ["slot-1", "slot-2", "slot-3", "slot-4"]) {
      expect(screen.getByTestId(slot)).toBeInTheDocument();
    }
    expect(screen.getByTestId("output-panel")).toBeInTheDocument();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("keeps the four slots in pipeline order in the DOM", () => {
    renderPage();
    const order = screen
      .getAllByTestId(/^slot-[1-4]$/)
      .map((el) => el.dataset.testid);
    expect(order).toEqual(["slot-1", "slot-2", "slot-3", "slot-4"]);
  });

  it("downloads nothing on mount", () => {
    renderPage();
    // Both halves matter: `autoLoad: false` is what keeps the worker unspawned,
    // and `load` not being called is what proves nothing routed around it.
    expect(useVadSpy).toHaveBeenCalledWith(expect.any(String), false);
    expect(mockLoad).not.toHaveBeenCalled();
  });

  it("offers both detectors, including the one with no download", () => {
    renderPage();
    expect(
      screen.getByRole("button", { name: /silero vad/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /energy vad/i }),
    ).toBeInTheDocument();
  });

  it("states the download cost once, and does not warn about 2 MB", () => {
    renderPage();
    expect(screen.getByTestId("model-size-note")).toHaveTextContent(/MB on WebGPU/);
    expect(screen.queryByTestId("model-size-warning")).not.toBeInTheDocument();
  });

  it("loads from the LOAD slot", () => {
    renderPage();
    fireEvent.click(
      within(screen.getByTestId("slot-2")).getByRole("button", {
        name: /load model/i,
      }),
    );
    expect(mockLoad).toHaveBeenCalledTimes(1);
  });

  it("keeps the transport disabled until the model is ready", () => {
    renderPage();
    expect(screen.getByRole("button", { name: /record/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /upload audio/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /^jfk$/i })).toBeDisabled();
  });

  it("decodes at the default 16 kHz — this is not the 48 kHz route", async () => {
    mockState = readyState;
    mockRun.mockResolvedValue(detection());
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: /record/i }));
    await waitFor(() => expect(recordMic).toHaveBeenCalled());
    // No explicit rate: `recordMic`'s own default is 16 kHz. Passing 48000 here
    // would be the audio-to-audio assumption leaking across.
    expect(recordMic).toHaveBeenCalledWith(expect.any(Number));
  });

  it("hands the decoded samples straight to run, and still draws them", async () => {
    mockState = { ...readyState, result: detection() };
    mockRun.mockResolvedValue(mockState.result!);
    renderPage();

    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(input, {
      target: { files: [new File(["x"], "take.wav", { type: "audio/wav" })] },
    });

    await waitFor(() => expect(decodeToMono).toHaveBeenCalled());
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    // The decoded array itself goes to the worker — `useVad` transfers its
    // buffer rather than copying a long take across the boundary.
    expect(mockRun.mock.calls[0][0]).toBe(
      await decodeToMono.mock.results[0].value,
    );
    // And the timeline still renders, which it could not do if the route had
    // kept the (now detached) original instead of a copy.
    await waitFor(() =>
      expect(
        screen.getByRole("img", { name: /speech probability/i }),
      ).toBeInTheDocument(),
    );
  });

  it("shows the segments a detection implies", async () => {
    mockState = { ...readyState, result: detection() };
    mockRun.mockResolvedValue(mockState.result!);
    renderPage();

    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(input, {
      target: { files: [new File(["x"], "a.wav", { type: "audio/wav" })] },
    });

    // Frames 2–9 at 32 ms each: 0:00 → 0:00, and 8 frames is 0.3s.
    await waitFor(() =>
      expect(screen.getByText(/0:00 → 0:00/)).toBeInTheDocument(),
    );
    expect(screen.getByText(/1 segment/)).toBeInTheDocument();
  });

  it("re-derives segments from the threshold without re-running the model", async () => {
    mockState = { ...readyState, result: detection() };
    mockRun.mockResolvedValue(mockState.result!);
    renderPage();

    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(input, {
      target: { files: [new File(["x"], "a.wav", { type: "audio/wav" })] },
    });
    await waitFor(() => expect(screen.getByText(/1 segment/)).toBeInTheDocument());
    expect(mockRun).toHaveBeenCalledTimes(1);

    // Above every probability in the clip: the segments must vanish, and the
    // model must not be asked again.
    // By role, not by label: the timeline canvas describes itself with the
    // threshold too, so `getByLabelText` matches both.
    fireEvent.change(screen.getByRole("slider"), { target: { value: "0.95" } });
    await waitFor(() =>
      expect(screen.getByText(/no speech above this threshold/i)).toBeInTheDocument(),
    );
    expect(mockRun).toHaveBeenCalledTimes(1);
  });

  it("surfaces a load error in the LOAD slot and offers a retry", () => {
    mockState = {
      ...baseState,
      status: "error",
      idle: false,
      error: "onnx/model.onnx: 404 Not Found",
    };
    renderPage();

    const load = within(screen.getByTestId("slot-2"));
    expect(load.getByTestId("error-note")).toHaveTextContent(/404/);
    fireEvent.click(load.getByRole("button", { name: /retry/i }));
    expect(mockRetry).toHaveBeenCalledTimes(1);
  });

  it("shows a decode failure in the RUN slot, not against the model", async () => {
    mockState = readyState;
    decodeToMono.mockRejectedValueOnce(new Error("Unsupported audio format"));
    renderPage();

    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(input, {
      target: { files: [new File(["x"], "a.txt", { type: "text/plain" })] },
    });

    await waitFor(() =>
      expect(
        within(screen.getByTestId("slot-3")).getByTestId("error-note"),
      ).toHaveTextContent(/unsupported audio format/i),
    );
    expect(mockRun).not.toHaveBeenCalled();
  });
});
