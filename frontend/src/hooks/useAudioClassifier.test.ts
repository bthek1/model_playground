import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  CLASSIFIER_MODELS,
  DEFAULT_CLASSIFIER_MODEL,
} from "@/audio/classification";

// Mock the generic pipeline hook so we test only the classifier's arg-shaping.
const run = vi.fn();
const pipeState = {
  status: "ready" as const,
  loading: false,
  ready: true,
  progress: null,
  backend: "wasm" as string | null,
  running: false,
  error: null as string | null,
  run,
  load: vi.fn(),
  retry: vi.fn(),
};
// Records its arguments: the classifier's job includes choosing the pipeline
// task from the catalogue and forwarding `autoLoad`, so dropping the args here
// would hide both.
const usePipeline = vi.fn((task: string, model: string, autoLoad?: boolean) => {
  void task;
  void model;
  void autoLoad;
  return pipeState;
});
vi.mock("@/hooks/usePipeline", () => ({
  usePipeline: (task: string, model: string, autoLoad?: boolean) =>
    usePipeline(task, model, autoLoad),
}));

const { useAudioClassifier } = await import("./useAudioClassifier");

const ZERO_SHOT_MODEL = CLASSIFIER_MODELS.find(
  (m) => m.task === "zero-shot-audio-classification",
)!.id;

describe("useAudioClassifier", () => {
  afterEach(() => vi.clearAllMocks());

  it("selects the pipeline task from the catalogue entry", () => {
    renderHook(() => useAudioClassifier(DEFAULT_CLASSIFIER_MODEL));
    expect(usePipeline).toHaveBeenCalledWith(
      "audio-classification",
      DEFAULT_CLASSIFIER_MODEL,
      false,
    );

    vi.clearAllMocks();
    renderHook(() => useAudioClassifier(ZERO_SHOT_MODEL));
    expect(usePipeline).toHaveBeenCalledWith(
      "zero-shot-audio-classification",
      ZERO_SHOT_MODEL,
      false,
    );
  });

  // The default is `false`, both here and in `usePipeline`. It used to be
  // `true`, which made "forget the second argument" a silent download.
  it("defers the download by default, and forwards an explicit opt-in", () => {
    renderHook(() => useAudioClassifier(DEFAULT_CLASSIFIER_MODEL));
    expect(usePipeline).toHaveBeenCalledWith(
      expect.any(String),
      DEFAULT_CLASSIFIER_MODEL,
      false,
    );

    vi.clearAllMocks();
    renderHook(() => useAudioClassifier(DEFAULT_CLASSIFIER_MODEL, true));
    expect(usePipeline).toHaveBeenCalledWith(
      expect.any(String),
      DEFAULT_CLASSIFIER_MODEL,
      true,
    );
  });

  it("re-exposes the load actions so the LOAD slot can drive them", () => {
    const { result } = renderHook(() =>
      useAudioClassifier(DEFAULT_CLASSIFIER_MODEL, false),
    );
    result.current.load();
    result.current.retry();
    expect(pipeState.load).toHaveBeenCalledOnce();
    expect(pipeState.retry).toHaveBeenCalledOnce();
  });

  it("falls back to the first catalogue entry for an unknown model id", () => {
    renderHook(() => useAudioClassifier("not-a-model"));
    expect(usePipeline).toHaveBeenCalledWith(
      CLASSIFIER_MODELS[0].task,
      CLASSIFIER_MODELS[0].id,
      false,
    );
  });

  it("passes { top_k } for a fixed-label model and stores the result", async () => {
    run.mockResolvedValue([{ label: "Speech", score: 0.9 }]);
    const { result } = renderHook(() => useAudioClassifier(DEFAULT_CLASSIFIER_MODEL));
    expect(result.current.isZeroShot).toBe(false);

    await act(async () => {
      await result.current.classify(new Float32Array([0.1]));
    });

    expect(run).toHaveBeenCalledWith(expect.any(Float32Array), [{ top_k: 6 }]);
    await waitFor(() =>
      expect(result.current.result).toEqual([{ label: "Speech", score: 0.9 }]),
    );
  });

  it("passes trimmed candidate labels for a zero-shot model", async () => {
    run.mockResolvedValue([{ label: "a dog", score: 0.7 }]);
    const { result } = renderHook(() => useAudioClassifier(ZERO_SHOT_MODEL));
    expect(result.current.isZeroShot).toBe(true);

    await act(async () => {
      await result.current.classify(new Float32Array([0]), ["  a dog ", "", "rain "]);
    });

    expect(run).toHaveBeenCalledWith(expect.any(Float32Array), [["a dog", "rain"]]);
  });

  it("errors (without calling run) when a zero-shot model gets no labels", async () => {
    const { result } = renderHook(() => useAudioClassifier(ZERO_SHOT_MODEL));

    await act(async () => {
      await result.current.classify(new Float32Array([0]), ["  ", ""]);
    });

    expect(run).not.toHaveBeenCalled();
    await waitFor(() =>
      expect(result.current.error).toMatch(/at least one label/i),
    );
  });
});
