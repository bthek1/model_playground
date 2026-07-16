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
};
const usePipeline = vi.fn(() => pipeState);
vi.mock("@/hooks/usePipeline", () => ({ usePipeline: () => usePipeline() }));

const { useAudioClassifier } = await import("./useAudioClassifier");

const ZERO_SHOT_MODEL = CLASSIFIER_MODELS.find(
  (m) => m.task === "zero-shot-audio-classification",
)!.id;

describe("useAudioClassifier", () => {
  afterEach(() => vi.clearAllMocks());

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
