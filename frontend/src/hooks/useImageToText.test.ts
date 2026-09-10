import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { CaptionResult } from "@/vision/caption/types";

const post = vi.fn();
const workerState = {
  status: "ready" as const,
  idle: false,
  loading: false,
  ready: true,
  progress: null,
  loadProgress: null,
  loadedInMs: 5000,
  backend: "webgpu",
  running: false,
  result: null,
  error: null,
  load: vi.fn(),
  retry: vi.fn(),
  cancel: vi.fn(),
  run: post,
};
const useModelWorker = vi.fn(() => workerState);
vi.mock("@/model/useModelWorker", () => ({
  useModelWorker: (...a: unknown[]) => useModelWorker(...(a as [])),
}));
vi.mock("@/vision/caption/client", () => ({ createCaptionWorker: vi.fn() }));

const { useImageToText } = await import("@/hooks/useImageToText");

const image = {
  width: 8,
  height: 8,
  channels: 3,
  data: new Uint8ClampedArray(8 * 8 * 3),
} as never;

const reply: CaptionResult = {
  kind: "text",
  mode: "<CAPTION>",
  text: "two cats on a sofa",
  ms: 1800,
};

const FLORENCE = "onnx-community/Florence-2-base-ft";
const VITGPT2 = "Xenova/vit-gpt2-image-captioning";

beforeEach(() => {
  vi.clearAllMocks();
  post.mockResolvedValue(reply);
});

describe("useImageToText", () => {
  it("keys the worker on the model and carries the family", () => {
    // The family is what tells the worker which of the two implementations to
    // build — the pipeline cannot load Florence-2 at all.
    renderHook(() => useImageToText());
    expect(useModelWorker).toHaveBeenCalledWith(
      expect.objectContaining({
        key: `caption:${FLORENCE}`,
        autoLoad: false,
        loadMessage: { model: FLORENCE, family: "florence2" },
      }),
    );
  });

  it("offers only the modes the selected checkpoint declares", () => {
    // A task token a model has never seen does not error: it produces a
    // confident, fluent, unrelated sentence.
    const florence = renderHook(() => useImageToText(FLORENCE));
    expect(florence.result.current.modes).toEqual([
      "<CAPTION>",
      "<DETAILED_CAPTION>",
      "<OCR>",
      "<OD>",
    ]);

    const vitgpt2 = renderHook(() => useImageToText(VITGPT2));
    expect(vitgpt2.result.current.modes).toEqual(["<CAPTION>"]);
  });

  it("ignores a mode the model does not declare", () => {
    const { result } = renderHook(() => useImageToText(VITGPT2));
    act(() => result.current.setMode("<OCR>"));
    expect(result.current.mode).toBe("<CAPTION>");
  });

  it("resets an unsupported mode when the model changes, rather than sending it", () => {
    const { result, rerender } = renderHook(
      ({ model }) => useImageToText(model),
      { initialProps: { model: FLORENCE } },
    );
    act(() => result.current.setMode("<OCR>"));
    expect(result.current.mode).toBe("<OCR>");

    rerender({ model: VITGPT2 });
    expect(result.current.mode).toBe("<CAPTION>");
  });

  it("sends the task token verbatim, with the mode's own token budget", async () => {
    const { result } = renderHook(() => useImageToText(FLORENCE));
    act(() => result.current.setMode("<DETAILED_CAPTION>"));
    await act(async () => {
      await result.current.run(image);
    });

    const [payload, transfer] = post.mock.calls[0];
    expect(payload).toMatchObject({
      mode: "<DETAILED_CAPTION>",
      maxNewTokens: 256,
    });
    // The image buffer is transferred, not copied — the page keeps its own
    // RawImage for the preview and the overlay.
    expect(transfer).toHaveLength(1);
  });

  it("caps a plain caption far below a detailed one", async () => {
    // A caption is a sentence and a detailed one is a paragraph; the cap is
    // what keeps the cheap mode cheap.
    const { result } = renderHook(() => useImageToText(FLORENCE));
    await act(async () => {
      await result.current.run(image);
    });
    expect(post.mock.calls[0][0].maxNewTokens).toBeLessThan(256);
  });

  it("keeps the last answer as `result`", async () => {
    const { result } = renderHook(() => useImageToText(FLORENCE));
    await act(async () => {
      await result.current.run(image);
    });
    await waitFor(() => expect(result.current.result).toEqual(reply));
  });
});
