import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

import { useModelPrefs } from "./models";

// What a page reads back after a refresh is whatever survived in localStorage,
// so these tests assert the persisted bytes as well as the in-memory state.
const persisted = () =>
  JSON.parse(localStorage.getItem("model-prefs") ?? "{}").state ?? {};

beforeEach(() => {
  localStorage.clear();
  useModelPrefs.setState({ selected: {}, autoResume: {} });
});

describe("useModelPrefs", () => {
  it("starts empty — a first visit has no opinion about any route", () => {
    const { result } = renderHook(() => useModelPrefs());
    expect(result.current.selected).toEqual({});
    expect(result.current.autoResume).toEqual({});
  });

  it("stores a selection per route, and writes it through to localStorage", () => {
    const { result } = renderHook(() => useModelPrefs());

    act(() => result.current.selectModel("asr", "onnx-community/whisper-base"));
    act(() => result.current.selectModel("tts", "onnx-community/Kokoro-82M"));

    expect(result.current.selected).toEqual({
      asr: "onnx-community/whisper-base",
      tts: "onnx-community/Kokoro-82M",
    });
    // Routes must not read each other's choices.
    expect(persisted().selected.asr).toBe("onnx-community/whisper-base");
  });

  it("does not carry a load consent over to a different model", () => {
    const { result } = renderHook(() => useModelPrefs());
    act(() => result.current.setAutoResume("asr", true));

    act(() => result.current.selectModel("asr", "onnx-community/moonshine-tiny"));

    // A different model is a different download: the old consent says nothing
    // about it, and resuming on it would spend bandwidth unasked.
    expect(result.current.autoResume.asr).toBe(false);
  });

  it("records and clears the intent to load", () => {
    const { result } = renderHook(() => useModelPrefs());

    act(() => result.current.setAutoResume("tts", true));
    expect(persisted().autoResume.tts).toBe(true);

    act(() => result.current.setAutoResume("tts", false));
    expect(result.current.autoResume.tts).toBe(false);
  });

  it("forgets a route entirely, for a model id we no longer ship", () => {
    const { result } = renderHook(() => useModelPrefs());
    act(() => result.current.selectModel("asr", "retired/model"));
    act(() => result.current.setAutoResume("asr", true));

    act(() => result.current.forget("asr"));

    expect(result.current.selected.asr).toBeUndefined();
    expect(result.current.autoResume.asr).toBeUndefined();
  });

  it("survives a store rehydration — the point of persisting at all", async () => {
    const { result } = renderHook(() => useModelPrefs());
    act(() => result.current.selectModel("tts", "kokoro"));
    act(() => result.current.setAutoResume("tts", true));

    // Stand in for a page reload: drop the in-memory state, then rehydrate the
    // way a fresh page load does. The stored copy is re-seeded first because
    // clearing the state persists *that* — a real reload never sees the write.
    const stored = localStorage.getItem("model-prefs")!;
    useModelPrefs.setState({ selected: {}, autoResume: {} });
    localStorage.setItem("model-prefs", stored);
    await act(async () => {
      await useModelPrefs.persist.rehydrate();
    });

    expect(useModelPrefs.getState().selected.tts).toBe("kokoro");
    expect(useModelPrefs.getState().autoResume.tts).toBe(true);
  });

  it("keeps working when storage is unavailable", () => {
    // Private-browsing modes throw on access rather than returning null; a
    // preference is a convenience, so the page must not care.
    const original = globalThis.localStorage;
    Object.defineProperty(globalThis, "localStorage", {
      value: {
        getItem: () => {
          throw new Error("access denied");
        },
        setItem: () => {
          throw new Error("access denied");
        },
        removeItem: () => {
          throw new Error("access denied");
        },
      },
      configurable: true,
    });

    const { result } = renderHook(() => useModelPrefs());
    expect(() =>
      act(() => result.current.selectModel("asr", "some/model")),
    ).not.toThrow();
    expect(result.current.selected.asr).toBe("some/model");

    Object.defineProperty(globalThis, "localStorage", {
      value: original,
      configurable: true,
    });
  });
});
