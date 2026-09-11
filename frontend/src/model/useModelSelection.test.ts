import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useModelPrefs } from "@/store/models";

import { useModelSelection } from "./useModelSelection";

vi.mock("./cache", () => ({
  cachedModels: vi.fn(async () => new Set<string>()),
  evictModel: vi.fn(async () => {}),
}));

const { cachedModels, evictModel } = await import("./cache");
const probeReturns = (...ids: string[]) =>
  vi.mocked(cachedModels).mockResolvedValue(new Set(ids));

const MODELS = [
  { id: "small", label: "Tiny" },
  { id: "large", label: "Big" },
] as const;

function setup(routeKey = "asr") {
  return renderHook(() =>
    useModelSelection({ routeKey, models: MODELS, fallback: MODELS[0] }),
  );
}

beforeEach(() => {
  localStorage.clear();
  useModelPrefs.setState({ selected: {} });
  probeReturns();
});
afterEach(() => vi.clearAllMocks());

describe("useModelSelection — the selection", () => {
  it("starts on the fallback and remembers a change", async () => {
    const { result } = setup();
    expect(result.current.model.id).toBe("small");

    act(() => result.current.setModel(MODELS[1]));
    expect(result.current.model.id).toBe("large");
    // Which is what a fresh mount — i.e. a page refresh — reads back.
    expect(setup().result.current.model.id).toBe("large");
  });

  it("falls back when the stored model is no longer in the catalogue", async () => {
    useModelPrefs.setState({ selected: { asr: "retired-model" } });
    const { result } = setup();

    expect(result.current.model.id).toBe("small");
    // …and the dead id is dropped rather than wedging every future mount.
    await waitFor(() =>
      expect(useModelPrefs.getState().selected.asr).toBeUndefined(),
    );
  });
});

// The point of the whole hook, and the thing it used to get wrong: selecting a
// model states an intent, it does not spend bandwidth. There is no `autoLoad`
// on the result any more — nothing here can put Machine A into `loading`, so
// there is no code path for a page to start downloading on its own.
describe("useModelSelection — selecting never loads", () => {
  it("exposes no way to start a load", () => {
    const { result } = setup();
    expect(result.current).not.toHaveProperty("autoLoad");
    expect(result.current).not.toHaveProperty("restoring");
  });

  it("does not load a cached model the user selected", async () => {
    probeReturns("small");
    const { result } = setup();

    await waitFor(() => expect(result.current.isCached).toBe(true));
    // Cached makes the click cheap. It does not make the click unnecessary.
    expect(result.current).not.toHaveProperty("autoLoad");
  });

  it("does not load a model that was loaded before the refresh", async () => {
    // The old store recorded this as `autoResume`, and a fresh mount turned it
    // into a download. A stale key from that era must not resurrect it.
    localStorage.setItem(
      "model-prefs",
      JSON.stringify({ state: { selected: { asr: "small" }, autoResume: { asr: true } }, version: 0 }),
    );
    probeReturns("small");
    await act(async () => {
      await useModelPrefs.persist.rehydrate();
    });
    const { result } = setup();

    await waitFor(() => expect(result.current.isCached).toBe(true));
    expect(result.current.model.id).toBe("small");
    expect(result.current).not.toHaveProperty("autoLoad");
  });
});

describe("useModelSelection — the LOAD actions", () => {
  it("onLoad runs the load it was handed, and records nothing", () => {
    const load = vi.fn();
    const { result } = setup();

    act(() => result.current.onLoad(load)());

    expect(load).toHaveBeenCalledOnce();
    // Nothing about the load is persisted, so the next visit asks again.
    expect(
      Object.keys(JSON.parse(localStorage.getItem("model-prefs") ?? "{}").state ?? {}),
    ).toEqual(["selected"]);
  });

  it("onCancel runs the cancel it was handed", () => {
    const cancel = vi.fn();
    const { result } = setup();

    act(() => result.current.onCancel(cancel)());

    expect(cancel).toHaveBeenCalledOnce();
  });
});

describe("useModelSelection — eviction", () => {
  it("drops the weights and re-probes", async () => {
    probeReturns("small");
    const { result } = setup();
    await waitFor(() => expect(result.current.isCached).toBe(true));

    probeReturns();
    await act(async () => {
      await result.current.evict("small");
    });

    expect(evictModel).toHaveBeenCalledWith("small");
    await waitFor(() => expect(result.current.isCached).toBe(false));
  });
});
