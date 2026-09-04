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
  useModelPrefs.setState({ selected: {}, autoResume: {} });
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

  it("does not carry consent over to a different model", async () => {
    useModelPrefs.setState({ autoResume: { asr: true } });
    const { result } = setup();
    act(() => result.current.setModel(MODELS[1]));
    expect(useModelPrefs.getState().autoResume.asr).toBe(false);
  });
});

describe("useModelSelection — resuming after a refresh", () => {
  it("never auto-loads before the cache probe has answered", async () => {
    useModelPrefs.setState({ autoResume: { asr: true } });
    const { result } = setup();
    // The whole guardrail: an undecided page downloads nothing.
    expect(result.current.autoLoad).toBe(false);
  });

  it("resumes a model that is already downloaded", async () => {
    useModelPrefs.setState({ autoResume: { asr: true } });
    probeReturns("small");
    const { result } = setup();

    await waitFor(() => expect(result.current.autoLoad).toBe(true));
    expect(result.current.isCached).toBe(true);
    // Labelled as what it is — a re-load from cache, not a surviving session.
    expect(result.current.restoring).toBe(true);
  });

  it("does NOT resume an uncached model, however recently it was used", async () => {
    useModelPrefs.setState({ autoResume: { asr: true } });
    probeReturns("large"); // a different model is cached
    const { result } = setup();

    await waitFor(() => expect(result.current.cached.size).toBe(1));
    expect(result.current.autoLoad).toBe(false);
    expect(result.current.restoring).toBe(false);
  });

  it("does not resume a cached model the user never asked for", async () => {
    probeReturns("small");
    const { result } = setup();

    await waitFor(() => expect(result.current.isCached).toBe(true));
    expect(result.current.autoLoad).toBe(false);
  });
});

describe("useModelSelection — recording consent", () => {
  it("onLoad runs the load and remembers it for next time", async () => {
    const load = vi.fn();
    const { result } = setup();

    act(() => result.current.onLoad(load)());
    expect(load).toHaveBeenCalledOnce();
    expect(useModelPrefs.getState().autoResume.asr).toBe(true);
    // A manual load is not a restore.
    expect(result.current.restoring).toBe(false);
  });

  it("a probe that lands after a manual load does not re-decide", async () => {
    useModelPrefs.setState({ autoResume: { asr: true } });
    probeReturns("small");
    const { result } = setup();

    act(() => result.current.onLoad(() => {})());
    await waitFor(() => expect(result.current.isCached).toBe(true));
    // Flipping autoLoad true here would tear the user's own load down.
    expect(result.current.autoLoad).toBe(false);
  });

  it("cancelling clears the consent, so a refresh does not restart it", () => {
    const cancel = vi.fn();
    useModelPrefs.setState({ autoResume: { asr: true } });
    const { result } = setup();

    act(() => result.current.onCancel(cancel)());
    expect(cancel).toHaveBeenCalledOnce();
    expect(useModelPrefs.getState().autoResume.asr).toBe(false);
  });
});

describe("useModelSelection — eviction", () => {
  it("drops the weights, the consent, and re-probes", async () => {
    useModelPrefs.setState({ autoResume: { asr: true } });
    probeReturns("small");
    const { result } = setup();
    await waitFor(() => expect(result.current.isCached).toBe(true));

    probeReturns();
    await act(async () => {
      await result.current.evict("small");
    });

    expect(evictModel).toHaveBeenCalledWith("small");
    expect(useModelPrefs.getState().autoResume.asr).toBe(false);
    await waitFor(() => expect(result.current.isCached).toBe(false));
  });
});
