import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { parseCsv } from "@/tabular/csv";
import { MAX_ROWS } from "@/tabular/limits";
import type { SampleDataset } from "@/tabular/samples";

import { useDataset } from "./useDataset";

const CSV = "a,b\n1,x\n2,y\n";

const sample: SampleDataset = {
  id: "s",
  label: "Sample",
  blurb: "",
  load: () => Promise.resolve(CSV),
  licence: "CC0",
  source: "here",
  target: "b",
};

function fakeParse() {
  return vi.fn(async (text: string, options: { name?: string; maxRows?: number } = {}) =>
    parseCsv(text, options),
  );
}

describe("useDataset", () => {
  it("holds nothing until asked", () => {
    const hook = renderHook(() => useDataset(fakeParse()));
    expect(hook.result.current.dataset).toBeNull();
    expect(hook.result.current.parsing).toBe(false);
  });

  it("parses a sample in the worker, capped at MAX_ROWS", async () => {
    const parse = fakeParse();
    const hook = renderHook(() => useDataset(parse));
    await act(async () => {
      await hook.result.current.loadSample(sample);
    });
    expect(parse).toHaveBeenCalledWith(CSV, { name: "Sample", maxRows: MAX_ROWS });
    expect(hook.result.current.dataset?.rowCount).toBe(2);
    expect(hook.result.current.sample).toBe(sample);
  });

  it("reports a parse failure and keeps no half-loaded dataset", async () => {
    const parse = vi.fn(async () => {
      throw new Error("The file is empty.");
    });
    const hook = renderHook(() => useDataset(parse as never));
    await act(async () => {
      await hook.result.current.loadSample(sample);
    });
    expect(hook.result.current.error).toMatch(/empty/i);
    expect(hook.result.current.dataset).toBeNull();
  });

  it("lands the newest parse, not whichever finishes last", async () => {
    // Dropping a second file while the first is still parsing is an ordinary
    // thing to do, and the slower of the two arriving last would quietly
    // replace the file the user is looking at.
    let resolveFirst: (v: unknown) => void = () => {};
    const parse = vi
      .fn()
      .mockImplementationOnce(
        () => new Promise((r) => {
          resolveFirst = r;
        }),
      )
      .mockImplementationOnce(async () => parseCsv("z\n9\n", { name: "second" }));
    const hook = renderHook(() => useDataset(parse as never));
    act(() => {
      void hook.result.current.loadSample({ ...sample, label: "first" });
    });
    await act(async () => {
      await hook.result.current.loadSample({ ...sample, label: "second" });
    });
    act(() => resolveFirst(parseCsv(CSV, { name: "first" })));
    await waitFor(() =>
      expect(hook.result.current.dataset?.columns[0].name).toBe("z"),
    );
  });

  it("never writes the dataset anywhere it could outlive the tab", async () => {
    // The page's whole claim. `lib/mnistCache.ts` and `lib/proteinsCache.ts`
    // both cache their dataset to IndexedDB and are right to — they cache a
    // public benchmark. This one would be caching someone's payroll.
    const setItem = vi.spyOn(Storage.prototype, "setItem");
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const hook = renderHook(() => useDataset(fakeParse()));
    await act(async () => {
      await hook.result.current.loadSample(sample);
    });
    expect(setItem).not.toHaveBeenCalled();
    expect(fetchSpy).not.toHaveBeenCalled();
    setItem.mockRestore();
    fetchSpy.mockRestore();
  });

  it("clears back to empty", async () => {
    const hook = renderHook(() => useDataset(fakeParse()));
    await act(async () => {
      await hook.result.current.loadSample(sample);
    });
    act(() => hook.result.current.clear());
    expect(hook.result.current.dataset).toBeNull();
    expect(hook.result.current.sample).toBeNull();
  });
});
