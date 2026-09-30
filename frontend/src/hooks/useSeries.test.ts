import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ForecastSample } from "@/forecast/samples";

import { useSeries } from "./useSeries";

const sample: ForecastSample = {
  id: "s",
  label: "Sample",
  blurb: "",
  load: vi.fn().mockResolvedValue("2024-01-01,1\n2024-01-02,2\n2024-01-03,3\n"),
  licence: "CC0",
  source: "here",
  season: 7,
  horizon: 2,
};

describe("useSeries", () => {
  it("holds nothing, and reports nothing, until there is text", () => {
    const hook = renderHook(() => useSeries());
    expect(hook.result.current.series).toBeNull();
    expect(hook.result.current.error).toBeNull();
  });

  it("parses on every change, so the gap report is live", () => {
    const hook = renderHook(() => useSeries());
    act(() => hook.result.current.setText("2024-01-01,1\n2024-01-02,2\n2024-01-05,3\n"));
    expect(hook.result.current.series?.gaps).toHaveLength(1);
    expect(hook.result.current.error).toBeNull();
  });

  it("reports a parse error without dropping the text the user typed", () => {
    const hook = renderHook(() => useSeries());
    act(() => hook.result.current.setText("1\n2\noops\n"));
    expect(hook.result.current.series).toBeNull();
    expect(hook.result.current.error).toMatch(/Line 3/);
    expect(hook.result.current.text).toContain("oops");
  });

  it("loads a sample and credits it", async () => {
    const hook = renderHook(() => useSeries());
    await act(async () => {
      await hook.result.current.loadSample(sample);
    });
    expect(hook.result.current.sample).toBe(sample);
    expect(hook.result.current.series?.values.length).toBe(3);
  });

  it("stops crediting the sample once the text is edited", async () => {
    // Otherwise the panel keeps attributing a licence and a source to data the
    // user has since changed.
    const hook = renderHook(() => useSeries());
    await act(async () => {
      await hook.result.current.loadSample(sample);
    });
    act(() => hook.result.current.setText("1\n2\n3\n4\n"));
    expect(hook.result.current.sample).toBeNull();
  });

  it("constructs no worker, because this page has none", () => {
    // "No worker" is the design (see the note at the top of the hook), and a
    // later refactor could quietly add one — so it is asserted rather than
    // described.
    // happy-dom ships no `Worker`, so one is installed for the duration — which
    // also means a hook that reached for it would have found something.
    const spy = vi.fn();
    const original = (globalThis as { Worker?: unknown }).Worker;
    (globalThis as { Worker?: unknown }).Worker = spy;
    try {
      const hook = renderHook(() => useSeries());
      act(() => hook.result.current.setText("1\n2\n3\n"));
      expect(spy).not.toHaveBeenCalled();
    } finally {
      (globalThis as { Worker?: unknown }).Worker = original;
    }
  });
});
