import { renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { findEntry, useCatalogueEntry } from "./catalogue";

const MODELS = [{ id: "a" }, { id: "b" }] as const;

describe("findEntry", () => {
  it("finds the entry by id", () => {
    expect(findEntry(MODELS, "b")).toBe(MODELS[1]);
  });

  it("falls back to the first entry for an id the catalogue no longer has", () => {
    expect(findEntry(MODELS, "gone")).toBe(MODELS[0]);
  });
});

describe("useCatalogueEntry", () => {
  it("keeps the same object across renders while the id is unchanged", () => {
    const { result, rerender } = renderHook(({ id }) => useCatalogueEntry(MODELS, id), {
      initialProps: { id: "b" },
    });
    const first = result.current;
    rerender({ id: "b" });
    expect(result.current).toBe(first);
    rerender({ id: "a" });
    expect(result.current).toBe(MODELS[0]);
  });
});
