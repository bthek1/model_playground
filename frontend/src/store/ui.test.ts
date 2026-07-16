import { describe, it, expect, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useUIStore } from "./ui";

describe("useUIStore — category expansion", () => {
  beforeEach(() => {
    useUIStore.setState({ sidebarOpen: true, expandedCategories: {} });
  });

  it("starts with no categories expanded", () => {
    const { result } = renderHook(() => useUIStore());
    expect(result.current.expandedCategories).toEqual({});
  });

  it("toggleCategory flips a category open then closed", () => {
    const { result } = renderHook(() => useUIStore());

    act(() => result.current.toggleCategory("Audio"));
    expect(result.current.expandedCategories.Audio).toBe(true);

    act(() => result.current.toggleCategory("Audio"));
    expect(result.current.expandedCategories.Audio).toBe(false);
  });

  it("setCategoryExpanded sets an explicit state without affecting others", () => {
    const { result } = renderHook(() => useUIStore());

    act(() => result.current.setCategoryExpanded("Tabular", true));
    expect(result.current.expandedCategories.Tabular).toBe(true);

    act(() => result.current.setCategoryExpanded("Tabular", false));
    expect(result.current.expandedCategories.Tabular).toBe(false);
    expect(result.current.expandedCategories.Audio).toBeUndefined();
  });
});
