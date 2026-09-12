import { create } from "zustand";
import { immer } from "zustand/middleware/immer";

export type Theme = "light" | "dark" | "system";

/** Where the system panel's open/closed choice is remembered. */
const PANEL_KEY = "systemPanel";

function getInitialTheme(): Theme {
  if (typeof window === "undefined") return "system";
  const stored = localStorage.getItem("theme");
  if (stored === "light" || stored === "dark") return stored;
  return "system";
}

/**
 * Closed by default, and deliberately so: the panel samples nothing while it is
 * shut, and a user who has never asked for it should pay nothing for it.
 * A restored `true` is a decision the user made, the same way a restored theme is.
 */
function getInitialPanelOpen(): boolean {
  if (typeof window === "undefined") return false;
  try {
    return localStorage.getItem(PANEL_KEY) === "open";
  } catch {
    return false;
  }
}

function rememberPanel(open: boolean) {
  try {
    localStorage.setItem(PANEL_KEY, open ? "open" : "closed");
  } catch {
    /* a browser that refuses storage still gets a working panel */
  }
}

interface UIState {
  sidebarOpen: boolean;
  setSidebarOpen: (open: boolean) => void;
  toggleSidebar: () => void;
  /** Which sidebar task categories are expanded, keyed by category label. */
  expandedCategories: Record<string, boolean>;
  toggleCategory: (label: string) => void;
  setCategoryExpanded: (label: string, open: boolean) => void;
  /** The system panel (GPU / memory / CPU / storage readouts) on the right. */
  rightPanelOpen: boolean;
  toggleRightPanel: () => void;
  setRightPanelOpen: (open: boolean) => void;
  theme: Theme;
  setTheme: (theme: Theme) => void;
}

export const useUIStore = create<UIState>()(
  immer((set) => ({
    sidebarOpen: true,
    setSidebarOpen: (open) =>
      set((s) => {
        s.sidebarOpen = open;
      }),
    toggleSidebar: () =>
      set((s) => {
        s.sidebarOpen = !s.sidebarOpen;
      }),
    expandedCategories: {},
    toggleCategory: (label) =>
      set((s) => {
        s.expandedCategories[label] = !s.expandedCategories[label];
      }),
    setCategoryExpanded: (label, open) =>
      set((s) => {
        s.expandedCategories[label] = open;
      }),
    rightPanelOpen: getInitialPanelOpen(),
    toggleRightPanel: () =>
      set((s) => {
        s.rightPanelOpen = !s.rightPanelOpen;
        rememberPanel(s.rightPanelOpen);
      }),
    setRightPanelOpen: (open) =>
      set((s) => {
        s.rightPanelOpen = open;
        rememberPanel(open);
      }),
    theme: getInitialTheme(),
    setTheme: (theme) =>
      set((s) => {
        s.theme = theme;
      }),
  })),
);
