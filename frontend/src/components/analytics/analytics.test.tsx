import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const flags = vi.hoisted(() => ({ enabled: false }));
vi.mock("@/lib/features", () => ({
  get ANALYTICS_ENABLED() {
    return flags.enabled;
  },
  POSTHOG_KEY: "phc_test",
  POSTHOG_HOST: "/ingest",
}));

import { OPT_OUT_KEY } from "@/analytics";

import { ANALYTICS_NOTE, AnalyticsNote } from "./AnalyticsNote";
import { AnalyticsToggle } from "./AnalyticsToggle";

beforeEach(() => {
  localStorage.clear();
  flags.enabled = false;
});
afterEach(cleanup);

describe("a build without analytics says nothing about it", () => {
  it("renders neither the note nor the switch", () => {
    render(
      <>
        <AnalyticsNote />
        <AnalyticsToggle />
      </>,
    );
    expect(screen.queryByTestId("analytics-note")).toBeNull();
    expect(screen.queryByTestId("analytics-toggle")).toBeNull();
  });
});

describe("a build with analytics", () => {
  beforeEach(() => {
    flags.enabled = true;
  });

  it("states exactly what is collected, and what never is", () => {
    render(<AnalyticsNote />);
    expect(screen.getByTestId("analytics-note")).toHaveTextContent(ANALYTICS_NOTE);
    expect(ANALYTICS_NOTE).toBe(
      "Anonymous counts of which pages and models are used — never your inputs or results.",
    );
  });

  it("the switch is on by default, beside the same sentence", () => {
    render(<AnalyticsToggle />);
    const toggle = screen.getByRole("switch", { name: /anonymous usage analytics/i });
    expect(toggle).toBeChecked();
    expect(screen.getByTestId("analytics-toggle")).toHaveTextContent(ANALYTICS_NOTE);
  });

  it("turning it off is stored under its own key, and is read back on the next render", () => {
    render(<AnalyticsToggle />);
    fireEvent.click(screen.getByRole("switch"));
    expect(localStorage.getItem(OPT_OUT_KEY)).toBe("1");
    cleanup();
    render(<AnalyticsToggle />); // a reload
    expect(screen.getByRole("switch")).not.toBeChecked();
    fireEvent.click(screen.getByRole("switch"));
    expect(localStorage.getItem(OPT_OUT_KEY)).toBeNull();
  });
});
