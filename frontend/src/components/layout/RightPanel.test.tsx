import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useUIStore } from "@/store/ui";

// The panel body owns the sampling loop; this file is about the shell, so the
// body is stubbed and its own loop is tested in telemetry/useTelemetry.test.ts.
const systemPanel = vi.fn();
vi.mock("@/components/telemetry/SystemPanel", () => ({
  SystemPanel: (props: { active: boolean }) => {
    systemPanel(props);
    return <div data-testid="system-panel-body">cards</div>;
  },
}));

const matches = vi.fn().mockReturnValue(true);
vi.mock("@/hooks/useMediaQuery", () => ({
  useMediaQuery: (query: string) => matches(query) as boolean,
}));

import { RightPanel } from "./RightPanel";

describe("RightPanel", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
    matches.mockReturnValue(true);
    useUIStore.setState({ rightPanelOpen: false });
  });

  it("renders nothing while closed — and mounts no sampling loop", () => {
    render(<RightPanel />);
    expect(screen.queryByTestId("right-panel")).not.toBeInTheDocument();
    expect(systemPanel).not.toHaveBeenCalled();
  });

  it("docks beside the workbench on a wide screen", () => {
    useUIStore.setState({ rightPanelOpen: true });
    render(<RightPanel />);

    const panel = screen.getByTestId("right-panel");
    expect(panel.tagName).toBe("ASIDE");
    expect(screen.getByLabelText("System panel")).toBe(panel);
    expect(systemPanel).toHaveBeenCalledWith({ active: true });
  });

  it("mounts exactly one panel body, so there is only one sampling loop", () => {
    useUIStore.setState({ rightPanelOpen: true });
    render(<RightPanel />);
    expect(screen.getAllByTestId("system-panel-body")).toHaveLength(1);
    expect(systemPanel).toHaveBeenCalledTimes(1);
  });

  it("becomes an overlay below lg rather than squeezing the workbench", () => {
    matches.mockReturnValue(false);
    useUIStore.setState({ rightPanelOpen: true });
    render(<RightPanel />);

    expect(screen.getByTestId("right-panel").tagName).not.toBe("ASIDE");
    expect(screen.getByText("System")).toBeInTheDocument();
    expect(screen.getAllByTestId("system-panel-body")).toHaveLength(1);
  });

  it("closes from its own header", async () => {
    const user = userEvent.setup();
    useUIStore.setState({ rightPanelOpen: true });
    render(<RightPanel />);

    await user.click(screen.getByLabelText("Close system panel"));
    expect(useUIStore.getState().rightPanelOpen).toBe(false);
  });

  it("toggles on Alt+Shift+M, from closed and from open", async () => {
    const user = userEvent.setup();
    render(<RightPanel />);

    await user.keyboard("{Alt>}{Shift>}M{/Shift}{/Alt}");
    expect(useUIStore.getState().rightPanelOpen).toBe(true);

    await user.keyboard("{Alt>}{Shift>}M{/Shift}{/Alt}");
    expect(useUIStore.getState().rightPanelOpen).toBe(false);
  });

  it("ignores the shortcut while typing in a field", async () => {
    const user = userEvent.setup();
    render(
      <>
        <input aria-label="prompt" />
        <RightPanel />
      </>,
    );

    await user.click(screen.getByLabelText("prompt"));
    await user.keyboard("{Alt>}{Shift>}M{/Shift}{/Alt}");
    expect(useUIStore.getState().rightPanelOpen).toBe(false);
  });

  it("stops listening for the shortcut once unmounted", async () => {
    const user = userEvent.setup();
    const { unmount } = render(<RightPanel />);
    unmount();

    await user.keyboard("{Alt>}{Shift>}M{/Shift}{/Alt}");
    expect(useUIStore.getState().rightPanelOpen).toBe(false);
  });
});
