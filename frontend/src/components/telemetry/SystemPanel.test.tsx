import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { TelemetryView } from "@/telemetry/useTelemetry";

const useTelemetry = vi.fn();
vi.mock("@/telemetry/useTelemetry", () => ({
  useTelemetry: (active: boolean) => useTelemetry(active) as TelemetryView,
}));

import { SystemPanel } from "./SystemPanel";

const view: TelemetryView = {
  tick: 7,
  intervalMs: 1000,
  memory: { status: "unavailable", reason: "no performance.memory here" },
  memoryHistory: [],
  cpu: { lagMs: 4, longTasks: null, longTaskMs: null, busyFraction: 0 },
  lagHistory: [4, 4],
  busyHistory: [0, 0],
  longTaskReason: "Chromium only.",
  gpu: { status: "ok", value: { liveBytes: 0, peakBytes: 0, buffers: 0, lastPassMs: null } },
  gpuHistory: [],
  identity: { status: "unavailable", reason: "no adapter" },
  storage: { status: "ok", value: { usageBytes: 10, quotaBytes: 100, cachedModels: 1 } },
  download: null,
  downloadHistory: [],
  capacity: {
    cores: { status: "ok", value: 8 },
    deviceMemoryGiB: { status: "unavailable", reason: "not reported" },
  },
};

describe("SystemPanel", () => {
  it("renders all four cards", () => {
    useTelemetry.mockReturnValue(view);
    render(<SystemPanel active />);

    for (const id of ["gpu", "memory", "cpu", "storage"]) {
      expect(screen.getByTestId(`telemetry-${id}`)).toBeInTheDocument();
    }
  });

  it("says up front that this is load, not utilisation", () => {
    useTelemetry.mockReturnValue(view);
    render(<SystemPanel active />);
    expect(
      screen.getByText(/load and capacity, never utilisation/),
    ).toBeInTheDocument();
  });

  it("passes its active flag to the sampling loop", () => {
    useTelemetry.mockReturnValue(view);
    render(<SystemPanel active={false} />);
    expect(useTelemetry).toHaveBeenCalledWith(false);
  });

  it("exposes the sample count, so a test can prove sampling stopped", () => {
    useTelemetry.mockReturnValue(view);
    render(<SystemPanel active />);
    expect(screen.getByTestId("system-panel-body")).toHaveAttribute("data-tick", "7");
  });
});
