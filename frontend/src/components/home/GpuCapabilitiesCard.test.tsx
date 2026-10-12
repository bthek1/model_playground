import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { WebGPUCapabilities } from "@/webgpu/types";

const webgpu = vi.hoisted(() => ({
  state: { capabilities: null as WebGPUCapabilities | null, loading: true },
}));

vi.mock("@/hooks/useWebGPU", () => ({ useWebGPU: () => webgpu.state }));

const { GpuCapabilitiesCard } = await import("./GpuCapabilitiesCard");

const READY: WebGPUCapabilities = {
  status: "ready",
  adapter: { vendor: "nvidia", architecture: "ampere" },
  features: ["shader-f16", "timestamp-query"],
  limits: { maxBufferSize: 268435456, maxComputeInvocationsPerWorkgroup: 256 },
  isFallbackAdapter: false,
} as unknown as WebGPUCapabilities;

describe("GpuCapabilitiesCard", () => {
  beforeEach(() => {
    webgpu.state = { capabilities: null, loading: true };
  });

  // Detection resolves after first paint, so the checking state is what the
  // answer replaces. It has the answer's shape — the browser row and a status
  // box with a title and a hint — or the card grows and everything below it
  // shifts (#67).
  it("holds the answer's shape while it checks", () => {
    render(<GpuCapabilitiesCard />);
    const checking = screen.getByTestId("gpu-checking");
    expect(checking).toHaveTextContent("Browser");
    expect(checking).toHaveTextContent("Checking GPU access…");
    expect(checking.querySelectorAll("p")).toHaveLength(2);
  });

  it("states a missing adapter in place of the checking box", () => {
    webgpu.state = { capabilities: { status: "no-adapter" } as WebGPUCapabilities, loading: false };
    render(<GpuCapabilitiesCard />);
    expect(screen.queryByTestId("gpu-checking")).toBeNull();
    expect(screen.getByText("No GPU access")).toBeInTheDocument();
    expect(screen.getByText(/no adapter was offered/)).toBeInTheDocument();
  });

  // The limits table is ~300 px tall. Rendered open, it pushed the home page
  // down after load on every machine with a GPU — the case a headless audit
  // never sees.
  it("keeps a ready adapter's details behind a closed disclosure", () => {
    webgpu.state = { capabilities: READY, loading: false };
    render(<GpuCapabilitiesCard />);
    expect(screen.getByText("GPU accessible")).toBeInTheDocument();
    const details = screen.getByTestId("gpu-details");
    expect(details.tagName).toBe("DETAILS");
    expect(details).not.toHaveAttribute("open");
    expect(details).toHaveTextContent("Adapter, limits and features");
    expect(details).toHaveTextContent("shader-f16");
  });
});
