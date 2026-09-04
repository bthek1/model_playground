import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { WebGPUCapabilities } from "@/webgpu/types";

import { DeviceStatus } from "./DeviceStatus";

/** A capabilities object in whatever state the test needs. */
function caps(over: Partial<WebGPUCapabilities> = {}): WebGPUCapabilities {
  return {
    status: "ready",
    adapter: {
      vendor: "amd",
      architecture: "rdna-3",
      device: "",
      description: "",
    },
    isFallbackAdapter: false,
    features: [],
    limits: {},
    ...over,
  };
}

// The LOAD band for pages that probe a device instead of downloading weights.
// Every non-ready status has to say *why*, because "no WebGPU" is nearly always
// a fixable local problem (insecure origin, a Firefox flag) rather than a dead
// end — see webgpu-inference.md.
describe("DeviceStatus", () => {
  it("shows a probing state while the capability check is in flight", () => {
    render(<DeviceStatus capabilities={null} loading />);
    expect(screen.getByText(/probing the gpu/i)).toBeInTheDocument();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("names the adapter once a device is acquired", () => {
    render(<DeviceStatus capabilities={caps()} loading={false} />);
    const ready = screen.getByTestId("device-ready");
    expect(ready).toHaveTextContent(/GPU ready/);
    expect(ready).toHaveTextContent(/amd rdna-3/);
  });

  it("falls back to the adapter description when vendor/architecture are blank", () => {
    render(
      <DeviceStatus
        capabilities={caps({
          adapter: {
            vendor: "",
            architecture: "",
            device: "",
            description: "Software renderer",
          },
        })}
        loading={false}
      />,
    );
    expect(screen.getByTestId("device-ready")).toHaveTextContent(
      "Software renderer",
    );
  });

  it("flags a software fallback adapter — the timings would otherwise mislead", () => {
    render(
      <DeviceStatus
        capabilities={caps({ isFallbackAdapter: true })}
        loading={false}
      />,
    );
    expect(screen.getByTestId("device-ready")).toHaveTextContent(
      /software fallback/i,
    );
  });

  it.each([
    ["unsupported", /secure context/i],
    ["no-adapter", /no GPU adapter/i],
    ["no-device", /requesting a device failed/i],
  ] as const)("explains the %s status specifically", (status, expected) => {
    render(
      <DeviceStatus capabilities={caps({ status })} loading={false} />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent(expected);
    expect(screen.queryByTestId("device-ready")).toBeNull();
  });

  it("treats a missing probe result as unsupported rather than rendering blank", () => {
    render(<DeviceStatus capabilities={null} loading={false} />);
    expect(screen.getByRole("alert")).toHaveTextContent(/secure context/i);
  });
});
