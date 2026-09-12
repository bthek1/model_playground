import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type {
  CpuSample,
  GpuMemorySample,
  MemorySample,
  Metric,
  StorageSample,
} from "@/telemetry/types";
import type { WebGPUCapabilities } from "@/webgpu/types";

import { CpuCard } from "./CpuCard";
import { GpuCard } from "./GpuCard";
import { MemoryCard } from "./MemoryCard";
import { StorageCard } from "./StorageCard";

const caps: WebGPUCapabilities = {
  status: "ready",
  adapter: {
    vendor: "nvidia",
    architecture: "ampere",
    device: "",
    description: "NVIDIA GeForce RTX 3080",
  },
  isFallbackAdapter: false,
  features: [],
  limits: {},
};

const gpuMemory: Metric<GpuMemorySample> = {
  status: "ok",
  value: { liveBytes: 4_194_304, peakBytes: 8_388_608, buffers: 3, lastPassMs: 0.42 },
};

const memory: Metric<MemorySample> = {
  status: "ok",
  value: { usedBytes: 12_582_912, totalBytes: 25_165_824, limitBytes: 4_294_967_296 },
};

const cpu: CpuSample = {
  lagMs: 8.4,
  longTasks: 2,
  longTaskMs: 210,
  busyFraction: 0.35,
};

const storage: Metric<StorageSample> = {
  status: "ok",
  value: { usageBytes: 524_288_000, quotaBytes: 10_737_418_240, cachedModels: 4 },
};

describe("GpuCard", () => {
  it("shows tracked bytes, the adapter and the last pass", () => {
    render(<GpuCard identity={{ status: "ok", value: caps }} memory={gpuMemory} history={[1, 2, 3]} />);

    expect(screen.getByText("4 MiB")).toBeInTheDocument();
    expect(screen.getByText("NVIDIA GeForce RTX 3080")).toBeInTheDocument();
    expect(screen.getByText("8 MiB")).toBeInTheDocument();
    expect(screen.getByText("0.42 ms")).toBeInTheDocument();
  });

  it("says a pass was never timed rather than showing zero", () => {
    render(
      <GpuCard
        identity={{ status: "ok", value: caps }}
        memory={{ ...gpuMemory, value: { ...gpuMemory.value as GpuMemorySample, lastPassMs: null } }}
        history={[]}
      />,
    );
    expect(screen.getByText("not timed")).toBeInTheDocument();
  });

  it("flags a software adapter — it is not the machine's GPU", () => {
    render(
      <GpuCard
        identity={{ status: "ok", value: { ...caps, isFallbackAdapter: true } }}
        memory={gpuMemory}
        history={[]}
      />,
    );
    expect(screen.getByText("Software fallback")).toBeInTheDocument();
  });

  it("renders the reason, and no chart, where there is no GPU", () => {
    render(
      <GpuCard
        identity={{ status: "unavailable", reason: "This browser doesn't expose WebGPU." }}
        memory={null}
        history={[1, 2, 3]}
      />,
    );
    expect(screen.getByTestId("telemetry-gpu-unavailable")).toHaveTextContent(
      /doesn't expose WebGPU/,
    );
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });
});

describe("MemoryCard", () => {
  it("shows the heap against its ceiling", () => {
    render(
      <MemoryCard
        memory={memory}
        deviceMemoryGiB={{ status: "ok", value: 8 }}
        history={[1, 2]}
      />,
    );
    expect(screen.getByText("12 MiB")).toBeInTheDocument();
    expect(screen.getByText("/ 4 GiB")).toBeInTheDocument();
    expect(screen.getByText("~8 GiB")).toBeInTheDocument();
  });

  it("names the page heap in its note — the workers hold the weights", () => {
    render(
      <MemoryCard memory={memory} deviceMemoryGiB={{ status: "ok", value: 8 }} history={[]} />,
    );
    expect(screen.getByText(/Web Workers, whose heaps no API exposes/)).toBeInTheDocument();
  });

  it("renders the reason, and no chart, in a browser without performance.memory", () => {
    render(
      <MemoryCard
        memory={{ status: "unavailable", reason: "Only Chrome and Edge expose a JS heap size." }}
        deviceMemoryGiB={{ status: "unavailable", reason: "no" }}
        history={[1, 2, 3]}
      />,
    );
    expect(screen.getByTestId("telemetry-memory-unavailable")).toHaveTextContent(
      /Only Chrome and Edge/,
    );
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.queryByRole("meter")).not.toBeInTheDocument();
  });
});

describe("CpuCard", () => {
  it("shows activity, lag, long tasks and cores", () => {
    render(
      <CpuCard
        cpu={cpu}
        cores={{ status: "ok", value: 16 }}
        longTaskReason={null}
        lagHistory={[1, 2]}
        busyHistory={[0.2, 0.35]}
      />,
    );
    expect(screen.getByText("35%")).toBeInTheDocument();
    expect(screen.getByText("8 ms")).toBeInTheDocument();
    expect(screen.getByText("2 (210 ms)")).toBeInTheDocument();
    expect(screen.getByText("16")).toBeInTheDocument();
  });

  it("says long tasks are unavailable rather than reporting none", () => {
    render(
      <CpuCard
        cpu={{ ...cpu, longTasks: null, longTaskMs: null }}
        cores={{ status: "ok", value: 4 }}
        longTaskReason="Long-task timing isn't implemented in this browser — Chromium only."
        lagHistory={[]}
        busyHistory={[]}
      />,
    );
    expect(screen.getByText("unavailable")).toBeInTheDocument();
    expect(screen.queryByText("0")).not.toBeInTheDocument();
  });

  it("disclaims host CPU utilisation, which is not measurable", () => {
    render(
      <CpuCard
        cpu={cpu}
        cores={{ status: "unavailable", reason: "no" }}
        longTaskReason={null}
        lagHistory={[]}
        busyHistory={[]}
      />,
    );
    expect(
      screen.getByText(/No browser reports host CPU utilisation/),
    ).toBeInTheDocument();
  });
});

describe("StorageCard", () => {
  it("shows usage against the quota and the cached model count", () => {
    render(<StorageCard storage={storage} download={null} history={[]} />);
    expect(screen.getByText("500 MiB")).toBeInTheDocument();
    expect(screen.getByText("/ 10 GiB")).toBeInTheDocument();
    expect(screen.getByText("4")).toBeInTheDocument();
  });

  it("shows no download block when nothing is downloading", () => {
    render(<StorageCard storage={storage} download={null} history={[]} />);
    expect(screen.queryByTestId("telemetry-download")).not.toBeInTheDocument();
  });

  it("shows bytes and rate while a download is in flight", () => {
    render(
      <StorageCard
        storage={storage}
        download={{ loadedBytes: 52_428_800, totalBytes: 104_857_600, bytesPerSecond: 8_388_608 }}
        history={[1, 2]}
      />,
    );
    expect(screen.getByTestId("telemetry-download")).toBeInTheDocument();
    expect(screen.getByText("50 MiB / 100 MiB")).toBeInTheDocument();
    expect(screen.getByText("8 MiB/s")).toBeInTheDocument();
  });

  it("says it is still measuring before it can difference two samples", () => {
    render(
      <StorageCard
        storage={storage}
        download={{ loadedBytes: 1, totalBytes: 100, bytesPerSecond: null }}
        history={[]}
      />,
    );
    expect(screen.getByText("measuring…")).toBeInTheDocument();
  });

  it("renders the reason where the estimate is refused", () => {
    render(
      <StorageCard
        storage={{ status: "unavailable", reason: "private browsing blocks it" }}
        download={null}
        history={[]}
      />,
    );
    expect(screen.getByTestId("telemetry-storage-unavailable")).toHaveTextContent(
      /private browsing/,
    );
  });
});

// The scaling rules, which are the only thing in a sparkline that can be wrong
// in a way a reader would believe. Asserted through the cards because that is
// where the choice of scale is made.
describe("Sparkline scaling", () => {
  /** The y coordinates of the named chart's line. Up on screen is smaller. */
  function chartYs(label: RegExp) {
    // By label, not by tag: every card header renders a Lucide icon, which is
    // also an <svg>, and `querySelector("svg")` finds that one first.
    const chart = screen.getByLabelText(label);
    return (chart.querySelector("polyline")?.getAttribute("points") ?? "")
      .split(" ")
      .map((pair) => Number(pair.split(",")[1]));
  }

  it("does not peg a near-constant series to the top of its box", () => {
    // Three samples of 0.13 ms of frame lag: a real machine doing nothing.
    render(
      <CpuCard
        cpu={{ ...cpu, lagMs: 0.13 }}
        cores={{ status: "ok", value: 8 }}
        longTaskReason={null}
        lagHistory={[0.13, 0.13, 0.13]}
        busyHistory={[]}
      />,
    );
    const ys = chartYs(/Worst frame overrun/);
    expect(Math.min(...ys)).toBeGreaterThan(3);
  });

  it("puts an idle activity series on the floor of its fixed axis", () => {
    render(
      <CpuCard
        cpu={{ ...cpu, busyFraction: 0 }}
        cores={{ status: "ok", value: 8 }}
        longTaskReason={null}
        lagHistory={[]}
        busyHistory={[0, 0, 0]}
      />,
    );
    const ys = chartYs(/Inference activity/);
    expect(Math.max(...ys)).toBeGreaterThan(20);
  });

  it("shows a rising heap as rising, against its own range", () => {
    render(
      <MemoryCard
        memory={memory}
        deviceMemoryGiB={{ status: "ok", value: 8 }}
        history={[10_000_000, 11_000_000, 12_000_000]}
      />,
    );
    const ys = chartYs(/Page heap over the last two minutes/);
    // Up on screen is a smaller y.
    expect(ys[0]).toBeGreaterThan(ys[ys.length - 1]);
  });

  it("renders a placeholder, not a line, from a single sample", () => {
    const { container } = render(
      <MemoryCard
        memory={memory}
        deviceMemoryGiB={{ status: "unavailable", reason: "no" }}
        history={[10_000_000]}
      />,
    );
    expect(container.querySelector("polyline")).toBeNull();
    expect(
      screen.getByLabelText(/not enough samples yet/),
    ).toBeInTheDocument();
  });
});
