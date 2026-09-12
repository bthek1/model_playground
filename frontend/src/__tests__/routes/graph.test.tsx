import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { GnnMetrics } from "@/webgpu/gnn";
import type { GraphSummary } from "@/webgpu/graphSession";

vi.mock("@tanstack/react-router", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    createFileRoute: vi
      .fn()
      .mockImplementation(
        (path: string) => (opts: Record<string, unknown>) => ({ path, options: opts }),
      ),
  };
});

vi.mock("@/components/charts/EChart", () => ({
  default: () => <div data-testid="echart" />,
}));

const gpuState = {
  supported: true,
  loading: false,
  capabilities: {
    status: "ready",
    adapter: { vendor: "test", architecture: "gpu", device: "", description: "" },
    isFallbackAdapter: false,
    features: [],
    limits: {},
  },
};
vi.mock("@/hooks/useWebGPU", () => ({ useWebGPU: () => gpuState }));

/** A tiny graph: 4 nodes, a path, with node 0 labelled. */
function summary(): GraphSummary {
  return {
    nNodes: 4,
    nFeat: 5,
    nClasses: 7,
    nEdges: 6,
    rowPtr: Uint32Array.from([0, 1, 3, 5, 6]),
    colIdx: Uint32Array.from([1, 0, 2, 1, 3, 2]),
    labels: Uint8Array.from([0, 1, 2, 3]),
    trainMask: Uint8Array.from([1, 0, 0, 0]),
    x: Float32Array.from([0, 0.3, 0.6, 1]),
    y: Float32Array.from([0, 0.5, 0.5, 1]),
    layoutMs: 912,
    backend: "webgpu",
  };
}

function metric(over: Partial<GnnMetrics> = {}): GnnMetrics {
  return {
    epoch: 41,
    totalEpochs: 200,
    loss: 0.42,
    trainAcc: 1,
    valAcc: 0.788,
    testAcc: 0.776,
    smoothness: 0.947,
    deadFraction: 0,
    ...over,
  };
}

const hookState = {
  status: "idle" as string,
  summary: null as GraphSummary | null,
  loadError: null as string | null,
  backend: null as string | null,
  load: vi.fn(),
  retry: vi.fn(),
  training: false,
  metrics: [] as GnnMetrics[],
  predictions: null as Uint8Array | null,
  elapsedMs: null as number | null,
  trainError: null as string | null,
  depthHistory: [] as unknown[],
  start: vi.fn(),
  stop: vi.fn(),
};
vi.mock("@/hooks/useGraphTraining", () => ({
  useGraphTraining: () => hookState,
}));

const { Route } = await import("@/routes/graph");
const GraphPage = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!GraphPage) throw new Error("Graph route component not found");
  render(<GraphPage />);
}

/** Put the page in the state it reaches after a successful load. */
function ready() {
  Object.assign(hookState, { status: "ready", summary: summary(), backend: "webgpu" });
}

describe("GraphPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    Object.assign(hookState, {
      status: "idle",
      summary: null,
      loadError: null,
      backend: null,
      training: false,
      metrics: [],
      predictions: null,
      elapsedMs: null,
      trainError: null,
      depthHistory: [],
    });
  });

  it("renders all four slots, with the output empty before any run", () => {
    renderPage();
    for (const step of [1, 2, 3, 4]) {
      expect(screen.getByTestId(`slot-${step}`)).toBeInTheDocument();
    }
    expect(screen.getByTestId("output-panel")).toBeInTheDocument();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("downloads nothing on mount — the graph loads only when asked", () => {
    renderPage();
    expect(hookState.load).not.toHaveBeenCalled();
    expect(hookState.start).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /load graph/i }));
    expect(hookState.load).toHaveBeenCalledTimes(1);
  });

  it("says the dataset is bundled, so the wait is the layout and not a download", () => {
    renderPage();
    expect(screen.getByText(/bundled with the app/i)).toBeInTheDocument();
  });

  it("disables Train until the graph is ready, while the controls stay live", () => {
    renderPage();
    expect(screen.getByRole("button", { name: /^train$/i })).toBeDisabled();
    // Choosing an architecture or a depth is free and commits to nothing.
    expect(screen.getByRole("button", { name: /GCN/ })).toBeEnabled();
    expect(screen.getByLabelText(/depth/i)).toBeEnabled();
  });

  it("changing the architecture or the depth runs nothing by itself", () => {
    ready();
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: /GraphSAGE/ }));
    fireEvent.change(screen.getByLabelText(/depth/i), { target: { value: "6" } });
    expect(hookState.start).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /^train$/i }));
    expect(hookState.start).toHaveBeenCalledTimes(1);
    expect(hookState.start).toHaveBeenCalledWith(
      expect.objectContaining({ arch: "sage", layers: 6 }),
    );
  });

  it("reports the loaded graph and which backend a run will use", () => {
    ready();
    renderPage();
    const note = screen.getByTestId("model-ready");
    expect(note).toHaveTextContent("4 nodes");
    expect(note).toHaveTextContent("3 edges"); // nEdges is directed; citations are half
    expect(note).toHaveTextContent("912 ms");
    expect(note).toHaveTextContent(/the GPU/);
  });

  it("draws nothing in OUTPUT until a run has produced an epoch", () => {
    ready();
    renderPage();
    expect(screen.queryByTestId("graph-canvas")).not.toBeInTheDocument();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("draws the graph and the scores once trained", () => {
    ready();
    Object.assign(hookState, { metrics: [metric()], elapsedMs: 1400 });
    renderPage();
    expect(screen.getByTestId("graph-canvas")).toBeInTheDocument();
    expect(screen.queryByTestId("output-empty")).not.toBeInTheDocument();
    expect(screen.getByText("Test accuracy")).toBeInTheDocument();
    expect(screen.getByText("77.6%")).toBeInTheDocument();
    expect(screen.getByText("0.947")).toBeInTheDocument();
  });

  it("switches the colouring without re-running the model", () => {
    ready();
    Object.assign(hookState, { metrics: [metric()] });
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: /^true$/i }));
    expect(hookState.start).not.toHaveBeenCalled();
  });

  it("calls GIN's overflow what it is, rather than oversmoothing", () => {
    ready();
    Object.assign(hookState, { metrics: [metric({ deadFraction: 1 })] });
    renderPage();
    expect(screen.getByTestId("collapse-note")).toHaveTextContent(
      /unnormalised sum overflowing/i,
    );
  });

  it("says nothing about overflow when the representations are alive", () => {
    ready();
    Object.assign(hookState, { metrics: [metric({ deadFraction: 0 })] });
    renderPage();
    expect(screen.queryByTestId("collapse-note")).not.toBeInTheDocument();
  });

  it("puts a load failure in the LOAD slot, with a retry", () => {
    Object.assign(hookState, { status: "error", loadError: "decode failed" });
    renderPage();
    const slot = screen.getByTestId("slot-2");
    expect(slot).toContainElement(screen.getByTestId("error-note"));
    expect(screen.getByText("decode failed")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /retry/i }));
    expect(hookState.retry).toHaveBeenCalledTimes(1);
  });

  it("puts a training failure in the RUN slot", () => {
    ready();
    Object.assign(hookState, { trainError: "device lost" });
    renderPage();
    expect(screen.getByTestId("slot-3")).toContainElement(
      screen.getByTestId("error-note"),
    );
  });

  it("disables Stop when nothing is running", () => {
    ready();
    renderPage();
    expect(screen.getByRole("button", { name: /stop/i })).toBeDisabled();
  });

  it("offers Stop while a run is in flight", () => {
    ready();
    Object.assign(hookState, { training: true });
    renderPage();
    expect(screen.getByRole("button", { name: /stop/i })).toBeEnabled();
  });

  it("says depth re-trains and the colour switch does not", () => {
    renderPage();
    expect(screen.getByText(/moving the slider re-trains/i)).toBeInTheDocument();
  });
});
