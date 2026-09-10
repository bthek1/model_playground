import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { UseDepthResult } from "@/hooks/useDepth";

const fakeImage = { width: 4, height: 4, channels: 3, data: [] } as never;
const fromFile = vi.fn().mockResolvedValue(fakeImage);
const fromUrl = vi.fn().mockResolvedValue(fakeImage);
const downscale = vi.fn(async (img: unknown) => img);
vi.mock("@/vision/image", () => ({
  fromFile: (...a: unknown[]) => fromFile(...a),
  fromUrl: (...a: unknown[]) => fromUrl(...a),
  downscale: (...a: unknown[]) => downscale(...(a as [never])),
  fromVideo: vi.fn(),
  openCamera: vi.fn().mockResolvedValue(() => {}),
  toPayload: () => ({
    data: new Uint8ClampedArray(4 * 4 * 3),
    width: 4,
    height: 4,
    channels: 3,
  }),
}));

// The renderer needs a real GPUDevice and a `webgpu` canvas context, neither of
// which happy-dom has. Its own behaviour is not what a route test is for.
const supported = { value: true as boolean | null };
vi.mock("@/hooks/usePointCloudView", () => ({
  usePointCloudView: () => ({
    canvasRef: vi.fn(),
    supported: supported.value,
    camera: { yaw: 0, pitch: 0, distance: 3, pointSize: 1.5 },
    setCamera: vi.fn(),
    onPointerDown: vi.fn(),
    onWheel: vi.fn(),
    reset: vi.fn(),
  }),
}));

// `detectWebGPU()` never throws — it reports a status. The route has to render
// something sensible for each of them, which is exactly what this stubs.
const gpuStatus = { value: "ready" as string, loading: false };
vi.mock("@/hooks/useWebGPU", () => ({
  useWebGPU: () => ({
    capabilities: { status: gpuStatus.value, adapter: null, isFallbackAdapter: false, features: [], limits: {} },
    loading: gpuStatus.loading,
    supported: gpuStatus.value === "ready",
  }),
}));

vi.mock("@tanstack/react-router", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    createFileRoute: vi
      .fn()
      .mockImplementation(
        (path: string) => (opts: Record<string, unknown>) => ({
          path,
          options: opts,
        }),
      ),
  };
});

/** A 4x4 depth map with a real spread, so the cloud is non-degenerate. */
const DEPTH = {
  predicted_depth: {
    data: new Float32Array(Array.from({ length: 16 }, (_, i) => i / 15)),
    dims: [1, 4, 4],
  },
};

const mockRun = vi.fn().mockResolvedValue(DEPTH);
const baseState: UseDepthResult = {
  status: "idle",
  idle: true,
  loading: false,
  ready: false,
  progress: null,
  loadProgress: null,
  loadedInMs: null,
  backend: null,
  running: false,
  error: null,
  result: null,
  run: mockRun,
  load: vi.fn(),
  retry: vi.fn(),
  cancel: vi.fn(),
};
let mockState: UseDepthResult = { ...baseState };
const useDepth = vi.fn(() => mockState);

vi.mock("@/hooks/useDepth", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return { ...actual, useDepth: (...a: unknown[]) => useDepth(...(a as [])) };
});

const { Route } = await import("@/routes/image-to-3d");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Image-to-3D route component not found");
  render(<Page />);
}

const ready = (extra: Partial<UseDepthResult> = {}) => ({
  ...baseState,
  status: "ready" as const,
  idle: false,
  ready: true,
  ...extra,
});

describe("ImageTo3DPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockState = { ...baseState };
    mockRun.mockResolvedValue(DEPTH);
    supported.value = true;
    gpuStatus.value = "ready";
    gpuStatus.loading = false;
    URL.createObjectURL = vi.fn(() => "blob:preview");
    URL.revokeObjectURL = vi.fn();
  });

  it("renders the heading and the depth catalogue it borrows", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /image to 3d/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /depth anything v2 small/i }),
    ).toBeInTheDocument();
  });

  it("says full reconstruction stays on a server", () => {
    renderPage();
    expect(screen.getByText(/stays on a server/i)).toBeInTheDocument();
  });

  it("downloads nothing on arrival, and loads only on request", () => {
    renderPage();
    expect(useDepth).toHaveBeenCalledWith(
      "onnx-community/depth-anything-v2-small",
      false,
    );
    expect(baseState.load).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /load model/i }));
    expect(baseState.load).toHaveBeenCalledOnce();
  });

  it("renders the four slots with an empty OUTPUT before any run", () => {
    renderPage();
    expect(screen.getAllByRole("region")).toHaveLength(4);
    expect(screen.getByTestId("slot-1")).toBeInTheDocument();
    expect(screen.getByTestId("slot-4")).toBeInTheDocument();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("keeps the run control disabled until a model is ready", () => {
    renderPage();
    expect(
      screen.getByRole("button", { name: /build point cloud/i }),
    ).toBeDisabled();
  });

  it("says the focal length is an assumption, next to the control", () => {
    // The page's substantive claim: relative depth carries no intrinsics, so
    // the geometry is plausible rather than measured. A 3-D view without this
    // sentence teaches something false.
    renderPage();
    // Two assertions rather than one regex: the word is emphasised, so the
    // sentence is split across elements and no single matcher spans both.
    const emphasis = screen.getByText("assumption");
    const caveat = screen.getByText(/not a measurement/i);
    expect(screen.getByTestId("slot-3")).toContainElement(emphasis);
    expect(screen.getByTestId("slot-3")).toContainElement(caveat);
    expect(screen.getByTestId("focal-slider")).toBeInTheDocument();
  });

  it("builds a cloud as soon as a sample is picked", async () => {
    mockState = ready();
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /^tiger$/i }));
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
  });

  it("re-derives from the cached depth map when a slider moves — no re-run", async () => {
    // Both controls are pure derivations over a depth map already in hand.
    // Re-running the model on a slider drag is the failure mode; /vad settled
    // the same question for its threshold.
    mockState = ready({ result: DEPTH });
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /^tiger$/i }));
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));

    fireEvent.change(screen.getByTestId("focal-slider"), {
      target: { value: "1.4" },
    });
    fireEvent.click(screen.getByTestId("stride-4"));

    expect(mockRun).toHaveBeenCalledTimes(1);
  });

  it("changes the point count with the stride", async () => {
    mockState = ready({ result: DEPTH });
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /^tiger$/i }));
    await screen.findByTestId("cloud-canvas");

    // 4x4 at stride 2 is 2x2 = 4 points; at stride 1 it is 16.
    expect(screen.getByTestId("output-panel")).toHaveTextContent(/4 points/);
    fireEvent.click(screen.getByTestId("stride-1"));
    expect(screen.getByTestId("output-panel")).toHaveTextContent(/16 points/);
  });

  it("renders the depth map and an explanation when there is no GPU", async () => {
    // `detectWebGPU()` never throws, so the failure is a status — and a page
    // that answered it with an empty canvas would look like the model failed
    // rather than like the machine lacking a device.
    gpuStatus.value = "no-adapter";
    supported.value = false;
    mockState = ready({ result: DEPTH });
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /^tiger$/i }));

    await screen.findByTestId("cloud-fallback");
    expect(screen.queryByTestId("cloud-canvas")).not.toBeInTheDocument();
    expect(screen.getByTestId("depth-map")).toBeInTheDocument();
    expect(screen.getByText(/needs a WebGPU device/i)).toBeInTheDocument();
  });

  it("warns in the LOAD slot before a download when there is no GPU", () => {
    // Better before 50 MB than after it.
    gpuStatus.value = "unsupported";
    renderPage();
    const note = screen.getByTestId("webgpu-unavailable");
    expect(screen.getByTestId("slot-2")).toContainElement(note);
  });

  it("says nothing about WebGPU when a device is available", () => {
    renderPage();
    expect(screen.queryByTestId("webgpu-unavailable")).not.toBeInTheDocument();
  });

  it("surfaces a failed decode in the RUN slot, not the output", async () => {
    mockState = ready();
    fromUrl.mockRejectedValueOnce(new Error("Unsupported image type"));
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: /^tiger$/i }));
    const note = await screen.findByText(/unsupported image type/i);
    expect(screen.getByTestId("slot-3")).toContainElement(note);
    expect(mockRun).not.toHaveBeenCalled();
  });

  it("puts a load failure in the LOAD slot, not in OUTPUT", () => {
    mockState = {
      ...baseState,
      status: "error",
      idle: false,
      error: "404 not found",
    };
    renderPage();
    const note = screen.getByText(/404 not found/i);
    expect(screen.getByTestId("slot-2")).toContainElement(note);
    expect(screen.getByTestId("slot-4")).not.toContainElement(note);
  });

  it("puts an inference failure in OUTPUT, where the model stays loaded", () => {
    mockState = ready({ error: "Non-zero status code" });
    renderPage();
    const note = screen.getByText(/non-zero status code/i);
    expect(screen.getByTestId("slot-4")).toContainElement(note);
    expect(screen.getByTestId("slot-2")).not.toContainElement(note);
  });
});
