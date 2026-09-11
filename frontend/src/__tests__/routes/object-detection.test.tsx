import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { UseObjectDetectorResult } from "@/hooks/useObjectDetector";

const fakeImage = { width: 8, height: 8, channels: 3, data: [] } as never;
const fromFile = vi.fn().mockResolvedValue(fakeImage);
const fromUrl = vi.fn().mockResolvedValue(fakeImage);
const downscale = vi.fn(async (img: unknown) => img);
const fromVideo = vi.fn(() => fakeImage);
const stopCamera = vi.fn();
const openCamera = vi.fn().mockResolvedValue(stopCamera);
vi.mock("@/vision/image", () => ({
  fromFile: (...a: unknown[]) => fromFile(...a),
  fromUrl: (...a: unknown[]) => fromUrl(...a),
  downscale: (...a: unknown[]) => downscale(...(a as [never])),
  fromVideo: (...a: unknown[]) => fromVideo(...(a as [])),
  openCamera: (...a: unknown[]) => openCamera(...a),
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

const DETECTIONS = [
  { label: "person", score: 0.91, box: { xmin: 0, ymin: 0, xmax: 4, ymax: 8 } },
  { label: "car", score: 0.42, box: { xmin: 4, ymin: 2, xmax: 8, ymax: 6 } },
  { label: "kite", score: 0.09, box: { xmin: 1, ymin: 1, xmax: 2, ymax: 2 } },
];

const mockRun = vi.fn().mockResolvedValue(DETECTIONS);
const baseState: UseObjectDetectorResult = {
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
let mockState: UseObjectDetectorResult = { ...baseState };
const useObjectDetector = vi.fn(() => mockState);

vi.mock("@/hooks/useObjectDetector", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    useObjectDetector: (...a: unknown[]) => useObjectDetector(...(a as [])),
  };
});

const { Route } = await import("@/routes/object-detection");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Object detection route component not found");
  return render(<Page />);
}

const RUN_BUTTON = /^detect$/i;

// The RUN trigger. Picking an input never starts an inference any more
// (model-page-pattern.md §1.6), so a test that wants a result asks for one.
async function pickAndRun(sample: RegExp) {
  fireEvent.click(screen.getByRole("button", { name: sample }));
  await waitFor(() =>
    expect(screen.getByRole("button", { name: RUN_BUTTON })).toBeEnabled(),
  );
  fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));
}

const ready = (extra: Partial<UseObjectDetectorResult> = {}) => ({
  ...baseState,
  status: "ready" as const,
  idle: false,
  ready: true,
  ...extra,
});

/** Hand-driven rAF, so a test can pump exactly N frames. */
let ticks: FrameRequestCallback[] = [];

beforeEach(() => {
  vi.clearAllMocks();
  mockState = { ...baseState };
  mockRun.mockResolvedValue(DETECTIONS);
  openCamera.mockResolvedValue(stopCamera);
  URL.createObjectURL = vi.fn(() => "blob:preview");
  URL.revokeObjectURL = vi.fn();
  ticks = [];
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => {
    ticks.push(cb);
    return ticks.length;
  });
  vi.stubGlobal("cancelAnimationFrame", () => {});
  let clock = 0;
  vi.stubGlobal("performance", { now: () => (clock += 25) });
});
afterEach(() => vi.unstubAllGlobals());

/** Give the route's <video> the dimensions a real playing stream would have. */
function playing(container: HTMLElement) {
  const video = container.querySelector("video")!;
  Object.defineProperty(video, "videoWidth", { value: 640, configurable: true });
  Object.defineProperty(video, "videoHeight", { value: 480, configurable: true });
  return video;
}

async function pump(n: number) {
  for (let i = 0; i < n; i++) {
    const next = ticks.shift();
    if (!next) return;
    await act(async () => {
      next(performance.now());
    });
  }
}

describe("ObjectDetectionPage", () => {
  it("renders the heading and every model option", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /object detection/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /d-fine nano/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^DETR r50/ })).toBeInTheDocument();
  });

  it("downloads nothing on arrival, and loads only on request", () => {
    renderPage();
    expect(useObjectDetector).toHaveBeenCalledWith("onnx-community/dfine_n_coco-ONNX");
    expect(baseState.load).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /load model/i }));
    expect(baseState.load).toHaveBeenCalledOnce();
  });

  it("renders the four slots with an empty OUTPUT before any run", () => {
    renderPage();
    expect(screen.getAllByRole("region")).toHaveLength(4);
    expect(screen.getByTestId("slot-3")).toBeInTheDocument();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("keeps the run control disabled until a model is ready", () => {
    renderPage();
    expect(screen.getByRole("button", { name: /^detect$/i })).toBeDisabled();
  });

  it("picks a sample without running, then detects when asked, on a capped frame", async () => {
    mockState = ready();
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: /^city street$/i }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: RUN_BUTTON })).toBeEnabled(),
    );
    // Picking an input is not asking for an inference.
    expect(mockRun).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    // Resolution is the throttle: the source is capped before inference, and
    // the boxes are mapped back onto the original for display.
    expect(downscale).toHaveBeenCalledWith(fakeImage, 640);
  });

  it("re-filters the boxes as the threshold moves, without a second run", async () => {
    // The pure-derivation trick /vad uses for its threshold. A slider that
    // re-runs the model is a slider nobody drags.
    mockState = ready({ result: DETECTIONS });
    renderPage();
    await pickAndRun(/^city street$/i);
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));

    expect(screen.getByText("person")).toBeInTheDocument();
    expect(screen.getByText("car")).toBeInTheDocument();
    expect(screen.queryByText("kite")).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(/confidence threshold/i), {
      target: { value: "0.05" },
    });
    expect(screen.getByText("kite")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(/confidence threshold/i), {
      target: { value: "0.9" },
    });
    expect(screen.queryByText("car")).not.toBeInTheDocument();

    expect(mockRun).toHaveBeenCalledTimes(1);
  });

  it("lists every detection beside the canvas", async () => {
    // A wrong box is only legible next to its label.
    mockState = ready({ result: DETECTIONS });
    renderPage();
    await pickAndRun(/^city street$/i);

    await waitFor(() =>
      expect(screen.getByTestId("detection-canvas")).toBeInTheDocument(),
    );
    expect(screen.getByText("0.91")).toBeInTheDocument();
    expect(screen.getByText(/2 of 3 above/)).toBeInTheDocument();
  });

  it("opens the camera only when asked, and closes it on unmount", async () => {
    // A leaked MediaStream leaves the webcam light on after the user has
    // navigated away.
    mockState = ready();
    const { unmount } = renderPage();
    expect(openCamera).not.toHaveBeenCalled();

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /use camera/i }));
    });
    await waitFor(() => expect(openCamera).toHaveBeenCalled());

    unmount();
    expect(stopCamera).toHaveBeenCalled();
  });

  it("never queues frames: one in flight however many ticks fire", async () => {
    let release: (() => void) | null = null;
    mockRun.mockImplementation(
      () => new Promise((resolve) => (release = () => resolve(DETECTIONS))),
    );
    mockState = ready();
    const { container } = renderPage();
    // happy-dom's <video> reports no intrinsic size, and the pump deliberately
    // skips a frame the stream has not sized yet.
    playing(container);

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /use camera/i }));
    });
    await pump(5);
    expect(mockRun).toHaveBeenCalledTimes(1);

    await act(async () => release?.());
    await pump(1);
    expect(mockRun).toHaveBeenCalledTimes(2);
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
    mockState = { ...baseState, status: "error", idle: false, error: "404 not found" };
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
  });
});
