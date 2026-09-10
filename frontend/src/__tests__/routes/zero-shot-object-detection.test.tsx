import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { UseZeroShotDetectorResult } from "@/hooks/useZeroShotDetector";

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
  {
    label: "a person",
    score: 0.31,
    box: { xmin: 0, ymin: 0, xmax: 4, ymax: 8 },
  },
  { label: "a car", score: 0.12, box: { xmin: 4, ymin: 2, xmax: 8, ymax: 6 } },
  {
    label: "a traffic light",
    score: 0.03,
    box: { xmin: 1, ymin: 1, xmax: 2, ymax: 2 },
  },
];

const mockRun = vi.fn().mockResolvedValue(DETECTIONS);
const baseState: UseZeroShotDetectorResult = {
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
let mockState: UseZeroShotDetectorResult = { ...baseState };
const useZeroShotDetector = vi.fn(() => mockState);

vi.mock("@/hooks/useZeroShotDetector", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    useZeroShotDetector: (...a: unknown[]) => useZeroShotDetector(...(a as [])),
  };
});

const { Route } = await import("@/routes/zero-shot-object-detection");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Zero-shot detection route component not found");
  return render(<Page />);
}

const ready = (extra: Partial<UseZeroShotDetectorResult> = {}) => ({
  ...baseState,
  status: "ready" as const,
  idle: false,
  ready: true,
  ...extra,
});

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

describe("ZeroShotObjectDetectionPage", () => {
  it("renders the heading and every model option", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /zero-shot object detection/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /owlv2 base/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /grounding dino tiny/i }),
    ).toBeInTheDocument();
  });

  it("downloads nothing on arrival, and loads only on request", () => {
    renderPage();
    expect(useZeroShotDetector).toHaveBeenCalledWith(
      "Xenova/owlv2-base-patch16-ensemble",
      false,
    );
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

  it("lets the queries be written before anything is loaded", () => {
    renderPage();
    const run = within(screen.getByTestId("slot-3"));
    expect(run.getByText("a person")).toBeInTheDocument();

    fireEvent.click(run.getByRole("button", { name: /remove a car/i }));
    expect(run.queryByText("a car")).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(/^queries$/i), {
      target: { value: "a red umbrella" },
    });
    fireEvent.click(run.getByRole("button", { name: /^add$/i }));
    expect(run.getByText("a red umbrella")).toBeInTheDocument();
  });

  it("detects as soon as a sample is picked, on a capped frame", async () => {
    mockState = ready();
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: /^city street$/i }));
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    expect(downscale).toHaveBeenCalledWith(fakeImage, 640);
    expect(mockRun).toHaveBeenCalledWith(
      fakeImage,
      ["a person", "a car", "a traffic light"],
      { consume: false },
    );
  });

  it("re-filters as the threshold moves, and re-runs only when a query changes", async () => {
    mockState = ready({ result: DETECTIONS });
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /^city street$/i }));
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));

    const out = within(screen.getByTestId("query-groups"));
    // Default threshold is 0.1: the person and the car are in, the traffic
    // light is not.
    expect(out.getByText(/^0\.310$/)).toBeInTheDocument();
    expect(out.getByText(/^0\.120$/)).toBeInTheDocument();
    expect(out.queryByText(/^0\.030$/)).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(/confidence threshold/i), {
      target: { value: "0.02" },
    });
    expect(
      within(screen.getByTestId("query-groups")).getByText(/^0\.030$/),
    ).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(/confidence threshold/i), {
      target: { value: "0.5" },
    });
    expect(
      within(screen.getByTestId("query-groups")).queryByText(/^0\.310$/),
    ).not.toBeInTheDocument();

    // The whole point: none of that asked the model anything.
    expect(mockRun).toHaveBeenCalledTimes(1);
  });

  it("lists every query, including the ones that found nothing", async () => {
    mockState = ready({ result: DETECTIONS });
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /^city street$/i }));

    await waitFor(() =>
      expect(screen.getByTestId("query-groups")).toBeInTheDocument(),
    );
    const out = within(screen.getByTestId("query-groups"));
    expect(out.getByText("a traffic light")).toBeInTheDocument();
    expect(out.getByText(/nothing above the threshold/i)).toBeInTheDocument();
  });

  it("starts the threshold well below a closed-vocabulary detector's", () => {
    // An open-vocabulary detector's confident hits land near 0.1. A 0.4 default
    // shows an empty canvas on a picture full of correctly-found objects.
    renderPage();
    const slider = screen.getByLabelText(/confidence threshold/i);
    expect(Number((slider as HTMLInputElement).value)).toBeLessThanOrEqual(0.1);
  });

  it("opens the camera only when asked, and closes it on unmount", async () => {
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
  });
});
