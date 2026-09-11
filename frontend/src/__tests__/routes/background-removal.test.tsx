import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { UseBackgroundRemovalResult } from "@/hooks/useBackgroundRemoval";
import { MATTE_MODELS } from "@/vision/backgroundRemoval";

// The image helpers reach for RawImage / canvas / getUserMedia, none of which
// exists under happy-dom. The route only passes what they return to `run`.
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

/** 2x1 RGBA: one fully covered pixel, one half-covered. A soft matte. */
const CUTOUT = {
  data: new Uint8ClampedArray([200, 100, 50, 255, 200, 100, 50, 128]),
  width: 2,
  height: 1,
  channels: 4,
};

const mockRun = vi.fn().mockResolvedValue(CUTOUT);
const baseState: UseBackgroundRemovalResult = {
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
  meta: MATTE_MODELS[0],
  run: mockRun,
  load: vi.fn(),
  retry: vi.fn(),
  cancel: vi.fn(),
};
let mockState: UseBackgroundRemovalResult = { ...baseState };
const useBackgroundRemoval = vi.fn(() => mockState);

vi.mock("@/hooks/useBackgroundRemoval", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    useBackgroundRemoval: (...a: unknown[]) =>
      useBackgroundRemoval(...(a as [])),
  };
});

const { Route } = await import("@/routes/background-removal");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Background removal route component not found");
  render(<Page />);
}

const RUN_BUTTON = /remove background/i;

const ready = (extra: Partial<UseBackgroundRemovalResult> = {}) => ({
  ...baseState,
  status: "ready" as const,
  idle: false,
  ready: true,
  ...extra,
});

describe("BackgroundRemovalPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockState = { ...baseState };
    mockRun.mockResolvedValue(CUTOUT);
    URL.createObjectURL = vi.fn(() => "blob:preview");
    URL.revokeObjectURL = vi.fn();
  });

  it("renders the heading and every model option", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /background removal/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /modnet/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /rmbg-1\.4/i })).toBeInTheDocument();
  });

  it("downloads nothing on arrival, and loads only on request", () => {
    renderPage();
    expect(useBackgroundRemoval).toHaveBeenCalledWith("Xenova/modnet");
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
      screen.getByRole("button", { name: /remove background/i }),
    ).toBeDisabled();
  });

  it("states the licence beside the model choice, in the SELECT slot", () => {
    // The default is permissive and says so quietly.
    renderPage();
    const note = screen.getByTestId("model-licence");
    expect(screen.getByTestId("slot-1")).toContainElement(note);
    expect(note).toHaveTextContent(/apache-2\.0/i);
    expect(note).toHaveTextContent(/commercial use permitted/i);
  });

  it("warns when the selected model is non-commercial", () => {
    // RMBG-1.4 is the better matte and is CC non-commercial. Shipping it as a
    // silent option is the trap this line exists to close.
    mockState = { ...baseState, meta: MATTE_MODELS[1] };
    renderPage();
    const note = screen.getByTestId("model-licence");
    expect(note).toHaveTextContent(/non-commercial/i);
    expect(note).toHaveTextContent(/bria/i);
  });

  it("picks a sample without running, then cuts out when asked", async () => {
    mockState = ready();
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: /^tiger$/i }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: RUN_BUTTON })).toBeEnabled(),
    );
    // Picking an input is not asking for an inference.
    expect(mockRun).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    expect(downscale).toHaveBeenCalledWith(fakeImage, 1024);
  });

  it("shows a picked image without running anything when no model is loaded", async () => {
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /^tiger$/i }));

    await waitFor(() =>
      expect(screen.getByAltText(/selected input: tiger/i)).toBeInTheDocument(),
    );
    expect(mockRun).not.toHaveBeenCalled();
  });

  it("offers the raw matte as its own view, and says the edge is soft", () => {
    mockState = ready({ result: CUTOUT });
    renderPage();

    expect(screen.getByTestId("cutout-view")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /^matte$/i }));
    expect(screen.getByTestId("matte-view")).toBeInTheDocument();

    const caveat = screen.getByText(/in-between value rather than being rounded/i);
    expect(screen.getByTestId("slot-4")).toContainElement(caveat);
  });

  it("swaps the background by re-deriving, never by re-running the model", () => {
    // The matte is already in the alpha channel, so a different backdrop is
    // arithmetic over pixels we have. Same rule /vad applies to its threshold.
    mockState = ready({ result: CUTOUT });
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: /studio green/i }));
    expect(mockRun).not.toHaveBeenCalled();
    expect(screen.getByTestId("cutout-view")).toBeInTheDocument();
  });

  it("reports how much of the frame is subject", () => {
    // A degenerate all-on or all-off matte is what a broken preprocessing path
    // produces, and both look clean. This is the number that gives it away.
    mockState = ready({ result: CUTOUT });
    renderPage();
    expect(screen.getByTestId("matte-coverage")).toHaveTextContent(/% of/);
  });

  it("offers a PNG download, the only common format that keeps alpha", () => {
    mockState = ready({ result: CUTOUT });
    renderPage();
    expect(screen.getByTestId("download-png")).toHaveAttribute(
      "download",
      "cutout.png",
    );
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
