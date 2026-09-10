import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { UseSuperResResult } from "@/hooks/useSuperRes";
import { SUPER_RES_MODELS } from "@/vision/superRes";

// The image helpers reach for RawImage / canvas, neither of which exists under
// happy-dom. The route only passes what they return to `run`.
const fakeImage = { width: 300, height: 220, channels: 3, data: [] } as never;
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

// `upscale` goes through a real canvas. Return the source untouched — the route
// only needs *an* image of the right shape for the comparison to render.
vi.mock("@/vision/resample", () => ({
  upscale: (src: unknown) => src,
  resample: (src: unknown) => src,
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

const SOURCE = {
  data: new Uint8ClampedArray(300 * 220 * 3),
  width: 300,
  height: 220,
  channels: 3,
};
const RESULT = {
  data: new Uint8ClampedArray(600 * 440 * 3),
  width: 600,
  height: 440,
  channels: 3,
};

const mockRun = vi.fn().mockResolvedValue(RESULT);
const mockStop = vi.fn();
const baseState: UseSuperResResult = {
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
  source: null,
  tiles: null,
  meta: SUPER_RES_MODELS[0],
  run: mockRun,
  stop: mockStop,
  load: vi.fn(),
  retry: vi.fn(),
  cancel: vi.fn(),
};
let mockState: UseSuperResResult = { ...baseState };
const useSuperRes = vi.fn(() => mockState);

vi.mock("@/hooks/useSuperRes", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return { ...actual, useSuperRes: (...a: unknown[]) => useSuperRes(...(a as [])) };
});

const { Route } = await import("@/routes/super-resolution");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Super-resolution route component not found");
  render(<Page />);
}

const ready = (extra: Partial<UseSuperResResult> = {}) => ({
  ...baseState,
  status: "ready" as const,
  idle: false,
  ready: true,
  ...extra,
});

describe("SuperResolutionPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockState = { ...baseState };
    mockRun.mockResolvedValue(RESULT);
    URL.createObjectURL = vi.fn(() => "blob:preview");
    URL.revokeObjectURL = vi.fn();
  });

  it("renders the heading and the catalogue", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /super resolution/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /swin2sr/i })).toBeInTheDocument();
  });

  it("says it covers super-resolution only, not image-to-image editing", () => {
    // The taxonomy slug promises more than the model delivers, so the page has
    // to say what it actually does — in the header, not a tooltip.
    renderPage();
    expect(screen.getByText(/stay on a server/i)).toBeInTheDocument();
  });

  it("downloads nothing on arrival, and loads only on request", () => {
    renderPage();
    expect(useSuperRes).toHaveBeenCalledWith(
      "Xenova/swin2SR-classical-sr-x2-64",
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
    expect(screen.getByRole("button", { name: /upscale 2x/i })).toBeDisabled();
  });

  it("states the tile count and rough time before anything is upscaled", async () => {
    // Discovering that an upscale is four minutes long halfway through it is
    // the difference between a slow page and one that appears to have hung.
    mockState = ready();
    renderPage();
    expect(screen.queryByTestId("size-guard")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /^tiger$/i }));
    const guard = await screen.findByTestId("size-guard");

    expect(guard).toHaveTextContent(/600x440/);
    expect(guard).toHaveTextContent(/4 tiles/);
    expect(guard).toHaveTextContent(/min|s\b/);
    expect(mockRun).not.toHaveBeenCalled();
  });

  it("does not upscale until asked — picking an image is free", async () => {
    mockState = ready();
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /^tiger$/i }));
    await screen.findByTestId("size-guard");
    expect(mockRun).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /upscale 2x/i }));
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
  });

  it("caps the source before upscaling", async () => {
    mockState = ready();
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /^tiger$/i }));
    await waitFor(() => expect(downscale).toHaveBeenCalledWith(fakeImage, 512));
  });

  it("offers no Stop control when nothing is running", () => {
    mockState = ready();
    renderPage();
    expect(screen.queryByTestId("stop-run")).not.toBeInTheDocument();
    expect(screen.queryByTestId("tile-progress")).not.toBeInTheDocument();
  });

  it("offers Stop and per-tile progress while a run is in flight", () => {
    // A run is thirty inferences, not one. Without a way out, the only way to
    // abandon a four-minute upscale is to reload the page — which also throws
    // away the weights.
    mockState = ready({ running: true, tiles: { done: 3, total: 12 } });
    renderPage();

    expect(screen.getByTestId("tile-progress")).toHaveTextContent("tile 3 / 12");
    fireEvent.click(screen.getByTestId("stop-run"));
    expect(mockStop).toHaveBeenCalledOnce();
    // Stopping abandons the tiles, never the download.
    expect(baseState.cancel).not.toHaveBeenCalled();
  });

  it("shows the comparison against a bicubic baseline, not the result alone", () => {
    // A 2x image on its own proves nothing; the split against bicubic is what
    // shows the model doing work.
    mockState = ready({ result: RESULT, source: SOURCE });
    renderPage();

    expect(screen.getByTestId("sr-compare")).toBeInTheDocument();
    expect(screen.getByTestId("compare-handle")).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: /bicubic/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /swin2sr/i })).toBeInTheDocument();
  });

  it("offers the result as a PNG download", () => {
    mockState = ready({ result: RESULT, source: SOURCE });
    renderPage();
    expect(screen.getByTestId("download-png")).toHaveAttribute(
      "download",
      "upscaled.png",
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
