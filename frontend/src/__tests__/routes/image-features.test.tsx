import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { IndexEntry, UseImageFeaturesResult } from "@/hooks/useImageFeatures";
import { poolEmbedding } from "@/vision/features";
import { GALLERY_IMAGES } from "@/vision/gallery";

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

/** `[1, 3, 2]` — CLS then two patches, so both poolings exist and differ. */
const embedding = (cls: [number, number], p1: [number, number]) =>
  poolEmbedding({
    data: Float32Array.from([...cls, ...p1, ...p1]),
    dims: [1, 3, 2],
  });

const QUERY = embedding([1, 0], [0, 1]);
// `corgi` is CLS-close to the query and mean-far; `airport` is the reverse.
// That asymmetry is what lets a test tell the two poolings apart.
const INDEX: IndexEntry[] = [
  { image: GALLERY_IMAGES[0], embedding: embedding([1, 0.05], [0, 1]) },
  { image: GALLERY_IMAGES[5], embedding: embedding([0, 1], [0, 1]) },
];

const mockRun = vi.fn().mockResolvedValue(QUERY);
const buildIndex = vi.fn().mockResolvedValue(undefined);
const addToIndex = vi.fn().mockResolvedValue(undefined);
const clearIndex = vi.fn();

const baseState: UseImageFeaturesResult = {
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
  index: [],
  indexing: null,
  buildIndex,
  addToIndex,
  clearIndex,
};
let mockState: UseImageFeaturesResult = { ...baseState };
const useImageFeatures = vi.fn(() => mockState);

vi.mock("@/hooks/useImageFeatures", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    useImageFeatures: (...a: unknown[]) => useImageFeatures(...(a as [])),
  };
});

const { Route } = await import("@/routes/image-features");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Image features route component not found");
  return render(<Page />);
}

const ready = (extra: Partial<UseImageFeaturesResult> = {}) => ({
  ...baseState,
  status: "ready" as const,
  idle: false,
  ready: true,
  ...extra,
});

beforeEach(() => {
  vi.clearAllMocks();
  mockState = { ...baseState };
  mockRun.mockResolvedValue(QUERY);
  openCamera.mockResolvedValue(stopCamera);
  URL.createObjectURL = vi.fn(() => "blob:preview");
  URL.revokeObjectURL = vi.fn();
  vi.stubGlobal("requestAnimationFrame", () => 1);
  vi.stubGlobal("cancelAnimationFrame", () => {});
});
afterEach(() => vi.unstubAllGlobals());

describe("ImageFeaturesPage", () => {
  it("renders the heading and every model option", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /image feature extraction/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /dinov2 small/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /clip vit-b\/32/i }),
    ).toBeInTheDocument();
  });

  it("downloads nothing on arrival, and loads only on request", () => {
    renderPage();
    expect(useImageFeatures).toHaveBeenCalledWith("Xenova/dinov2-small", false);
    expect(baseState.load).not.toHaveBeenCalled();
    expect(buildIndex).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /load model/i }));
    expect(baseState.load).toHaveBeenCalledOnce();
  });

  it("renders the four slots with an empty OUTPUT before any run", () => {
    renderPage();
    expect(screen.getAllByRole("region")).toHaveLength(4);
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("keeps the run controls disabled until a model is ready", () => {
    renderPage();
    expect(screen.getByRole("button", { name: /^embed$/i })).toBeDisabled();
    expect(
      screen.getByRole("button", { name: /add to gallery/i }),
    ).toBeDisabled();
    expect(screen.getByRole("button", { name: /embed the \d+ bundled/i })).toBeDisabled();
  });

  it("embeds the gallery only when asked", async () => {
    mockState = ready();
    renderPage();
    fireEvent.click(
      screen.getByRole("button", { name: /embed the \d+ bundled/i }),
    );
    await waitFor(() => expect(buildIndex).toHaveBeenCalledWith(GALLERY_IMAGES));
  });

  it("embeds as soon as a sample is picked, on a capped frame", async () => {
    mockState = ready();
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /^tiger$/i }));
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    expect(downscale).toHaveBeenCalledWith(fakeImage, 640);
  });

  it("ranks the neighbours, and re-ranks on a pooling switch without a run", async () => {
    // The whole point of deriving both vectors from one pass: the toggle is
    // arithmetic over numbers already in hand.
    mockState = ready({ result: QUERY, index: INDEX });
    renderPage();

    const list = () =>
      within(screen.getByTestId("neighbours"))
        .getAllByRole("listitem")
        .map((li) => li.textContent ?? "");

    await waitFor(() => expect(screen.getByTestId("neighbours")).toBeVisible());
    expect(list()[0]).toMatch(/corgi/i);

    fireEvent.click(screen.getByRole("button", { name: /mean of patches/i }));
    // Under mean pooling both entries are identical to the query, so the tie
    // falls back to index order — and, critically, nothing re-ran.
    expect(screen.getByTestId("neighbours")).toBeVisible();
    expect(mockRun).not.toHaveBeenCalled();
  });

  it("re-ranks on a k change without a run", async () => {
    mockState = ready({ result: QUERY, index: INDEX });
    renderPage();
    await waitFor(() => expect(screen.getByTestId("neighbours")).toBeVisible());
    expect(
      within(screen.getByTestId("neighbours")).getAllByRole("listitem"),
    ).toHaveLength(2);

    // By role, not by label text: the OUTPUT band is itself a region labelled
    // "Nearest neighbours", so `getByLabelText` matches both it and the slider
    // inside it (model-page-pattern.md §8).
    fireEvent.change(screen.getByRole("slider", { name: /neighbours/i }), {
      target: { value: "1" },
    });
    expect(
      within(screen.getByTestId("neighbours")).getAllByRole("listitem"),
    ).toHaveLength(1);
    expect(mockRun).not.toHaveBeenCalled();
  });

  it("shows the norm before and after normalising", () => {
    // "The vectors are normalised" is a claim, and this is the page that can
    // simply show it.
    mockState = ready({ result: QUERY, index: INDEX });
    renderPage();
    const facts = screen.getByTestId("embedding-facts");
    expect(facts).toHaveTextContent(/2 dimensions/);
    expect(facts).toHaveTextContent(/1\.000/);
  });

  it("offers no pooling choice for a model that returns one projected vector", () => {
    // CLIP as a feature extractor: `image_embeds`, and nothing to choose.
    const pooled = poolEmbedding({
      data: Float32Array.from([0, 1]),
      dims: [1, 2],
    });
    mockState = ready({ result: pooled, index: [] });
    renderPage();
    expect(
      screen.getByRole("button", { name: /projected embedding/i }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /mean of patches/i }),
    ).not.toBeInTheDocument();
  });

  it("says so when a picture is embedded but nothing is indexed", () => {
    mockState = ready({ result: QUERY, index: [] });
    renderPage();
    expect(
      screen.getByText(/nothing to compare it with yet/i),
    ).toBeInTheDocument();
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
