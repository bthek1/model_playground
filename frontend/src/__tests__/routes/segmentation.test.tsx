import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { UseSegmenterResult } from "@/hooks/useSegmenter";

const fakeImage = { width: 2, height: 2, channels: 3, data: [] } as never;
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

const MASKS = [
  {
    label: "sky",
    score: 0.9,
    mask: { data: new Uint8ClampedArray([255, 255, 255, 0]), width: 2, height: 2 },
  },
  {
    label: "tree",
    score: 0.6,
    mask: { data: new Uint8ClampedArray([0, 0, 0, 255]), width: 2, height: 2 },
  },
];

const mockRun = vi.fn().mockResolvedValue(MASKS);
const baseState: UseSegmenterResult = {
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
let mockState: UseSegmenterResult = { ...baseState };
const useSegmenter = vi.fn(() => mockState);

vi.mock("@/hooks/useSegmenter", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    useSegmenter: (...a: unknown[]) => useSegmenter(...(a as [])),
  };
});

const { Route } = await import("@/routes/segmentation");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Segmentation route component not found");
  render(<Page />);
}

const RUN_BUTTON = /^segment$/i;

// The RUN trigger. Picking an input never starts an inference any more
// (model-page-pattern.md §1.6), so a test that wants a result asks for one.
async function pickAndRun(sample: RegExp) {
  fireEvent.click(screen.getByRole("button", { name: sample }));
  await waitFor(() =>
    expect(screen.getByRole("button", { name: RUN_BUTTON })).toBeEnabled(),
  );
  fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));
}

const ready = (extra: Partial<UseSegmenterResult> = {}) => ({
  ...baseState,
  status: "ready" as const,
  idle: false,
  ready: true,
  ...extra,
});

describe("SegmentationPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockState = { ...baseState };
    mockRun.mockResolvedValue(MASKS);
    URL.createObjectURL = vi.fn(() => "blob:preview");
    URL.revokeObjectURL = vi.fn();
  });

  it("renders the heading and every model option", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /image segmentation/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /segformer-b0/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /face parsing/i })).toBeInTheDocument();
  });

  it("downloads nothing on arrival, and loads only on request", () => {
    renderPage();
    expect(useSegmenter).toHaveBeenCalledWith("Xenova/segformer-b0-finetuned-ade-512-512");
    expect(baseState.load).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /load model/i }));
    expect(baseState.load).toHaveBeenCalledOnce();
  });

  it("renders the four slots with an empty OUTPUT before any run", () => {
    renderPage();
    expect(screen.getAllByRole("region")).toHaveLength(4);
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("keeps the run control disabled until a model is ready", () => {
    renderPage();
    expect(screen.getByRole("button", { name: /^segment$/i })).toBeDisabled();
  });

  it("states the class space before the run, and which kind of segmentation", () => {
    // A segmenter can only ever say what its training set contained, and
    // "segmentation" covers three different tasks.
    renderPage();
    const note = screen.getByTestId("class-space");
    expect(note).toHaveTextContent(/semantic/i);
    expect(note).toHaveTextContent(/150 ADE20K/i);

    fireEvent.click(screen.getByRole("button", { name: /detr panoptic/i }));
    expect(screen.getByTestId("class-space")).toHaveTextContent(/panoptic/i);
  });

  it("composites one canvas from every mask, with a legend", async () => {
    mockState = ready({ result: MASKS });
    renderPage();
    await pickAndRun(/^city street$/i);

    await waitFor(() =>
      expect(screen.getByTestId("segmentation-canvas")).toBeInTheDocument(),
    );
    // One canvas, not one per class — the failure mode `drawMasks` exists to
    // prevent.
    expect(screen.getAllByTestId("segmentation-canvas")).toHaveLength(1);
    // Scoped to OUTPUT: the sample row carries a "City street" button, which a
    // loose /tree/i would happily match.
    const legend = within(screen.getByTestId("slot-4"));
    expect(legend.getByRole("button", { name: /sky/i })).toBeInTheDocument();
    expect(legend.getByRole("button", { name: /tree/i })).toBeInTheDocument();
    expect(screen.getByText(/2 of 2 classes/)).toBeInTheDocument();
  });

  it("hides a class without re-running the model", async () => {
    mockState = ready({ result: MASKS });
    renderPage();
    await pickAndRun(/^city street$/i);
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));

    const legend = within(screen.getByTestId("slot-4"));
    const sky = legend.getByRole("button", { name: /sky/i });
    expect(sky).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(sky);

    expect(legend.getByRole("button", { name: /sky/i })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
    expect(screen.getByText(/1 of 2 classes/)).toBeInTheDocument();
    expect(mockRun).toHaveBeenCalledTimes(1);
  });

  it("re-composites on an opacity change without re-running the model", async () => {
    mockState = ready({ result: MASKS });
    renderPage();
    await pickAndRun(/^city street$/i);
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));

    fireEvent.change(screen.getByLabelText(/overlay opacity/i), {
      target: { value: "0.9" },
    });
    expect(screen.getByLabelText(/overlay opacity/i)).toHaveValue("0.9");
    expect(mockRun).toHaveBeenCalledTimes(1);
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
