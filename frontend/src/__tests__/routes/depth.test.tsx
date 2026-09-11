import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { UseDepthResult } from "@/hooks/useDepth";

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

const DEPTH = {
  predicted_depth: { data: new Float32Array([0, 1, 2, 3]), dims: [1, 2, 2] },
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

const { Route } = await import("@/routes/depth");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Depth route component not found");
  render(<Page />);
}

const RUN_BUTTON = /estimate depth/i;

// The RUN trigger. Picking an input never starts an inference any more
// (model-page-pattern.md §1.6), so a test that wants a result asks for one.
async function pickAndRun(sample: RegExp) {
  fireEvent.click(screen.getByRole("button", { name: sample }));
  await waitFor(() =>
    expect(screen.getByRole("button", { name: RUN_BUTTON })).toBeEnabled(),
  );
  fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));
}

const ready = (extra: Partial<UseDepthResult> = {}) => ({
  ...baseState,
  status: "ready" as const,
  idle: false,
  ready: true,
  ...extra,
});

describe("DepthPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockState = { ...baseState };
    mockRun.mockResolvedValue(DEPTH);
    URL.createObjectURL = vi.fn(() => "blob:preview");
    URL.revokeObjectURL = vi.fn();
  });

  it("renders the heading and every model option", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /depth estimation/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /depth anything v2 small/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /depth pro/i }),
    ).toBeInTheDocument();
  });

  it("downloads nothing on arrival, and loads only on request", () => {
    renderPage();
    expect(useDepth).toHaveBeenCalledWith("onnx-community/depth-anything-v2-small");
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
    expect(screen.getByRole("button", { name: /estimate depth/i })).toBeDisabled();
  });

  it("gates the gigabyte model behind an explicit notice, not just a size line", () => {
    // The picker already quotes the size. Depth Pro gets a second, blunter
    // statement because a gigabyte is a decision, not a detail — the same gate
    // /text-to-audio puts in front of MusicGen.
    renderPage();
    expect(screen.queryByTestId("heavy-model-notice")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /depth pro/i }));
    const notice = screen.getByTestId("heavy-model-notice");
    expect(screen.getByTestId("slot-2")).toContainElement(notice);
    expect(baseState.load).not.toHaveBeenCalled();
  });

  it("picks a sample without running, then estimates when asked", async () => {
    mockState = ready();
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: /^tiger$/i }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: RUN_BUTTON })).toBeEnabled(),
    );
    // Picking is input. A depth pass on every sample click is bandwidth and
    // GPU time the user never asked for.
    expect(mockRun).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    expect(downscale).toHaveBeenCalledWith(fakeImage, 640);
  });

  it("shows a picked image without running anything when no model is loaded", async () => {
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /^tiger$/i }));

    await waitFor(() =>
      expect(screen.getByAltText(/selected input: tiger/i)).toBeInTheDocument(),
    );
    expect(mockRun).not.toHaveBeenCalled();
  });

  it("says the depths are relative, not metres", async () => {
    // A depth page that renders a pretty map without this sentence teaches
    // something false: the scale is arbitrary and re-fitted per image.
    mockState = ready({ result: DEPTH });
    renderPage();
    await pickAndRun(/^tiger$/i);

    await waitFor(() =>
      expect(screen.getByTestId("depth-map")).toBeInTheDocument(),
    );
    const caveat = screen.getByText(/arbitrary scale, not metres/i);
    expect(screen.getByTestId("slot-4")).toContainElement(caveat);
    expect(screen.getByTestId("depth-legend")).toBeInTheDocument();
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
    expect(screen.getByTestId("slot-2")).not.toContainElement(note);
  });
});
