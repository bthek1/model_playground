import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { UseImageClassifierResult } from "@/hooks/useImageClassifier";

// The image helpers reach for RawImage / canvas, neither of which exists under
// happy-dom. The route only ever passes what they return straight to `run`.
const fakeImage = { width: 4, height: 4, channels: 3 } as never;
const fromFile = vi.fn().mockResolvedValue(fakeImage);
const fromUrl = vi.fn().mockResolvedValue(fakeImage);
vi.mock("@/vision/image", () => ({
  fromFile: (...args: unknown[]) => fromFile(...args),
  fromUrl: (...args: unknown[]) => fromUrl(...args),
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

const mockRun = vi.fn().mockResolvedValue([]);
const baseState: UseImageClassifierResult = {
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
let mockState: UseImageClassifierResult = { ...baseState };
const useImageClassifier = vi.fn(() => mockState);

vi.mock("@/hooks/useImageClassifier", () => ({
  useImageClassifier: (...args: unknown[]) => useImageClassifier(...(args as [])),
}));

const { Route } = await import("@/routes/image-classification");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Image classification route component not found");
  render(<Page />);
}

const ready = (extra: Partial<UseImageClassifierResult> = {}) => ({
  ...baseState,
  status: "ready" as const,
  idle: false,
  ready: true,
  ...extra,
});

describe("ImageClassificationPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockState = { ...baseState };
    URL.createObjectURL = vi.fn(() => "blob:preview");
    URL.revokeObjectURL = vi.fn();
  });

  it("renders the heading and every model option", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /image classification/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /vit-base/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /resnet-50/i })).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /mobilenetv4/i }),
    ).toBeInTheDocument();
  });

  it("downloads nothing on arrival, and loads only on request", () => {
    renderPage();
    // Both halves matter: the hook is handed no auto-load option at all (its
    // default is `idle`), *and* nothing has called load() behind the user's
    // back.
    expect(useImageClassifier).toHaveBeenCalledWith("Xenova/vit-base-patch16-224");
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
    expect(screen.getByRole("button", { name: /^classify$/i })).toBeDisabled();
    expect(screen.getByText(/load a model to classify an image/i)).toBeInTheDocument();
  });

  it("keeps the run control disabled while ready but with no image picked", () => {
    mockState = ready();
    renderPage();
    expect(screen.getByRole("button", { name: /^classify$/i })).toBeDisabled();
  });

  // The core of the SELECT → LOAD → INPUT → GENERATE contract, and the thing
  // this page got wrong: picking an image used to classify it immediately, so a
  // user comparing the sample row spent one inference per click and the
  // Classify button beside them did nothing they hadn't already paid for.
  it("shows a picked sample and runs nothing, even with a model ready", async () => {
    mockState = ready();
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: /^tiger$/i }));

    await waitFor(() =>
      expect(screen.getByAltText(/selected input: tiger/i)).toBeInTheDocument(),
    );
    expect(fromUrl).toHaveBeenCalledWith(expect.stringContaining("tiger.jpg"));
    expect(mockRun).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("classifies the picked image when — and only when — Classify is pressed", async () => {
    mockState = ready();
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: /^tiger$/i }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /^classify$/i })).toBeEnabled(),
    );
    expect(mockRun).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /^classify$/i }));
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    expect(mockRun).toHaveBeenCalledWith(fakeImage);
  });

  it("shows a picked image without running anything when no model is loaded", async () => {
    renderPage(); // idle
    fireEvent.click(screen.getByRole("button", { name: /^tiger$/i }));

    await waitFor(() =>
      expect(screen.getByAltText(/selected input: tiger/i)).toBeInTheDocument(),
    );
    expect(mockRun).not.toHaveBeenCalled();
  });

  it("previews an uploaded file and revokes the preview it replaces", async () => {
    mockState = ready();
    renderPage();

    const input = screen.getByLabelText(/upload an image/i);
    const file = new File(["x"], "cat.png", { type: "image/png" });
    fireEvent.change(input, { target: { files: [file] } });

    await waitFor(() => expect(fromFile).toHaveBeenCalledWith(file));
    expect(URL.createObjectURL).toHaveBeenCalledWith(file);
    // Opening a file dialog is not asking for an inference.
    expect(mockRun).not.toHaveBeenCalled();

    // A second pick frees the first preview: an object URL that outlives its
    // <img> pins the decoded bitmap for the tab's lifetime.
    fireEvent.click(screen.getByRole("button", { name: /^cats$/i }));
    await waitFor(() => expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:preview"));
  });

  it("accepts a dropped image, and still waits to be told to run", async () => {
    mockState = ready();
    renderPage();

    const file = new File(["x"], "dropped.png", { type: "image/png" });
    const dropzone = screen.getByText(/drop an image here/i).parentElement!;
    fireEvent.drop(dropzone, { dataTransfer: { files: [file] } });

    await waitFor(() => expect(fromFile).toHaveBeenCalledWith(file));
    expect(mockRun).not.toHaveBeenCalled();
  });

  it("re-runs the image already picked when the user asks again", async () => {
    // Switching model and pressing Classify again must not require re-picking
    // the picture — the RawImage is kept, and `run` copies rather than consumes
    // its pixels precisely so this works twice.
    mockState = ready();
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: /^tiger$/i }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /^classify$/i })).toBeEnabled(),
    );

    fireEvent.click(screen.getByRole("button", { name: /^classify$/i }));
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("button", { name: /^classify$/i }));
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(2));
    expect(fromUrl).toHaveBeenCalledTimes(1); // the image was not re-fetched
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

  it("renders the top five with their scores", () => {
    mockState = ready({
      backend: "webgpu",
      result: [
        { label: "tabby", score: 0.51 },
        { label: "tiger cat", score: 0.29 },
        { label: "Egyptian cat", score: 0.11 },
        { label: "lynx", score: 0.05 },
        { label: "carton", score: 0.01 },
      ],
    });
    renderPage();

    expect(screen.getByText("tabby")).toBeInTheDocument();
    expect(screen.getByText("51%")).toBeInTheDocument();
    expect(screen.getByText("carton")).toBeInTheDocument();
    expect(screen.getByText(/top-2 margin 0.22/)).toBeInTheDocument();
  });

  it("calls out a near-tie at the top rather than presenting a confident answer", () => {
    mockState = ready({
      result: [
        { label: "tabby", score: 0.31 },
        { label: "tiger cat", score: 0.29 },
      ],
    });
    renderPage();
    expect(screen.getByText(/not confident/i)).toBeInTheDocument();
  });

  it("puts a load failure in the LOAD slot, not in OUTPUT", () => {
    mockState = {
      ...baseState,
      status: "error",
      idle: false,
      error: "404 model not found",
    };
    renderPage();

    const note = screen.getByText(/404 model not found/i);
    expect(screen.getByTestId("slot-2")).toContainElement(note);
    expect(screen.getByTestId("slot-4")).not.toContainElement(note);
  });

  it("puts an inference failure in OUTPUT, where the model stays loaded", () => {
    mockState = ready({ error: "Unsupported image format" });
    renderPage();

    const note = screen.getByText(/unsupported image format/i);
    expect(screen.getByTestId("slot-4")).toContainElement(note);
    expect(screen.getByTestId("slot-2")).not.toContainElement(note);
  });
});
