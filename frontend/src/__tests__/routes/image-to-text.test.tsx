import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { UseImageToTextResult } from "@/hooks/useImageToText";
import type { CaptionMode } from "@/vision/caption/types";

const fakeImage = { width: 8, height: 8, channels: 3, data: [] } as never;
const fromFile = vi.fn().mockResolvedValue(fakeImage);
const fromUrl = vi.fn().mockResolvedValue(fakeImage);
const downscale = vi.fn(async (img: unknown) => img);
vi.mock("@/vision/image", () => ({
  fromFile: (...a: unknown[]) => fromFile(...a),
  fromUrl: (...a: unknown[]) => fromUrl(...a),
  downscale: (...a: unknown[]) => downscale(...(a as [never])),
}));

const pickBackend = vi.fn(async () => "webgpu");
vi.mock("@/model/backend", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return { ...actual, pickBackend: () => pickBackend() };
});

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

const mockRun = vi.fn();
const mockSetMode = vi.fn();
let currentMode: CaptionMode = "<CAPTION>";
const ALL_MODES: CaptionMode[] = [
  "<CAPTION>",
  "<DETAILED_CAPTION>",
  "<OCR>",
  "<OD>",
];

const baseState = (): UseImageToTextResult => ({
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
  modes: ALL_MODES,
  mode: currentMode,
  setMode: mockSetMode,
  run: mockRun,
  load: vi.fn(),
  retry: vi.fn(),
  cancel: vi.fn(),
});
let mockState: UseImageToTextResult = baseState();
const useImageToText = vi.fn(() => mockState);

vi.mock("@/hooks/useImageToText", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    useImageToText: (...a: unknown[]) => useImageToText(...(a as [])),
  };
});

const { Route } = await import("@/routes/image-to-text");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Image-to-text route component not found");
  return render(<Page />);
}

const ready = (extra: Partial<UseImageToTextResult> = {}) => ({
  ...baseState(),
  status: "ready" as const,
  idle: false,
  ready: true,
  ...extra,
});

beforeEach(() => {
  vi.clearAllMocks();
  currentMode = "<CAPTION>";
  mockState = baseState();
  mockRun.mockResolvedValue({
    kind: "text",
    mode: "<CAPTION>",
    text: "two cats",
    ms: 1800,
  });
  pickBackend.mockResolvedValue("webgpu");
  URL.createObjectURL = vi.fn(() => "blob:preview");
  URL.revokeObjectURL = vi.fn();
});
afterEach(() => vi.clearAllMocks());

describe("ImageToTextPage", () => {
  it("renders the heading and both model options", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /image to text/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /florence-2 base/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /vit-gpt2/i }),
    ).toBeInTheDocument();
  });

  it("downloads nothing on arrival, and loads only on request", () => {
    renderPage();
    expect(useImageToText).toHaveBeenCalledWith(
      "onnx-community/Florence-2-base-ft",
      false,
    );
    expect(mockState.load).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /load model/i }));
    expect(mockState.load).toHaveBeenCalledOnce();
  });

  it("renders the four slots with an empty OUTPUT before any run", () => {
    renderPage();
    expect(screen.getAllByRole("region")).toHaveLength(4);
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("keeps the run control disabled until a model is ready", () => {
    renderPage();
    // "Generate", not the mode's own name: the mode selector already carries
    // that label, and two buttons reading "Caption" is ambiguous.
    expect(screen.getByRole("button", { name: /^generate$/i })).toBeDisabled();
  });

  it("offers exactly the modes the hook reports", () => {
    renderPage();
    const modes = within(screen.getByTestId("modes"));
    expect(modes.getByRole("button", { name: /^OCR$/ })).toBeInTheDocument();
    expect(modes.getByRole("button", { name: /^Grounding$/ })).toBeInTheDocument();
  });

  it("offers only the caption mode on a model that only captions", () => {
    mockState = { ...baseState(), modes: ["<CAPTION>"] };
    renderPage();
    const modes = within(screen.getByTestId("modes"));
    expect(modes.getByRole("button", { name: /^Caption$/ })).toBeInTheDocument();
    expect(modes.queryByRole("button", { name: /^OCR$/ })).not.toBeInTheDocument();
    expect(screen.getByTestId("slot-3")).toHaveTextContent(
      /only captions/i,
    );
  });

  it("asks the hook to change mode rather than sending one itself", () => {
    renderPage();
    fireEvent.click(
      within(screen.getByTestId("modes")).getByRole("button", { name: /^OCR$/ }),
    );
    expect(mockSetMode).toHaveBeenCalledWith("<OCR>");
    expect(mockRun).not.toHaveBeenCalled();
  });

  it("hides the WebGPU-only model when the probe reports WASM", async () => {
    // The limitation is known before the click. Offering it anyway turns it
    // into a failed 275 MB download.
    pickBackend.mockResolvedValue("wasm");
    renderPage();
    await waitFor(() =>
      expect(
        screen.getByTestId("model-unsupported-onnx-community/Florence-2-base-ft"),
      ).toBeInTheDocument(),
    );
    expect(
      screen.getByRole("button", { name: /florence-2 base/i }),
    ).toBeDisabled();
    // The CPU-capable model stays selectable.
    expect(
      screen.getByRole("button", { name: /vit-gpt2/i }),
    ).not.toBeDisabled();
  });

  it("does not gate anything while the probe is still undecided", () => {
    // `null` means "ask again in 10 ms", not "no GPU". Greying out every WebGPU
    // model for a frame reads as "this machine cannot run it".
    renderPage();
    expect(
      screen.queryByTestId("model-unsupported-onnx-community/Florence-2-base-ft"),
    ).not.toBeInTheDocument();
  });

  it("generates as soon as a sample is picked, on a capped frame", async () => {
    mockState = ready();
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /^tiger$/i }));
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    expect(downscale).toHaveBeenCalledWith(fakeImage, 1024);
  });

  it("says a generative run takes seconds, before the first one", () => {
    renderPage();
    expect(screen.getByTestId("slot-3")).toHaveTextContent(
      /generative decoder: expect\s+seconds/i,
    );
  });

  it("renders a text answer as prose", async () => {
    mockState = ready({
      result: { kind: "text", mode: "<OCR>", text: "PARKING", ms: 2100 },
    });
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("slot-4")).toHaveTextContent(/PARKING/),
    );
    expect(screen.queryByTestId("grounding-canvas")).not.toBeInTheDocument();
  });

  it("routes a grounded answer to the canvas, not to the text renderer", async () => {
    mockState = ready({
      result: {
        kind: "boxes",
        mode: "<OD>",
        ms: 3400,
        detections: [
          { label: "cat", score: 1, box: { xmin: 1, ymin: 2, xmax: 3, ymax: 4 } },
        ],
      },
    });
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /^tiger$/i }));

    await waitFor(() =>
      expect(screen.getByTestId("grounding-canvas")).toBeInTheDocument(),
    );
    expect(
      within(screen.getByTestId("grounding-list")).getByText("cat"),
    ).toBeInTheDocument();
    // No invented confidence: the model writes locations as tokens.
    expect(screen.getByTestId("slot-4")).toHaveTextContent(
      /no confidence score/i,
    );
  });

  it("shows how long the generation took", async () => {
    mockState = ready({
      result: { kind: "text", mode: "<CAPTION>", text: "a cat", ms: 1800 },
    });
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("generate-ms")).toHaveTextContent(/1\.8s/),
    );
  });

  it("explains an empty generation instead of showing a blank panel", () => {
    mockState = ready({
      result: { kind: "text", mode: "<OCR>", text: "", ms: 900 },
    });
    renderPage();
    expect(screen.getByTestId("slot-4")).toHaveTextContent(
      /generated nothing/i,
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
      ...baseState(),
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
