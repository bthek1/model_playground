import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type {
  ClipScores,
  UseVideoClassifierResult,
} from "@/hooks/useVideoClassifier";

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

// The chart is lazy and pulls in echarts; the route's own behaviour is what is
// under test, not the plotting library.
vi.mock("@/components/charts/EChart", () => ({
  default: () => <div data-testid="chart" />,
}));

const LABELS = ["an interview", "a football match", "a car chase"];
const RESULT: ClipScores = {
  labels: LABELS,
  duration: 6,
  capped: false,
  frames: [
    { time: 0.25, scores: [0.9, 0.05, 0.05], thumb: "data:image/jpeg;base64,a" },
    { time: 0.75, scores: [0.1, 0.8, 0.1], thumb: "data:image/jpeg;base64,b" },
    { time: 1.25, scores: [0.8, 0.1, 0.1], thumb: "data:image/jpeg;base64,c" },
  ],
};

const mockRun = vi.fn().mockResolvedValue(RESULT);
const mockStop = vi.fn();
const baseState: UseVideoClassifierResult = {
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
  clipProgress: null,
  run: mockRun,
  stop: mockStop,
  load: vi.fn(),
  retry: vi.fn(),
  cancel: vi.fn(),
};
let mockState: UseVideoClassifierResult = { ...baseState };
const useVideoClassifier = vi.fn(() => mockState);

vi.mock("@/hooks/useVideoClassifier", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    useVideoClassifier: (...a: unknown[]) => useVideoClassifier(...(a as [])),
  };
});

const { Route } = await import("@/routes/video-classification");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Video classification route component not found");
  return render(<Page />);
}

const ready = (extra: Partial<UseVideoClassifierResult> = {}) => ({
  ...baseState,
  status: "ready" as const,
  idle: false,
  ready: true,
  ...extra,
});

beforeEach(() => {
  vi.clearAllMocks();
  mockState = { ...baseState };
  mockRun.mockResolvedValue(RESULT);
  URL.createObjectURL = vi.fn(() => "blob:clip");
  URL.revokeObjectURL = vi.fn();
});
afterEach(() => vi.clearAllMocks());

describe("VideoClassificationPage", () => {
  it("renders the heading and the CLIP catalogue it reuses", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /video classification/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /clip vit-b\/32/i }),
    ).toBeInTheDocument();
  });

  it("downloads nothing on arrival, and loads only on request", () => {
    renderPage();
    expect(useVideoClassifier).toHaveBeenCalledWith(
      "Xenova/clip-vit-base-patch32",
      false,
    );
    expect(baseState.load).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /load model/i }));
    expect(baseState.load).toHaveBeenCalledOnce();
  });

  it("renders the four slots with an empty OUTPUT before any run", () => {
    renderPage();
    expect(screen.getAllByRole("region")).toHaveLength(4);
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("calls itself a frame-level baseline before anything has run", () => {
    // Not decoration: shipping this as "Video Classification" without the
    // framing teaches something false, so it is asserted like any other
    // correctness requirement.
    renderPage();
    expect(screen.getByTestId("slot-1").parentElement).toBeTruthy();
    expect(document.body).toHaveTextContent(/frame-level baseline/i);
    expect(document.body).toHaveTextContent(/never sees motion/i);
  });

  it("keeps the run control disabled until a model is ready", () => {
    renderPage();
    expect(
      screen.getByRole("button", { name: /score the clip/i }),
    ).toBeDisabled();
  });

  it("starts from the sample's own label set, and swaps it with the clip", () => {
    // A zero-shot score is relative to the labels given, so a clip whose list
    // has no plausible distractor demonstrates nothing.
    renderPage();
    const run = within(screen.getByTestId("slot-3"));
    expect(run.getByText("an interview")).toBeInTheDocument();

    fireEvent.click(run.getByRole("button", { name: /^courtroom$/i }));
    expect(run.getByText("a courtroom")).toBeInTheDocument();
    expect(run.queryByText("an interview")).not.toBeInTheDocument();
  });

  it("lets the labels be edited before a model exists", () => {
    renderPage();
    const run = within(screen.getByTestId("slot-3"));
    fireEvent.change(screen.getByLabelText(/^labels$/i), {
      target: { value: "a courtroom" },
    });
    fireEvent.click(run.getByRole("button", { name: /^add$/i }));
    expect(run.getByText("a courtroom")).toBeInTheDocument();
  });

  it("scores the selected clip with the templated labels", async () => {
    mockState = ready();
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /score the clip/i }));

    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    const [src, labels, template, opts] = mockRun.mock.calls[0];
    expect(src).toMatch(/interview\.mp4$/);
    expect(labels).toEqual(LABELS);
    expect(template).toBe("a photo of a {}");
    expect(opts).toMatchObject({ fps: 2 });
  });

  it("offers Stop instead of Run while a clip is being scored", () => {
    // The only page whose run is minutes long.
    mockState = ready({
      running: true,
      clipProgress: { phase: "scoring", done: 4, total: 12 },
    });
    renderPage();

    expect(
      screen.queryByRole("button", { name: /score the clip/i }),
    ).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /^stop$/i }));
    expect(mockStop).toHaveBeenCalled();

    expect(screen.getByTestId("clip-progress")).toHaveTextContent(/4 \/ 12/);
    expect(screen.getByTestId("clip-progress")).toHaveTextContent(/scoring/i);
  });

  it("charts the clip and names the pooled winner", async () => {
    mockState = ready({ result: RESULT });
    renderPage();
    await waitFor(() => expect(screen.getByTestId("chart")).toBeInTheDocument());
    // Means: interview 0.6, football 0.317, car chase 0.083.
    expect(screen.getByTestId("pooled-winner")).toHaveTextContent(
      /an interview/,
    );
    expect(screen.getByTestId("pooled-winner")).toHaveTextContent(/0\.600/);
  });

  it("re-pools without re-running when the window moves", async () => {
    mockState = ready({ result: RESULT });
    renderPage();
    await waitFor(() => expect(screen.getByTestId("chart")).toBeInTheDocument());

    fireEvent.change(screen.getByLabelText(/pooling window/i), {
      target: { value: "3" },
    });
    expect(screen.getByTestId("pooled-winner")).toHaveTextContent(/window 3/);
    // The whole point: re-scoring would be N CLIP passes over the clip.
    expect(mockRun).not.toHaveBeenCalled();
  });

  it("shows the sampled frames as a filmstrip, so a spike is traceable", async () => {
    mockState = ready({ result: RESULT });
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("filmstrip")).toBeInTheDocument(),
    );
    expect(
      within(screen.getByTestId("filmstrip")).getAllByRole("listitem"),
    ).toHaveLength(3);
    expect(screen.getByAltText(/frame at 0\.8s/i)).toBeInTheDocument();
  });

  it("states the limitation next to the result, not in a footnote", () => {
    // The route's actual correctness requirement, per #21.
    mockState = ready({ result: RESULT });
    renderPage();
    const note = screen.getByTestId("baseline-note");
    expect(screen.getByTestId("slot-4")).toContainElement(note);
    expect(note).toHaveTextContent(/frame-level baseline/i);
    expect(note).toHaveTextContent(/videomae/i);
    expect(note).toHaveTextContent(/no onnx export|none of them has an onnx export/i);
  });

  it("says when the frame cap cut the clip short", async () => {
    mockState = ready({ result: { ...RESULT, capped: true } });
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("slot-4")).toHaveTextContent(
        /capped, so this is not the whole clip/i,
      ),
    );
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

  it("puts a clip that will not decode in the RUN slot, where the file is", async () => {
    mockState = ready();
    mockRun.mockRejectedValueOnce(new Error("Could not decode that video"));
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: /score the clip/i }));
    const note = await screen.findByText(/could not decode that video/i);
    expect(screen.getByTestId("slot-3")).toContainElement(note);
  });

  it("puts an inference failure in OUTPUT, where the model stays loaded", () => {
    mockState = ready({ error: "Non-zero status code" });
    renderPage();
    const note = screen.getByText(/non-zero status code/i);
    expect(screen.getByTestId("slot-4")).toContainElement(note);
  });
});
