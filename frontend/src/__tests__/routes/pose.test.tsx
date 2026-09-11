import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { UsePoseResult } from "@/hooks/usePose";
import { COCO_KEYPOINTS } from "@/vision/pose/skeleton";

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

const person = (score: number, nose: [number, number], ankle: number) => ({
  box: { xmin: 0, ymin: 0, xmax: 4, ymax: 8 },
  score,
  keypoints: COCO_KEYPOINTS.map((_, index) => ({
    x: index === 0 ? nose[0] : 2,
    y: index === 0 ? nose[1] : index === 16 ? ankle : 4,
    // The ankle is a guess; every other joint is confident.
    score: index === 16 ? 0.1 : 0.9,
    index,
  })),
});

const RESULT = {
  people: [person(0.91, [2, 1], 7), person(0.55, [6, 2], 7)],
  detected: 3,
  detectMs: 18,
  poseMs: 120,
};

const mockRun = vi.fn().mockResolvedValue(RESULT);
const baseState: UsePoseResult = {
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
let mockState: UsePoseResult = { ...baseState };
const usePose = vi.fn(() => mockState);

vi.mock("@/hooks/usePose", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return { ...actual, usePose: (...a: unknown[]) => usePose(...(a as [])) };
});

const { Route } = await import("@/routes/pose");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Pose route component not found");
  return render(<Page />);
}

const RUN_BUTTON = /find poses/i;

// The RUN trigger. Picking an input never starts an inference any more
// (model-page-pattern.md §1.6), so a test that wants a result asks for one.
async function pickAndRun(sample: RegExp) {
  fireEvent.click(screen.getByRole("button", { name: sample }));
  await waitFor(() =>
    expect(screen.getByRole("button", { name: RUN_BUTTON })).toBeEnabled(),
  );
  fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));
}

const ready = (extra: Partial<UsePoseResult> = {}) => ({
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
  openCamera.mockResolvedValue(stopCamera);
  pickBackend.mockResolvedValue("webgpu");
  URL.createObjectURL = vi.fn(() => "blob:preview");
  URL.revokeObjectURL = vi.fn();
  vi.stubGlobal("requestAnimationFrame", () => 1);
  vi.stubGlobal("cancelAnimationFrame", () => {});
});
afterEach(() => vi.unstubAllGlobals());

describe("PosePage", () => {
  it("renders the heading and both model pairs", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /keypoint detection/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /d-fine nano \+ vitpose base/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /rt-detr r50 \+ vitpose base/i }),
    ).toBeInTheDocument();
  });

  it("downloads nothing on arrival, and loads only on request", () => {
    renderPage();
    expect(usePose).toHaveBeenCalledWith("dfine-n+vitpose-base");
    expect(baseState.load).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /load model/i }));
    expect(baseState.load).toHaveBeenCalledOnce();
  });

  it("quotes the combined download of both models", () => {
    // A guardrail that quotes half the bytes is worse than none, and the pair
    // crosses the large-model threshold less obviously than one big model.
    renderPage();
    const note = screen.getByTestId("model-size-note");
    expect(note).toHaveTextContent(/90M params/);
    expect(note).toHaveTextContent(/MB/);
  });

  it("fires the large-model warning on the sum, not on either half", async () => {
    // This is what "the pair crosses the threshold less obviously" means: the
    // RT-DETR pair is 88 MB of detector and 172 MB of pose model, each
    // comfortably under `LARGE_MODEL_BYTES`, and 260 MB together — over it. The
    // nano pair, correctly, does not warn at 180 MB.
    renderPage();
    expect(screen.queryByTestId("model-size-warning")).not.toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: /rt-detr r50 \+ vitpose base/i }),
    );
    await waitFor(() =>
      expect(screen.getByTestId("model-size-warning")).toBeInTheDocument(),
    );
  });

  it("renders the four slots with an empty OUTPUT before any run", () => {
    renderPage();
    expect(screen.getAllByRole("region")).toHaveLength(4);
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("keeps the run control disabled until both models are ready", () => {
    renderPage();
    expect(screen.getByRole("button", { name: /find poses/i })).toBeDisabled();
  });

  it("picks a sample without running, then estimates when asked, on a capped frame", async () => {
    mockState = ready();
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: /football match/i }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: RUN_BUTTON })).toBeEnabled(),
    );
    // Picking an input is not asking for an inference.
    expect(mockRun).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    expect(downscale).toHaveBeenCalledWith(fakeImage, 640);
    expect(mockRun).toHaveBeenCalledWith(fakeImage, {
      threshold: 0.4,
      maxPeople: 5,
      consume: false,
    });
  });

  it("re-runs when the threshold or the cap moves, because both change the work", async () => {
    // Unlike /object-detection's slider: filtering afterwards would mean
    // running the pose model on people the user has already excluded, and that
    // pass is the expensive half.
    mockState = ready({ result: RESULT });
    renderPage();
    await pickAndRun(/football match/i);
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));

    fireEvent.change(screen.getByLabelText(/person confidence/i), {
      target: { value: "0.6" },
    });
    fireEvent.click(screen.getByRole("button", { name: /find poses/i }));
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(2));
    expect(mockRun.mock.calls[1][1]).toMatchObject({ threshold: 0.6 });

    fireEvent.change(screen.getByLabelText(/most people to pose/i), {
      target: { value: "2" },
    });
    fireEvent.click(screen.getByRole("button", { name: /find poses/i }));
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(3));
    expect(mockRun.mock.calls[2][1]).toMatchObject({ maxPeople: 2 });
  });

  it("says how many people it posed out of how many it found, and what each half cost", async () => {
    mockState = ready({ result: RESULT });
    renderPage();
    await pickAndRun(/football match/i);

    await waitFor(() =>
      expect(screen.getByTestId("pose-canvas")).toBeInTheDocument(),
    );
    const out = within(screen.getByTestId("slot-4"));
    expect(out.getByText(/2 of 3/)).toBeInTheDocument();
    expect(out.getByText(/detect 18 ms/)).toBeInTheDocument();
    expect(out.getByText(/pose 120 ms/)).toBeInTheDocument();
  });

  it("lists one entry per person, and their joints on request", async () => {
    mockState = ready({ result: RESULT });
    renderPage();
    await pickAndRun(/football match/i);

    await waitFor(() =>
      expect(screen.getByTestId("people")).toBeInTheDocument(),
    );
    expect(
      within(screen.getByTestId("people")).getAllByRole("button"),
    ).toHaveLength(2);

    // Available but not dominant: the picture is the answer.
    const joints = within(screen.getByTestId("joints")).getAllByRole("listitem");
    expect(joints).toHaveLength(17);
    expect(joints[0]).toHaveTextContent(/nose 0\.90/);
    expect(joints[16]).toHaveTextContent(/right ankle 0\.10/);
    // Positions too, not only confidences: a skeleton offset by the crop's
    // origin looks like a mediocre model rather than a bug in our arithmetic.
    expect(joints[0]).toHaveTextContent(/2,1/);
    expect(joints[16]).toHaveTextContent(/2,7/);
  });

  it("says that a faint joint is a guess, not an absence", async () => {
    mockState = ready({ result: RESULT });
    renderPage();
    await pickAndRun(/football match/i);
    await waitFor(() =>
      expect(screen.getByTestId("slot-4")).toHaveTextContent(
        /the faint ones are guesses/i,
      ),
    );
  });

  it("explains an empty result instead of showing a blank canvas", async () => {
    mockState = ready({
      result: { people: [], detected: 0, detectMs: 12, poseMs: 0 },
    });
    renderPage();
    await pickAndRun(/football match/i);
    await waitFor(() =>
      expect(screen.getByTestId("slot-4")).toHaveTextContent(
        /no people above the confidence threshold/i,
      ),
    );
  });

  it("offers the camera on the nano pair only", async () => {
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

  it("puts an inference failure in OUTPUT, where both models stay loaded", () => {
    mockState = ready({ error: "Non-zero status code" });
    renderPage();
    const note = screen.getByText(/non-zero status code/i);
    expect(screen.getByTestId("slot-4")).toContainElement(note);
  });
});
