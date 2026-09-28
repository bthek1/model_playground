// `/robotics` — the page-pattern §8 checklist, plus the three things this page
// adds: the pair is quoted and loaded once, the threshold re-ranks without a
// run, and the "grounding, not control" note is on screen before any result.
//
// Layout is not asserted here (jsdom has no geometry); `e2e/specs/robotics.spec.ts`
// and `model-page.spec.ts` own that.

import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { GroundingResult, UseGroundingResult } from "@/hooks/useGrounding";
import { GROUNDING_PAIR } from "@/vision/grounding";

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

// The browser cache probe. Empty unless a test seeds it.
let cached = new Set<string>();
vi.mock("@/model/cache", () => ({
  cachedModels: () => Promise.resolve(cached),
  evictModel: vi.fn(() => Promise.resolve()),
}));

/**
 * An 8x8 inverse-depth map: the left half is near (10), the right half far (1).
 * The high-scoring car is on the right, the low-scoring bicycle on the left —
 * so "nearest" and "most confident" disagree, which is the only arrangement in
 * which a test can tell them apart.
 */
const DEPTH = {
  predicted_depth: {
    data: Array.from({ length: 64 }, (_, i) => (i % 8 < 4 ? 10 : 1)),
    dims: [1, 8, 8],
  },
};
const RESULT: GroundingResult = {
  detections: [
    { label: "a car", score: 0.31, box: { xmin: 4, ymin: 0, xmax: 8, ymax: 8 } },
    { label: "a bicycle", score: 0.12, box: { xmin: 0, ymin: 0, xmax: 4, ymax: 8 } },
    { label: "a car", score: 0.03, box: { xmin: 1, ymin: 1, xmax: 2, ymax: 2 } },
  ],
  depth: DEPTH,
  detectMs: 812,
  depthMs: 95,
};

const mockRun = vi.fn();
const base: UseGroundingResult = {
  pair: GROUNDING_PAIR,
  status: "idle",
  idle: true,
  loading: false,
  ready: false,
  loadProgress: null,
  loadedInMs: null,
  backend: null,
  error: null,
  running: false,
  result: null,
  load: vi.fn(),
  retry: vi.fn(),
  cancel: vi.fn(),
  run: mockRun,
};
let state: UseGroundingResult = { ...base };
const useGrounding = vi.fn<(...args: unknown[]) => UseGroundingResult>(
  () => state,
);
vi.mock("@/hooks/useGrounding", () => ({
  useGrounding: (...args: unknown[]) => useGrounding(...args),
}));

const { Route } = await import("@/routes/robotics");
const { useModelPrefs } = await import("@/store/models");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Robotics route component not found");
  return render(<Page />);
}

const LOCATE = /^locate$/i;
const ready = (extra: Partial<UseGroundingResult> = {}): UseGroundingResult => ({
  ...base,
  status: "ready",
  idle: false,
  ready: true,
  backend: "webgpu",
  ...extra,
});

async function pickAndLocate(sample = /^city street$/i) {
  fireEvent.click(screen.getByRole("button", { name: sample }));
  await waitFor(() =>
    expect(screen.getByRole("button", { name: LOCATE })).toBeEnabled(),
  );
  fireEvent.click(screen.getByRole("button", { name: LOCATE }));
  await waitFor(() => expect(screen.getByTestId("grounding-rows")).toBeInTheDocument());
}

const rows = () => within(screen.getByTestId("grounding-rows")).queryAllByTestId("grounding-row");

beforeEach(() => {
  vi.clearAllMocks();
  state = { ...base };
  cached = new Set();
  localStorage.clear();
  useModelPrefs.setState({ selected: {} });
  mockRun.mockResolvedValue(RESULT);
  fromUrl.mockResolvedValue(fakeImage);
  openCamera.mockResolvedValue(stopCamera);
  URL.createObjectURL = vi.fn(() => "blob:preview");
  URL.revokeObjectURL = vi.fn();
  vi.stubGlobal("requestAnimationFrame", () => 1);
  vi.stubGlobal("cancelAnimationFrame", () => {});
});
afterEach(() => vi.unstubAllGlobals());

describe("RoboticsPage", () => {
  describe("the shell", () => {
    it("renders the heading and the pair", () => {
      renderPage();
      expect(screen.getByRole("heading", { name: /^robotics$/i })).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: /OWLv2 base \+ Depth Anything V2 Small/i }),
      ).toBeInTheDocument();
    });

    it("renders all four slots, with an empty OUTPUT before any run", () => {
      renderPage();
      expect(screen.getAllByRole("region")).toHaveLength(4);
      for (const n of [1, 2, 3, 4]) {
        expect(screen.getByTestId(`slot-${n}`)).toBeInTheDocument();
      }
      expect(screen.getByTestId("output-empty")).toBeInTheDocument();
    });

    it("says this is grounding, not control, before any result exists", () => {
      renderPage();
      const note = screen.getByTestId("grounding-limitation");
      expect(screen.getByTestId("slot-4")).toContainElement(note);
      expect(screen.getByTestId("output-empty")).toBeInTheDocument();
      expect(note).toHaveTextContent(/grounding, not control/i);
      expect(note).toHaveTextContent(/issues no action/i);
      expect(note).toHaveTextContent(/demonstrations/i);
      expect(note).toHaveTextContent(/simulator/i);
      expect(note).toHaveTextContent(/relative, not metres/i);
    });

    it("quotes the combined size once, in SELECT, with the large-download warning", () => {
      renderPage();
      const select = screen.getByTestId("slot-1");
      expect(within(select).getByTestId("model-size-note")).toHaveTextContent(/341 MB/);
      expect(within(select).getByTestId("model-size-warning")).toBeInTheDocument();
      // ModelStatus must not repeat the number.
      expect(screen.getByTestId("slot-2")).not.toHaveTextContent(/MB/);
      // And no second opt-in: the pair is large, not heavy.
      expect(screen.queryByTestId("heavy-model-notice")).not.toBeInTheDocument();
    });

    it("says what kind of phrase the detector wants", () => {
      renderPage();
      expect(
        within(screen.getByTestId("slot-3")).getByText(/short noun phrases/i),
      ).toBeInTheDocument();
    });
  });

  describe("loading", () => {
    it("downloads nothing on arrival, and loads both only on request", () => {
      renderPage();
      // No `autoLoad` argument at all — its default is idle.
      expect(useGrounding).toHaveBeenLastCalledWith(GROUNDING_PAIR.id);
      expect(base.load).not.toHaveBeenCalled();

      fireEvent.click(screen.getByRole("button", { name: /^load model$/i }));
      expect(base.load).toHaveBeenCalledOnce();
    });

    it("lets a download in flight be abandoned", () => {
      state = { ...base, status: "loading", idle: false, loading: true };
      renderPage();
      fireEvent.click(screen.getByTestId("load-cancel"));
      expect(base.cancel).toHaveBeenCalledOnce();
    });

    it("retries from the LOAD slot after a failed load", () => {
      state = { ...base, status: "error", idle: false, error: "detector 404" };
      renderPage();
      fireEvent.click(
        within(screen.getByTestId("slot-2")).getByRole("button", { name: /^retry$/i }),
      );
      expect(base.retry).toHaveBeenCalled();
    });
  });

  describe("after a page refresh", () => {
    it("comes back on the stored selection, and a stale id falls back to the pair", () => {
      useModelPrefs.setState({ selected: { robotics: GROUNDING_PAIR.id } });
      const first = renderPage();
      expect(
        screen.getByRole("button", { name: /OWLv2 base \+ Depth Anything/i }),
      ).toHaveAttribute("aria-pressed", "true");
      first.unmount();

      useModelPrefs.setState({ selected: { robotics: "a-pair-we-no-longer-ship" } });
      renderPage();
      expect(useGrounding).toHaveBeenLastCalledWith(GROUNDING_PAIR.id);
    });

    it("stays idle even with both halves cached, and says the click is free", async () => {
      cached = new Set([GROUNDING_PAIR.detector.id, GROUNDING_PAIR.depth.id]);
      renderPage();
      await waitFor(() =>
        expect(
          screen.getByRole("button", { name: /load model \(cached\)/i }),
        ).toBeInTheDocument(),
      );
      expect(base.load).not.toHaveBeenCalled();
      expect(useGrounding).toHaveBeenLastCalledWith(GROUNDING_PAIR.id);
    });

    it("does not call the pair cached when only one half is", async () => {
      cached = new Set([GROUNDING_PAIR.depth.id]);
      renderPage();
      await act(async () => {});
      expect(screen.getByRole("button", { name: /^load model$/i })).toBeInTheDocument();
    });
  });

  describe("input spends nothing; only Locate runs", () => {
    it("gates Locate on ready, but never the input sources", () => {
      renderPage();
      expect(screen.getByRole("button", { name: LOCATE })).toBeDisabled();
      expect(screen.getByRole("button", { name: /^city street$/i })).toBeEnabled();
      expect(screen.getByRole("button", { name: /upload image/i })).toBeEnabled();
      expect(screen.getByLabelText(/^phrases$/i)).toBeEnabled();
    });

    it("picking a sample runs nothing; pressing Locate runs the pair once", async () => {
      state = ready();
      renderPage();

      fireEvent.click(screen.getByRole("button", { name: /^city street$/i }));
      await waitFor(() =>
        expect(screen.getByRole("button", { name: LOCATE })).toBeEnabled(),
      );
      expect(mockRun).not.toHaveBeenCalled();
      expect(screen.getByTestId("output-empty")).toBeInTheDocument();

      fireEvent.click(screen.getByRole("button", { name: LOCATE }));
      await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
      expect(downscale).toHaveBeenCalledWith(fakeImage, 640);
      expect(mockRun).toHaveBeenCalledWith(fakeImage, ["a car", "a bicycle"], {
        consume: false,
      });
    });

    it("editing the phrase list runs nothing", async () => {
      state = ready();
      renderPage();
      fireEvent.click(screen.getByRole("button", { name: /^city street$/i }));
      await waitFor(() =>
        expect(screen.getByRole("button", { name: LOCATE })).toBeEnabled(),
      );

      const run = within(screen.getByTestId("slot-3"));
      fireEvent.click(run.getByRole("button", { name: /remove a bicycle/i }));
      fireEvent.change(screen.getByLabelText(/^phrases$/i), {
        target: { value: "a red block" },
      });
      fireEvent.click(run.getByRole("button", { name: /^add$/i }));
      // By its remove control: the phrase also appears in the hint's example.
      expect(
        run.getByRole("button", { name: /remove a red block/i }),
      ).toBeInTheDocument();

      expect(mockRun).not.toHaveBeenCalled();
      expect(screen.getByTestId("output-empty")).toBeInTheDocument();

      fireEvent.click(screen.getByRole("button", { name: LOCATE }));
      await waitFor(() =>
        expect(mockRun).toHaveBeenCalledWith(fakeImage, ["a car", "a red block"], {
          consume: false,
        }),
      );
    });

    it("keeps the input across runs: two presses, one decode, two runs", async () => {
      state = ready();
      renderPage();
      await pickAndLocate();
      fireEvent.click(screen.getByRole("button", { name: LOCATE }));
      await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(2));
      expect(fromUrl).toHaveBeenCalledTimes(1);
    });

    it("flags a phrase that reads like an instruction", () => {
      renderPage();
      expect(screen.queryByTestId("phrase-warning")).not.toBeInTheDocument();
      fireEvent.change(screen.getByLabelText(/^phrases$/i), {
        target: { value: "pick up the red block nearest the camera" },
      });
      fireEvent.click(
        within(screen.getByTestId("slot-3")).getByRole("button", { name: /^add$/i }),
      );
      expect(screen.getByTestId("phrase-warning")).toHaveTextContent(
        /reads like a sentence/i,
      );
    });

    it("opens the camera only when asked, and closes it on unmount", async () => {
      state = ready();
      const { unmount } = renderPage();
      expect(openCamera).not.toHaveBeenCalled();
      await act(async () => {
        fireEvent.click(screen.getByRole("button", { name: /use camera/i }));
      });
      await waitFor(() => expect(openCamera).toHaveBeenCalled());
      unmount();
      expect(stopCamera).toHaveBeenCalled();
    });
  });

  describe("the result", () => {
    it("calls out the nearest match, which is not the most confident one", async () => {
      state = ready();
      renderPage();
      await pickAndLocate();

      expect(screen.getByTestId("nearest-callout")).toHaveTextContent(
        /nearest match:\s*a bicycle/i,
      );
      const [car, bicycle] = rows();
      expect(car).toHaveAttribute("data-label", "a car");
      expect(car).toHaveAttribute("data-rank", "2");
      expect(bicycle).toHaveAttribute("data-rank", "1");
      expect(bicycle).toHaveTextContent(/nearest/);
      // The raw value is quoted as the model's own number, never with a unit.
      expect(screen.getByTestId("grounding-rows")).not.toHaveTextContent(/\bm\b|metre/i);
      expect(screen.getByTestId("grounding-depth")).toBeInTheDocument();
      expect(screen.getByTestId("slot-4")).toHaveTextContent(/detect 812 ms · depth 95 ms/);
    });

    it("re-filters and re-ranks as the threshold moves, without running again", async () => {
      state = ready();
      renderPage();
      await pickAndLocate();
      expect(rows()).toHaveLength(2);

      fireEvent.change(screen.getByLabelText(/confidence threshold/i), {
        target: { value: "0.2" },
      });
      expect(rows()).toHaveLength(1);
      // With the bicycle filtered out, the car is the nearest remaining match.
      expect(screen.getByTestId("nearest-callout")).toHaveTextContent(/a car/);
      expect(screen.getByTestId("grounding-missing")).toHaveTextContent(/a bicycle/);

      fireEvent.change(screen.getByLabelText(/confidence threshold/i), {
        target: { value: "0.02" },
      });
      expect(rows()).toHaveLength(3);

      expect(mockRun).toHaveBeenCalledTimes(1);
    });

    it("says nothing was found rather than boxing something", async () => {
      state = ready();
      mockRun.mockResolvedValue({ ...RESULT, detections: [] });
      renderPage();
      await pickAndLocate();
      expect(screen.getByTestId("nearest-callout")).toHaveTextContent(
        /nothing above 0\.10 matched any phrase/i,
      );
    });

    it("labels the result with the phrases it was asked, not the ones typed since", async () => {
      state = ready();
      mockRun.mockResolvedValue({ ...RESULT, detections: [] });
      renderPage();
      await pickAndLocate();
      expect(screen.getByTestId("grounding-missing")).toHaveTextContent(/a car.*a bicycle/);

      fireEvent.change(screen.getByLabelText(/^phrases$/i), {
        target: { value: "a red block" },
      });
      fireEvent.click(
        within(screen.getByTestId("slot-3")).getByRole("button", { name: /^add$/i }),
      );
      expect(screen.getByTestId("grounding-missing")).not.toHaveTextContent(/red block/);
    });
  });

  describe("errors land in the slot that produced them", () => {
    it("a load failure in LOAD", () => {
      state = { ...base, status: "error", idle: false, error: "404 not found" };
      renderPage();
      const note = screen.getByText(/404 not found/i);
      expect(screen.getByTestId("slot-2")).toContainElement(note);
      expect(screen.getByTestId("slot-4")).not.toContainElement(note);
    });

    it("an inference failure in OUTPUT", () => {
      state = ready({ error: "Non-zero status code" });
      renderPage();
      const note = screen.getByText(/non-zero status code/i);
      expect(screen.getByTestId("slot-4")).toContainElement(note);
    });

    it("a failed decode in RUN, having run nothing", async () => {
      state = ready();
      fromUrl.mockRejectedValueOnce(new Error("Unsupported image type"));
      renderPage();
      fireEvent.click(screen.getByRole("button", { name: /^tiger$/i }));
      const note = await screen.findByText(/unsupported image type/i);
      expect(screen.getByTestId("slot-3")).toContainElement(note);
      expect(mockRun).not.toHaveBeenCalled();
    });
  });

  describe("the behaviour-cloning entry (#54)", () => {
    it("mounts none of the grounding pair's workers once it is selected", () => {
      useModelPrefs.setState({ selected: { robotics: "behaviour-cloning" } });
      useGrounding.mockClear();
      renderPage();
      expect(useGrounding).not.toHaveBeenCalled();
      expect(screen.getByRole("button", { name: /behaviour cloning/i })).toHaveAttribute("aria-pressed", "true");
      // Zero bytes: no large-download warning, and LOAD is already answered.
      expect(screen.queryByText(/large model/i)).toBeNull();
      expect(screen.getByTestId("model-ready")).toHaveTextContent(/nothing to download/i);
      expect(screen.getByTestId("slot-4")).toBeInTheDocument();
    });
  });
});
