import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { UseVlmResult } from "@/hooks/useVlm";
import type { SampledFrame } from "@/vision/video";

// The decode path, mocked: happy-dom has no video decoder and no canvas, and
// this suite is about what the page *asks* for, not about seeking.
const sampleVideo = vi.fn();
const thumbnail = vi.fn(() => "data:image/jpeg;base64,xx");
vi.mock("@/vision/video", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    sampleVideo: (...a: unknown[]) => sampleVideo(...a),
    thumbnail: () => thumbnail(),
  };
});

const pickBackend = vi.fn(async () => "webgpu");
const supportsShaderF16 = vi.fn(async () => true);
vi.mock("@/model/backend", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    pickBackend: () => pickBackend(),
    supportsShaderF16: () => supportsShaderF16(),
  };
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
const mockLoad = vi.fn();

const baseState = (): UseVlmResult => ({
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
  partial: null,
  run: mockRun,
  load: mockLoad,
  retry: vi.fn(),
  cancel: vi.fn(),
});
let mockState: UseVlmResult = baseState();
const useVlm = vi.fn(() => mockState);
vi.mock("@/hooks/useVlm", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return { ...actual, useVlm: (...a: unknown[]) => useVlm(...(a as [])) };
});

let cached = new Set<string>();
vi.mock("@/model/cache", () => ({
  cachedModels: () => Promise.resolve(cached),
  evictModel: vi.fn(() => Promise.resolve()),
}));

const { Route } = await import("@/routes/video-text-to-text");
const { useModelPrefs } = await import("@/store/models");
const { VIDEO_VLM_MODELS } = await import("@/multimodal/types");
const MODEL = VIDEO_VLM_MODELS[0].id;
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Video-text-to-text route component not found");
  return render(<Page />);
}

/** `n` fake frames at the timestamps the page would have asked for. */
function frames(n: number): SampledFrame[] {
  return Array.from({ length: n }, (_, i) => ({
    time: i + 0.5,
    image: { width: 8, height: 8, channels: 3, data: [i] } as never,
  }));
}

const RUN_BUTTON = /^generate$/i;
const CLIP = /^interview$/i;

const ready = (extra: Partial<UseVlmResult> = {}): UseVlmResult => ({
  ...baseState(),
  status: "ready",
  idle: false,
  ready: true,
  ...extra,
});

beforeEach(() => {
  vi.clearAllMocks();
  mockState = baseState();
  cached = new Set();
  useModelPrefs.setState({ selected: {} });
  // The default frame count is 4, so the decode answers with four frames
  // unless a test asks the sampler what it was called with.
  sampleVideo.mockImplementation(async () => ({
    frames: frames(4),
    duration: 8,
    capped: false,
  }));
  mockRun.mockResolvedValue({
    text: "Two people are talking.",
    ms: 9000,
    encodeMs: 4000,
    tokens: 6,
  });
  pickBackend.mockResolvedValue("webgpu");
  supportsShaderF16.mockResolvedValue(true);
  URL.createObjectURL = vi.fn(() => "blob:clip");
  URL.revokeObjectURL = vi.fn();
});
afterEach(() => vi.clearAllMocks());

describe("VideoTextToTextPage — the four-slot contract", () => {
  it("renders the heading and the video catalogue", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /video text to text/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /smolvlm2 256m video/i }),
    ).toBeInTheDocument();
    // Its own catalogue: the still-image entries must not appear here.
    expect(
      screen.queryByRole("button", { name: /^smolvlm 256m$/i }),
    ).not.toBeInTheDocument();
  });

  it("renders all four slots with an empty OUTPUT before any run", () => {
    renderPage();
    expect(screen.getAllByRole("region")).toHaveLength(4);
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("downloads nothing on arrival, and loads only on request", () => {
    renderPage();
    expect(useVlm).toHaveBeenCalledWith(MODEL);
    expect(mockLoad).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /load model/i }));
    expect(mockLoad).toHaveBeenCalledOnce();
  });

  it("keeps GENERATE disabled until ready, while the input sources stay live", () => {
    renderPage();
    expect(screen.getByRole("button", { name: RUN_BUTTON })).toBeDisabled();
    expect(screen.getByRole("button", { name: CLIP })).toBeEnabled();
    expect(screen.getByTestId("frame-count")).toBeEnabled();
    expect(screen.getByTestId("reverse-toggle")).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: CLIP }));
    expect(mockRun).not.toHaveBeenCalled();
    expect(sampleVideo).not.toHaveBeenCalled();
  });

  it("comes back on the stored selection and stays idle with weights cached", async () => {
    cached = new Set([MODEL]);
    useModelPrefs.setState({ selected: { "video-text-to-text": MODEL } });
    renderPage();
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /load model \(cached\)/i }),
      ).toBeEnabled(),
    );
    expect(mockLoad).not.toHaveBeenCalled();
  });
});

describe("VideoTextToTextPage — nothing runs until asked", () => {
  it("picking a clip decodes nothing and runs nothing", () => {
    mockState = ready();
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: CLIP }));
    // Not even the decode: frame extraction is seeks, and browsing the samples
    // must not pay for them.
    expect(sampleVideo).not.toHaveBeenCalled();
    expect(mockRun).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("changing the frame count runs nothing — it changes what the next run samples", async () => {
    mockState = ready();
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: CLIP }));

    fireEvent.change(screen.getByTestId("frame-count"), {
      target: { value: "8" },
    });
    expect(mockRun).not.toHaveBeenCalled();
    expect(sampleVideo).not.toHaveBeenCalled();

    sampleVideo.mockResolvedValueOnce({
      frames: frames(8),
      duration: 8,
      capped: false,
    });
    fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    // Eight frames reached the model, in one run.
    expect(mockRun.mock.calls[0][0]).toHaveLength(8);
  });

  it("flipping the frame order runs nothing on its own", () => {
    mockState = ready();
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: CLIP }));
    fireEvent.click(screen.getByTestId("reverse-toggle"));
    expect(mockRun).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("editing the question and tapping a preset run nothing", () => {
    mockState = ready();
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: CLIP }));
    fireEvent.change(screen.getByTestId("question-input"), {
      target: { value: "What colour is the room?" },
    });
    fireEvent.click(
      within(screen.getByTestId("presets")).getByRole("button", {
        name: /describe the scene/i,
      }),
    );
    expect(mockRun).not.toHaveBeenCalled();
    expect(screen.getByTestId("question-input")).toHaveValue(
      "Describe the scene.",
    );
  });

  it("survives its run: two generations cost one decode", async () => {
    mockState = ready();
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: CLIP }));

    fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(2));

    // Sampling is seeks; the cache is keyed on (clip, frame count).
    expect(sampleVideo).toHaveBeenCalledTimes(1);
  });

  it("will not run without a clip, or on an empty question", () => {
    mockState = ready();
    renderPage();
    expect(screen.getByRole("button", { name: RUN_BUTTON })).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: CLIP }));
    expect(screen.getByRole("button", { name: RUN_BUTTON })).toBeEnabled();

    fireEvent.change(screen.getByTestId("question-input"), {
      target: { value: "  " },
    });
    expect(screen.getByRole("button", { name: RUN_BUTTON })).toBeDisabled();
  });
});

describe("VideoTextToTextPage — the reverse experiment", () => {
  it("sends the frames in reverse, as a real second inference", async () => {
    mockState = ready();
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: CLIP }));

    fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    const forward = (mockRun.mock.calls[0][0] as Array<{ data: number[] }>).map(
      (f) => f.data[0],
    );
    expect(forward).toEqual([0, 1, 2, 3]);

    fireEvent.click(screen.getByTestId("reverse-toggle"));
    fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(2));
    const back = (mockRun.mock.calls[1][0] as Array<{ data: number[] }>).map(
      (f) => f.data[0],
    );
    // The model really sees the other order — a relabelled result would not be
    // an experiment.
    expect(back).toEqual([3, 2, 1, 0]);
    // And it cost no second decode: same clip, same count.
    expect(sampleVideo).toHaveBeenCalledTimes(1);
  });

  it("says a flip costs an inference, unlike a threshold", () => {
    renderPage();
    expect(screen.getByTestId("slot-3")).toHaveTextContent(
      /the next Generate is a real second inference/i,
    );
  });
});

describe("VideoTextToTextPage — the answer and the filmstrip", () => {
  it("shows the frames that were sent, in sampled order, labelled with the order used", async () => {
    mockState = ready({
      result: { text: "Two people talk.", ms: 9000, encodeMs: 4000, tokens: 4 },
    });
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: CLIP }));
    fireEvent.click(screen.getByTestId("reverse-toggle"));
    fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));

    await waitFor(() =>
      expect(screen.getByTestId("filmstrip")).toBeInTheDocument(),
    );
    // Exactly the frames that were sampled, and the strip stays in sampled
    // order so a reversed run reads as the same frames the other way.
    const strip = within(screen.getByTestId("filmstrip"));
    expect(strip.getAllByRole("listitem")).toHaveLength(4);
    expect(screen.getByTestId("answer-asked")).toHaveTextContent(
      /4 frames, in reverse/i,
    );
  });

  it("labels the answer with the question it was actually asked", async () => {
    mockState = ready({
      result: { text: "Two people talk.", ms: 9000, encodeMs: 4000, tokens: 4 },
    });
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: CLIP }));
    fireEvent.change(screen.getByTestId("question-input"), {
      target: { value: "Who is speaking?" },
    });
    fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));
    await waitFor(() =>
      expect(screen.getByTestId("answer-asked")).toHaveTextContent(
        /Who is speaking\?/,
      ),
    );

    fireEvent.change(screen.getByTestId("question-input"), {
      target: { value: "Something else" },
    });
    expect(screen.getByTestId("answer-asked")).toHaveTextContent(
      /Who is speaking\?/,
    );
  });

  it("states that it is a frame sampler, beside the result", async () => {
    // A correctness requirement rather than decoration: shipping this as
    // "Video Text to Text" without it implies a temporal understanding the
    // model does not have.
    mockState = ready({
      result: { text: "Two people talk.", ms: 9000, encodeMs: 4000, tokens: 4 },
    });
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: CLIP }));
    fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));
    await waitFor(() =>
      expect(screen.getByTestId("sampler-note")).toHaveTextContent(
        /frame sampler/i,
      ),
    );
    expect(screen.getByTestId("sampler-note")).toHaveTextContent(
      /nothing between them reached the model/i,
    );
  });

  it("names the encode rather than showing an unlabelled spinner", () => {
    mockState = ready({ running: true, partial: { stage: "encoding" } });
    renderPage();
    expect(screen.getByTestId("answer-encoding")).toHaveTextContent(
      /encoding .* frames/i,
    );
  });

  it("reports the timings", async () => {
    mockState = ready({
      result: { text: "Two people talk.", ms: 9000, encodeMs: 4000, tokens: 4 },
    });
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("generate-ms")).toHaveTextContent(/9\.0s/),
    );
    expect(screen.getByTestId("slot-4")).toHaveTextContent(/4\.0s encoding/);
  });
});

describe("VideoTextToTextPage — gating and errors", () => {
  it("disables the model when the probe reports WASM", async () => {
    pickBackend.mockResolvedValue("wasm");
    renderPage();
    await waitFor(() =>
      expect(
        screen.getByTestId(`model-unsupported-${MODEL}`),
      ).toBeInTheDocument(),
    );
  });

  it("explains a missing shader-f16", async () => {
    supportsShaderF16.mockResolvedValue(false);
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("f16-note")).toHaveTextContent(/shader-f16/),
    );
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

  it("puts an inference failure in OUTPUT", () => {
    mockState = ready({ error: "out of memory" });
    renderPage();
    expect(screen.getByTestId("slot-4")).toContainElement(
      screen.getByText(/out of memory/i),
    );
  });

  it("surfaces a failed decode in the RUN slot, and runs nothing", async () => {
    mockState = ready();
    sampleVideo.mockRejectedValueOnce(new Error("Could not decode that video"));
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: CLIP }));
    fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));

    const note = await screen.findByText(/could not decode that video/i);
    expect(screen.getByTestId("slot-3")).toContainElement(note);
    expect(mockRun).not.toHaveBeenCalled();
  });
});
