import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { UseVlmResult } from "@/hooks/useVlm";

const fakeImage = { width: 8, height: 8, channels: 3, data: [] } as never;
const fromFile = vi.fn().mockResolvedValue(fakeImage);
const fromUrl = vi.fn().mockResolvedValue(fakeImage);
const downscale = vi.fn(async (img: unknown) => img);
vi.mock("@/vision/image", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    fromFile: (...a: unknown[]) => fromFile(...a),
    fromUrl: (...a: unknown[]) => fromUrl(...a),
    downscale: (...a: unknown[]) => downscale(...(a as [never])),
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

// The browser cache probe. Empty unless a test seeds it.
let cached = new Set<string>();
vi.mock("@/model/cache", () => ({
  cachedModels: () => Promise.resolve(cached),
  evictModel: vi.fn(() => Promise.resolve()),
}));

const { Route } = await import("@/routes/visual-question-answering");
const { useModelPrefs } = await import("@/store/models");
const { VLM_MODELS } = await import("@/multimodal/types");
const SMALL = VLM_MODELS[0].id;
const LARGE = VLM_MODELS[1].id;
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Visual-question-answering route not found");
  return render(<Page />);
}

const RUN_BUTTON = /^generate$/i;
const SAMPLE = /^tiger$/i;

const ready = (extra: Partial<UseVlmResult> = {}): UseVlmResult => ({
  ...baseState(),
  status: "ready",
  idle: false,
  ready: true,
  ...extra,
});

async function pick() {
  fireEvent.click(screen.getByRole("button", { name: SAMPLE }));
  await waitFor(() =>
    expect(screen.getByRole("button", { name: RUN_BUTTON })).toBeEnabled(),
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  mockState = baseState();
  cached = new Set();
  useModelPrefs.setState({ selected: {} });
  mockRun.mockResolvedValue({
    text: "Tiger",
    ms: 2100,
    encodeMs: 900,
    tokens: 2,
  });
  pickBackend.mockResolvedValue("webgpu");
  supportsShaderF16.mockResolvedValue(true);
  URL.createObjectURL = vi.fn(() => "blob:preview");
  URL.revokeObjectURL = vi.fn();
});
afterEach(() => vi.clearAllMocks());

describe("VisualQuestionAnsweringPage — the four-slot contract", () => {
  it("renders the heading and the shared catalogue", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /visual question answering/i }),
    ).toBeInTheDocument();
    // The *same* catalogue as /image-text-to-text — no second one.
    expect(
      screen.getByRole("button", { name: /smolvlm 256m/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /smolvlm 500m/i }),
    ).toBeInTheDocument();
  });

  it("renders all four slots with an empty OUTPUT before any run", () => {
    renderPage();
    expect(screen.getAllByRole("region")).toHaveLength(4);
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("downloads nothing on arrival, and loads only on request", () => {
    renderPage();
    // No `autoLoad` argument at all: the hook's own default is `idle`.
    expect(useVlm).toHaveBeenCalledWith("HuggingFaceTB/SmolVLM-256M-Instruct");
    expect(mockLoad).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /load model/i }));
    expect(mockLoad).toHaveBeenCalledOnce();
  });

  it("keeps GENERATE disabled until ready, while the input sources stay live", async () => {
    renderPage();
    expect(screen.getByRole("button", { name: RUN_BUTTON })).toBeDisabled();
    expect(screen.getByRole("button", { name: SAMPLE })).toBeEnabled();
    expect(screen.getByTestId("terse-toggle")).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: SAMPLE }));
    await waitFor(() => expect(fromUrl).toHaveBeenCalled());
    expect(mockRun).not.toHaveBeenCalled();
  });
});

describe("VisualQuestionAnsweringPage — nothing runs until asked", () => {
  it("picks a sample without running, then generates when asked, on a capped frame", async () => {
    mockState = ready();
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: SAMPLE }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: RUN_BUTTON })).toBeEnabled(),
    );
    expect(mockRun).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    expect(downscale).toHaveBeenCalledWith(fakeImage, 512);
  });

  it("flipping the terse toggle runs nothing — it only changes what the next run sends", async () => {
    // The assertion this page exists for. The toggle *looks* like a filter,
    // which is exactly why it must not behave like one.
    mockState = ready();
    renderPage();
    await pick();

    fireEvent.click(screen.getByTestId("terse-toggle"));
    expect(mockRun).not.toHaveBeenCalled();
    fireEvent.click(screen.getByTestId("terse-toggle"));
    expect(mockRun).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("editing the question and tapping a preset run nothing", async () => {
    mockState = ready();
    renderPage();
    await pick();

    fireEvent.change(screen.getByTestId("question-input"), {
      target: { value: "What colour is it?" },
    });
    const presets = within(screen.getByTestId("presets"));
    fireEvent.click(presets.getByRole("button", { name: /how many people/i }));
    expect(mockRun).not.toHaveBeenCalled();
    expect(screen.getByTestId("question-input")).toHaveValue(
      "How many people are there?",
    );
  });

  it("survives its run: two generations cost one decode", async () => {
    mockState = ready();
    renderPage();
    await pick();

    fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(2));

    expect(fromUrl).toHaveBeenCalledTimes(1);
  });

  it("will not run on an empty question", async () => {
    mockState = ready();
    renderPage();
    await pick();

    fireEvent.change(screen.getByTestId("question-input"), {
      target: { value: "   " },
    });
    expect(screen.getByRole("button", { name: RUN_BUTTON })).toBeDisabled();
  });
});

describe("VisualQuestionAnsweringPage — the prompt is visible before it is sent", () => {
  it("shows the raw question with the toggle off", () => {
    mockState = ready();
    renderPage();
    fireEvent.change(screen.getByTestId("question-input"), {
      target: { value: "What animal is this?" },
    });
    expect(screen.getByTestId("composed-prompt")).toHaveTextContent(
      "What animal is this?",
    );
    expect(screen.getByTestId("composed-prompt")).not.toHaveTextContent(
      /answer in one word/i,
    );
  });

  it("shows the composed prompt with the toggle on, and sends exactly that", async () => {
    mockState = ready();
    renderPage();
    await pick();
    fireEvent.change(screen.getByTestId("question-input"), {
      target: { value: "What animal is this?" },
    });
    fireEvent.click(screen.getByTestId("terse-toggle"));

    expect(screen.getByTestId("composed-prompt")).toHaveTextContent(
      "What animal is this? Answer in one word.",
    );

    fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));
    // The displayed string and the sent string are the same string, and the
    // short cap travels with it.
    await waitFor(() =>
      expect(mockRun).toHaveBeenCalledWith(
        fakeImage,
        "What animal is this? Answer in one word.",
        16,
      ),
    );
  });

  it("does not instruct a question that already asks for one word", () => {
    mockState = ready();
    renderPage();
    fireEvent.change(screen.getByTestId("question-input"), {
      target: { value: "Answer in one word: what animal is this?" },
    });
    fireEvent.click(screen.getByTestId("terse-toggle"));
    expect(screen.getByTestId("composed-prompt")).toHaveTextContent(
      "Answer in one word: what animal is this?",
    );
    expect(screen.getByTestId("slot-3")).toHaveTextContent(
      /you already asked for a short answer/i,
    );
  });

  it("sends the full cap when the toggle is off", async () => {
    mockState = ready();
    renderPage();
    await pick();
    fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));
    await waitFor(() =>
      expect(mockRun).toHaveBeenCalledWith(
        fakeImage,
        "What animal is this?",
        128,
      ),
    );
  });
});

describe("VisualQuestionAnsweringPage — the answer", () => {
  it("labels the answer with the composed prompt, not the raw box", async () => {
    mockState = ready({
      result: { text: "Tiger", ms: 2100, encodeMs: 900, tokens: 2 },
    });
    renderPage();
    await pick();
    fireEvent.change(screen.getByTestId("question-input"), {
      target: { value: "What animal is this?" },
    });
    fireEvent.click(screen.getByTestId("terse-toggle"));
    fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));

    await waitFor(() =>
      expect(screen.getByTestId("answer-asked")).toHaveTextContent(
        "Asked: What animal is this? Answer in one word.",
      ),
    );

    // Flipping the toggle afterwards must not relabel a result on screen.
    fireEvent.click(screen.getByTestId("terse-toggle"));
    expect(screen.getByTestId("answer-asked")).toHaveTextContent(
      "Asked: What animal is this? Answer in one word.",
    );
  });

  it("names the encode rather than showing an unlabelled spinner", () => {
    mockState = ready({ running: true, partial: { stage: "encoding" } });
    renderPage();
    expect(screen.getByTestId("answer-encoding")).toHaveTextContent(
      /encoding the image/i,
    );
  });

  it("streams the answer as it arrives", () => {
    mockState = ready({
      running: true,
      partial: { stage: "generating", text: "Tig" },
    });
    renderPage();
    expect(screen.getByTestId("answer-text")).toHaveTextContent("Tig");
  });

  it("reports the timings", async () => {
    mockState = ready({
      result: { text: "Tiger", ms: 2100, encodeMs: 900, tokens: 2 },
    });
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("generate-ms")).toHaveTextContent(/2\.1s/),
    );
    expect(screen.getByTestId("slot-4")).toHaveTextContent(/0\.9s encoding/);
    expect(screen.getByTestId("slot-4")).toHaveTextContent(/2 tokens/);
  });
});

describe("VisualQuestionAnsweringPage — after a page refresh", () => {
  it("comes back on the model the user had selected", () => {
    // Its own `routeKey`: the two VLM pages share a catalogue, not a choice.
    useModelPrefs.setState({ selected: { "visual-question-answering": LARGE } });
    renderPage();
    expect(
      screen.getByRole("button", { name: /smolvlm 500m/i }),
    ).toHaveAttribute("aria-pressed", "true");
  });

  it("does not load a cached model, even one loaded on the sibling page", async () => {
    // The catalogue is shared, so the weights often *are* already cached here
    // — which makes the click cheap, not unnecessary.
    cached = new Set([SMALL]);
    renderPage();
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /load model \(cached\)/i }),
      ).toBeEnabled(),
    );
    expect(useVlm).toHaveBeenLastCalledWith(SMALL);
    expect(mockLoad).not.toHaveBeenCalled();
  });
});

describe("VisualQuestionAnsweringPage — gating and errors", () => {
  it("disables both models when the probe reports WASM", async () => {
    pickBackend.mockResolvedValue("wasm");
    renderPage();
    await waitFor(() =>
      expect(
        screen.getByTestId(
          "model-unsupported-HuggingFaceTB/SmolVLM-256M-Instruct",
        ),
      ).toBeInTheDocument(),
    );
    expect(screen.getByRole("button", { name: /smolvlm 256m/i })).toBeDisabled();
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

  it("puts an inference failure in OUTPUT, where the model stays loaded", () => {
    mockState = ready({ error: "out of memory" });
    renderPage();
    expect(screen.getByTestId("slot-4")).toContainElement(
      screen.getByText(/out of memory/i),
    );
  });

  it("surfaces a failed decode in the RUN slot, not the output", async () => {
    mockState = ready();
    fromUrl.mockRejectedValueOnce(new Error("Unsupported image type"));
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: SAMPLE }));
    const note = await screen.findByText(/unsupported image type/i);
    expect(screen.getByTestId("slot-3")).toContainElement(note);
    expect(mockRun).not.toHaveBeenCalled();
  });
});
