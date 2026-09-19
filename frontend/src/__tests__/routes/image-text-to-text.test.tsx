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

const { Route } = await import("@/routes/image-text-to-text");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Image-text-to-text route component not found");
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
  mockRun.mockResolvedValue({
    text: "A tiger lying in grass.",
    ms: 4200,
    encodeMs: 1100,
    tokens: 7,
  });
  pickBackend.mockResolvedValue("webgpu");
  URL.createObjectURL = vi.fn(() => "blob:preview");
  URL.revokeObjectURL = vi.fn();
});
afterEach(() => vi.clearAllMocks());

describe("ImageTextToTextPage — the four-slot contract", () => {
  it("renders the heading and both model options", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /image text to text/i }),
    ).toBeInTheDocument();
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
    // Choosing an input is free and commits to nothing, so it is not gated.
    expect(screen.getByRole("button", { name: SAMPLE })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: SAMPLE }));
    await waitFor(() => expect(fromUrl).toHaveBeenCalled());
    expect(mockRun).not.toHaveBeenCalled();
  });
});

describe("ImageTextToTextPage — nothing runs until asked", () => {
  it("picks a sample without running, then generates when asked, on a capped frame", async () => {
    mockState = ready();
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: SAMPLE }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: RUN_BUTTON })).toBeEnabled(),
    );
    // The single most valuable assertion on this page: an input that silently
    // starts an inference looks exactly like a working page.
    expect(mockRun).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    // 512, SmolVLM's own tile size — above it the processor splits the picture
    // into tiles that each cost their own image tokens.
    expect(downscale).toHaveBeenCalledWith(fakeImage, 512);
  });

  it("editing the question runs nothing", async () => {
    mockState = ready();
    renderPage();
    await pick();

    fireEvent.change(screen.getByTestId("prompt-input"), {
      target: { value: "What colour is it?" },
    });
    expect(mockRun).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));
    await waitFor(() =>
      expect(mockRun).toHaveBeenCalledWith(fakeImage, "What colour is it?"),
    );
  });

  it("a preset fills the box and runs nothing", async () => {
    mockState = ready();
    renderPage();
    await pick();

    const presets = within(screen.getByTestId("presets"));
    fireEvent.click(presets.getByRole("button", { name: /how many people/i }));
    expect(mockRun).not.toHaveBeenCalled();
    expect(screen.getByTestId("prompt-input")).toHaveValue(
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

    fireEvent.change(screen.getByTestId("prompt-input"), {
      target: { value: "   " },
    });
    expect(screen.getByRole("button", { name: RUN_BUTTON })).toBeDisabled();
  });
});

describe("ImageTextToTextPage — the pre-token pause", () => {
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
      partial: { stage: "generating", text: "A tiger ly" },
    });
    renderPage();
    expect(screen.getByTestId("answer-text")).toHaveTextContent("A tiger ly");
  });

  it("never shows a partial beside a finished answer", () => {
    // The hook clears `partial` on the result; the view prefers it while it
    // exists, so the two can't both be on screen.
    mockState = ready({
      partial: null,
      result: { text: "A tiger lying in grass.", ms: 4200, encodeMs: 1100, tokens: 7 },
    });
    renderPage();
    expect(screen.getByTestId("answer-text")).toHaveTextContent(
      "A tiger lying in grass.",
    );
    expect(screen.queryByTestId("answer-encoding")).not.toBeInTheDocument();
  });

  it("says a run takes seconds before the first one", () => {
    renderPage();
    expect(screen.getByTestId("slot-3")).toHaveTextContent(
      /encoded before the first word/i,
    );
  });
});

describe("ImageTextToTextPage — the answer", () => {
  it("reports the timings, including how long the blind wait was", async () => {
    mockState = ready({
      result: { text: "A tiger.", ms: 4200, encodeMs: 1100, tokens: 7 },
    });
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("generate-ms")).toHaveTextContent(/4\.2s/),
    );
    expect(screen.getByTestId("slot-4")).toHaveTextContent(/1\.1s encoding/);
    expect(screen.getByTestId("slot-4")).toHaveTextContent(/7 tokens/);
  });

  it("labels the answer with the question it was actually asked", async () => {
    // The hook is mocked, so its `result` is supplied rather than produced;
    // `asked` is the piece the page itself captures during the run.
    mockState = ready({
      result: { text: "A tiger.", ms: 4200, encodeMs: 1100, tokens: 7 },
    });
    renderPage();
    await pick();
    fireEvent.change(screen.getByTestId("prompt-input"), {
      target: { value: "What animal is this?" },
    });
    fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));

    await waitFor(() =>
      expect(screen.getByTestId("answer-asked")).toHaveTextContent(
        /What animal is this\?/,
      ),
    );

    // Editing the box afterwards must not relabel the result on screen.
    fireEvent.change(screen.getByTestId("prompt-input"), {
      target: { value: "Something else entirely" },
    });
    expect(screen.getByTestId("answer-asked")).toHaveTextContent(
      /What animal is this\?/,
    );
  });

  it("warns that a small VLM answers confidently either way", () => {
    mockState = ready({
      result: { text: "A tiger.", ms: 4200, encodeMs: 1100, tokens: 7 },
    });
    renderPage();
    expect(screen.getByTestId("slot-4")).toHaveTextContent(
      /answer confidently whether or not/i,
    );
  });

  it("explains an empty generation instead of showing a blank panel", () => {
    mockState = ready({
      result: { text: "", ms: 900, encodeMs: 700, tokens: 0 },
    });
    renderPage();
    expect(screen.getByTestId("slot-4")).toHaveTextContent(/generated nothing/i);
  });
});

describe("ImageTextToTextPage — backend gating and errors", () => {
  it("disables both models when the probe reports WASM", async () => {
    // A decoder on WASM is seconds per token. The limitation is known before
    // the click, so offering it would only buy a failed 189 MB download.
    pickBackend.mockResolvedValue("wasm");
    renderPage();
    await waitFor(() =>
      expect(
        screen.getByTestId("model-unsupported-HuggingFaceTB/SmolVLM-256M-Instruct"),
      ).toBeInTheDocument(),
    );
    expect(screen.getByRole("button", { name: /smolvlm 256m/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /smolvlm 500m/i })).toBeDisabled();
  });

  it("gates nothing while the probe is still undecided", () => {
    // `null` means "ask again", not "no GPU".
    renderPage();
    expect(
      screen.queryByTestId("model-unsupported-HuggingFaceTB/SmolVLM-256M-Instruct"),
    ).not.toBeInTheDocument();
  });

  it("puts a load failure in the LOAD slot, not in OUTPUT", () => {
    mockState = { ...baseState(), status: "error", idle: false, error: "404 not found" };
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
