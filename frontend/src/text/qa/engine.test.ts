import { describe, expect, it, vi } from "vitest";

import { answerFrom, createQaHandler, type QaEncoding, type QaReader } from "./engine";
import type { QaRequest, QaResponse } from "./types";

// A fake reader over a real tokenization. The ids are arbitrary but the
// **pieces** are the ones `Xenova/distilbert-base-cased-distilled-squad`
// actually emits for this passage, decoded one id at a time — an alignment
// test written against invented pieces pins our idea of WordPiece rather than
// the one the page runs.
const CLS = 101;
const SEP = 102;
const PAD = 0;

const CONTEXT = "Unlike WebGL, it supports general-purpose compute shaders.";
const CONTEXT_PIECES = [
  "Un", "##lik", "##e", "Web", "##GL", ",", "it", "supports", "general", "-",
  "purpose", "compute", "shade", "##rs", ".",
];
const QUESTION_PIECES = ["What", "does", "it", "support", "?"];

/** `[CLS] question [SEP] context [SEP]`, with ids that index the piece table. */
const PIECES = ["[CLS]", ...QUESTION_PIECES, "[SEP]", ...CONTEXT_PIECES, "[SEP]"];
const IDS = PIECES.map((_, i) => (i === 0 ? CLS : PIECES[i] === "[SEP]" ? SEP : i + 1000));
const CONTEXT_START = QUESTION_PIECES.length + 2; // past [CLS] … [SEP]

function pieceOf(id: number): string {
  if (id === CLS) return "[CLS]";
  if (id === SEP) return "[SEP]";
  if (id === PAD) return "[PAD]";
  return PIECES[id - 1000];
}

const reader: Pick<QaReader, "piece" | "decode" | "sepTokenId" | "specialIds"> = {
  sepTokenId: SEP,
  specialIds: [CLS, SEP, PAD],
  piece: pieceOf,
  // Approximates the WordPiece decoder: join pieces, attach continuations, and
  // space the rest — which is exactly what re-spaces `general-purpose`.
  decode: (ids) =>
    ids
      .map(pieceOf)
      .filter((p) => !["[CLS]", "[SEP]", "[PAD]"].includes(p))
      .reduce((acc, p) => (p.startsWith("##") ? acc + p.slice(2) : acc ? `${acc} ${p}` : p), ""),
};

/** Logits peaked on one token index. */
function peak(at: number, height = 12): number[] {
  return IDS.map((_, i) => (i === at ? height : 0));
}

function encodingFor(startToken: number, endToken: number): QaEncoding {
  return {
    ids: IDS,
    mask: IDS.map(() => 1),
    startLogits: peak(startToken),
    endLogits: peak(endToken),
  };
}

describe("answerFrom", () => {
  it("slices the answer out of the passage, at real character offsets", () => {
    const answer = answerFrom(
      encodingFor(CONTEXT_START + 8, CONTEXT_START + 13),
      CONTEXT,
      reader,
    );
    expect(answer.text).toBe("general-purpose compute shaders");
    expect(CONTEXT.slice(answer.start!, answer.end!)).toBe(answer.text);
    expect(answer.score).toBeGreaterThan(0.9);
    expect(answer.truncated).toBe(false);
  });

  // The single most important assertion in this file, and the reason the page
  // does not use `QuestionAnsweringPipeline`: the model's own decode of this
  // answer is not a substring of the passage. A page that rendered `decoded`,
  // or that located the highlight with `context.indexOf(decoded)`, would show
  // the wrong thing or nothing at all — with no error anywhere.
  it("keeps the tokenizer's decode separate, because it differs from the text", () => {
    const answer = answerFrom(
      encodingFor(CONTEXT_START + 8, CONTEXT_START + 13),
      CONTEXT,
      reader,
    );
    expect(answer.decoded).toBe("general - purpose compute shaders");
    expect(answer.decoded).not.toBe(answer.text);
    expect(CONTEXT.includes(answer.decoded)).toBe(false);
    expect(CONTEXT.includes(answer.text)).toBe(true);
  });

  it("marks a single-token answer", () => {
    const answer = answerFrom(
      encodingFor(CONTEXT_START + 7, CONTEXT_START + 7),
      CONTEXT,
      reader,
    );
    expect(answer.text).toBe("supports");
    expect(answer.start).toBe(CONTEXT.indexOf("supports"));
  });

  // No alignment, no highlight — never an approximate one.
  it("reports no offsets rather than a wrong highlight when alignment fails", () => {
    const answer = answerFrom(
      encodingFor(CONTEXT_START + 8, CONTEXT_START + 13),
      // A passage that is not the one these pieces came from.
      "A completely different sentence about something else.",
      reader,
    );
    expect(answer.start).toBeNull();
    expect(answer.end).toBeNull();
    expect(answer.text).toBe("");
    // The answer string survives, so the page still has something to show.
    expect(answer.decoded).toBe("general - purpose compute shaders");
    expect(answer.score).toBeGreaterThan(0);
  });

  it("flags a passage the model only saw part of", () => {
    const longer = `${CONTEXT} It also exposes a render pipeline.`;
    const answer = answerFrom(
      encodingFor(CONTEXT_START + 7, CONTEXT_START + 7),
      longer,
      reader,
    );
    // The tokenization stops at the original passage's full stop, so the tail
    // was never read — silent otherwise, and the page says so.
    expect(answer.truncated).toBe(true);
  });

  it("throws rather than inventing a span when there is nothing to answer from", () => {
    expect(() =>
      answerFrom(
        { ids: [CLS, 1001], mask: [1, 1], startLogits: [0, 0], endLogits: [0, 0] },
        CONTEXT,
        reader,
      ),
    ).toThrow(/no answerable span/i);
  });
});

/** The most recent message. `Array.prototype.at` is past this tsconfig's lib. */
function last(posted: QaResponse[]): QaResponse | undefined {
  return posted[posted.length - 1];
}

describe("createQaHandler", () => {
  function fakeReader(overrides: Partial<QaReader> = {}): QaReader {
    return {
      ...reader,
      ask: vi.fn(async () => encodingFor(CONTEXT_START + 7, CONTEXT_START + 7)),
      dispose: vi.fn(async () => {}),
      ...overrides,
    } as QaReader;
  }

  function harness(factoryReader: QaReader, opts?: { warmup?: boolean }) {
    const posted: QaResponse[] = [];
    const factory = vi.fn(async () => factoryReader);
    const handle = createQaHandler((m) => posted.push(m), factory, opts);
    return { posted, factory, handle };
  }

  const load: QaRequest = {
    type: "load",
    model: "Xenova/distilbert-base-cased-distilled-squad",
    opts: { device: "wasm", dtype: "q8" },
  };

  it("warms up before announcing ready, and never after", async () => {
    const r = fakeReader();
    const { posted, handle } = harness(r);
    await handle(load);

    expect(r.ask).toHaveBeenCalledTimes(1);
    const statuses = posted.map((m) => m.type);
    const warmup = posted.findIndex(
      (m) => m.type === "progress" && m.progress.status === "warmup",
    );
    expect(warmup).toBeGreaterThanOrEqual(0);
    expect(warmup).toBeLessThan(statuses.lastIndexOf("ready"));
    expect(last(posted)).toMatchObject({ type: "ready", backend: "wasm" });
  });

  it("does not fail the load when warm-up throws", async () => {
    const r = fakeReader({
      ask: vi.fn(async () => {
        throw new Error("shader compile failed");
      }) as QaReader["ask"],
    });
    const { posted, handle } = harness(r);
    await handle(load);
    expect(last(posted)).toMatchObject({ type: "ready" });
  });

  // The disposal *order* is the obligation: null the reference first, so a
  // teardown that throws cannot leave a stale model live.
  it("drops the previous reader before disposing it", async () => {
    const first = fakeReader();
    const second = fakeReader();
    const posted: QaResponse[] = [];
    const factory = vi
      .fn<() => Promise<QaReader>>()
      .mockResolvedValueOnce(first)
      .mockResolvedValueOnce(second);
    const handle = createQaHandler((m) => posted.push(m), factory);

    await handle(load);
    await handle(load);
    expect(first.dispose).toHaveBeenCalledOnce();
  });

  it("survives a reader whose teardown throws", async () => {
    const first = fakeReader({
      dispose: vi.fn(async () => {
        throw new Error("backend already gone");
      }),
    });
    const second = fakeReader();
    const posted: QaResponse[] = [];
    const factory = vi
      .fn<() => Promise<QaReader>>()
      .mockResolvedValueOnce(first)
      .mockResolvedValueOnce(second);
    const handle = createQaHandler((m) => posted.push(m), factory);

    await handle(load);
    await handle(load);
    expect(last(posted)).toMatchObject({ type: "ready" });
  });

  it("answers a run, correlated to its request id", async () => {
    const { posted, handle } = harness(fakeReader(), { warmup: false });
    await handle(load);
    await handle({
      type: "run",
      id: 7,
      question: "What does it support?",
      context: CONTEXT,
    });

    const result = last(posted);
    expect(result).toMatchObject({ type: "result", id: 7 });
    expect((result as { result: { text: string } }).result.text).toBe("supports");
  });

  // The two halves of the envelope's error contract: a load failure carries no
  // id (Machine A), a run failure carries one (Machine B stays `ready`).
  it("reports a load failure without an id", async () => {
    const posted: QaResponse[] = [];
    const handle = createQaHandler(
      (m) => posted.push(m),
      vi.fn(async () => {
        throw new Error("404 model not found");
      }),
    );
    await handle(load);
    expect(last(posted)).toEqual({ type: "error", error: "404 model not found" });
  });

  it("reports a run failure against its id", async () => {
    const { posted, handle } = harness(fakeReader(), { warmup: false });
    await handle({
      type: "run",
      id: 3,
      question: "What does it support?",
      context: CONTEXT,
    });
    expect(last(posted)).toMatchObject({ type: "error", id: 3 });
  });

  it("refuses an empty passage rather than answering from nothing", async () => {
    const { posted, handle } = harness(fakeReader(), { warmup: false });
    await handle(load);
    await handle({ type: "run", id: 1, question: "Who?", context: "   " });
    expect(last(posted)).toMatchObject({ type: "error", id: 1 });
    expect((last(posted) as { error: string }).error).toMatch(/no passage/i);
  });

  it("passes a per-backend precision pin through to the factory", async () => {
    const { factory, handle } = harness(fakeReader(), { warmup: false });
    await handle({ ...load, dtypes: { wasm: "fp32" } });
    expect(factory).toHaveBeenCalledWith(
      load.model,
      expect.objectContaining({ device: "wasm", dtype: "fp32" }),
    );
  });
});
