import { beforeEach, describe, expect, it, vi, type Mock } from "vitest";

import type { ImagePayload } from "../image";
import { coverage, SamSession, type SamTowers } from "./sam";
import type { SamMask } from "./types";

const image = (width = 4, height = 4): ImagePayload => ({
  data: new Uint8ClampedArray(width * height * 3),
  width,
  height,
  channels: 3,
});

const mask = (bytes: number[], width = 2, height = 2): SamMask => ({
  data: Uint8Array.from(bytes),
  width,
  height,
  score: 0.9,
});

let encode: Mock<SamTowers["encode"]>;
let decode: Mock<SamTowers["decode"]>;
let towers: SamTowers;

beforeEach(() => {
  encode = vi.fn<SamTowers["encode"]>(async () => ({ embeddings: "E" }));
  decode = vi.fn<SamTowers["decode"]>(async () => [mask([1, 0, 0, 0])]);
  towers = { encode, decode };
});

describe("SamSession", () => {
  it("encodes once for a token, and reuses it on a repeat", async () => {
    // The whole point of the module: a second click must not re-run the vision
    // encoder, which is the slow half.
    const session = new SamSession(towers);
    const first = await session.encode("a", image());
    const second = await session.encode("a", image());

    expect(encode).toHaveBeenCalledTimes(1);
    expect(first.cached).toBe(false);
    expect(second.cached).toBe(true);
  });

  it("re-encodes when the token changes", async () => {
    const session = new SamSession(towers);
    await session.encode("a", image());
    const next = await session.encode("b", image());

    expect(encode).toHaveBeenCalledTimes(2);
    expect(next.cached).toBe(false);
  });

  it("reports the encoded image's size, cached or not", async () => {
    const session = new SamSession(towers);
    const first = await session.encode("a", image(8, 6));
    const second = await session.encode("a", image(8, 6));
    expect(first).toMatchObject({ width: 8, height: 6 });
    expect(second).toMatchObject({ width: 8, height: 6, cached: true });
  });

  it("decodes against the cached embedding without touching the encoder", async () => {
    const session = new SamSession(towers);
    await session.encode("a", image());
    await session.decode([{ x: 1, y: 1, positive: true }]);
    await session.decode([{ x: 2, y: 2, positive: true }]);

    expect(encode).toHaveBeenCalledTimes(1);
    expect(decode).toHaveBeenCalledTimes(2);
    expect(decode.mock.calls[0][0]).toEqual({ embeddings: "E" });
  });

  it("passes positive and negative points through as given", async () => {
    // 1 means "the object", 0 means "not the object". Swapping them inverts
    // every mask a second click was meant to refine.
    const session = new SamSession(towers);
    await session.encode("a", image());
    await session.decode([
      { x: 1, y: 1, positive: true },
      { x: 3, y: 3, positive: false },
    ]);
    expect(decode.mock.calls[0][1]).toEqual([
      { x: 1, y: 1, positive: true },
      { x: 3, y: 3, positive: false },
    ]);
  });

  it("refuses to decode before anything is encoded", async () => {
    // A lazy re-encode here would hide the cost this module exists to avoid,
    // and the page's "encoding" state would stop meaning anything.
    const session = new SamSession(towers);
    await expect(session.decode([{ x: 1, y: 1, positive: true }])).rejects.toThrow(
      /encode an image/i,
    );
    expect(encode).not.toHaveBeenCalled();
  });

  it("refuses to decode with no points", async () => {
    const session = new SamSession(towers);
    await session.encode("a", image());
    await expect(session.decode([])).rejects.toThrow(/at least one point/i);
    expect(decode).not.toHaveBeenCalled();
  });

  it("drops the embedding when cleared", async () => {
    // A SAM embedding is 256x64x64 floats — 4 MB at fp32 — held for the tab's
    // lifetime otherwise.
    const session = new SamSession(towers);
    await session.encode("a", image());
    expect(session.ready).toBe(true);

    session.clear();
    expect(session.ready).toBe(false);
    await expect(session.decode([{ x: 1, y: 1, positive: true }])).rejects.toThrow();

    // And the same token now costs a real encode again.
    const again = await session.encode("a", image());
    expect(again.cached).toBe(false);
  });

  it("leaves nothing decodable when an encode throws", async () => {
    // Null-first: a failed encode must not leave the previous image's embedding
    // live to be decoded against clicks on a picture nobody is looking at.
    const session = new SamSession(towers);
    await session.encode("a", image());
    encode.mockRejectedValueOnce(new Error("out of memory"));

    await expect(session.encode("b", image())).rejects.toThrow(/out of memory/);
    expect(session.ready).toBe(false);
  });
});

describe("coverage", () => {
  it("measures the fraction of the frame a mask covers", () => {
    expect(coverage(mask([1, 1, 0, 0]))).toBeCloseTo(0.5, 6);
    expect(coverage(mask([0, 0, 0, 0]))).toBe(0);
    expect(coverage(mask([1, 1, 1, 1]))).toBe(1);
  });

  it("is zero for an empty mask rather than NaN", () => {
    expect(coverage({ data: new Uint8Array(), width: 0, height: 0, score: 0 })).toBe(
      0,
    );
  });
});
