import { describe, expect, it, vi } from "vitest";

import { parseCsvInWorker } from "./client";
import type { ParseResult } from "./types";

/** A stand-in Worker whose reply the test drives. */
class FakeWorker {
  static last: FakeWorker;
  onmessage: ((e: MessageEvent<unknown>) => void) | null = null;
  onerror: ((e: ErrorEvent) => void) | null = null;
  posted: unknown[] = [];
  terminated = 0;

  constructor() {
    FakeWorker.last = this;
  }
  postMessage(message: unknown) {
    this.posted.push(message);
  }
  terminate() {
    this.terminated++;
  }
  reply(data: unknown) {
    this.onmessage?.({ data } as MessageEvent<unknown>);
  }
  fail(message: string) {
    this.onerror?.({ message } as ErrorEvent);
  }
}

const spawn = () => new FakeWorker() as unknown as Worker;

const result = { dataset: { name: "x", columns: [], rowCount: 0, sourceRowCount: 0, sampled: false }, issues: [] } as unknown as ParseResult;

describe("parseCsvInWorker", () => {
  it("posts the text and the options through to the worker", async () => {
    const promise = parseCsvInWorker("a,b\n1,2\n", { name: "f.csv", maxRows: 100 }, spawn);
    expect(FakeWorker.last.posted[0]).toEqual({
      id: 1,
      text: "a,b\n1,2\n",
      name: "f.csv",
      maxRows: 100,
    });
    FakeWorker.last.reply({ id: 1, ok: true, result });
    await expect(promise).resolves.toBe(result);
  });

  it("resolves with the same ParseResult the pure parser returns", async () => {
    // The worker exists to keep a ten-megabyte parse off the main thread, not to
    // change the contract.
    const promise = parseCsvInWorker("a\n1\n", {}, spawn);
    FakeWorker.last.reply({ id: 1, ok: true, result });
    expect(await promise).toBe(result);
  });

  it("rejects with the parser's own message, not a generic one", async () => {
    const promise = parseCsvInWorker("", {}, spawn);
    FakeWorker.last.reply({ id: 1, ok: false, error: "The file is empty." });
    await expect(promise).rejects.toThrow("The file is empty.");
  });

  it("rejects rather than hanging when the worker itself fails", async () => {
    // A worker that throws before it can reply would otherwise leave the page
    // stuck on "Parsing…" forever.
    const promise = parseCsvInWorker("a\n1\n", {}, spawn);
    FakeWorker.last.fail("Script error");
    await expect(promise).rejects.toThrow("Script error");
  });

  it("still rejects when the failure carries no message", async () => {
    const promise = parseCsvInWorker("a\n1\n", {}, spawn);
    FakeWorker.last.fail("");
    await expect(promise).rejects.toThrow(/parser worker failed/i);
  });

  it("terminates the worker on success and on failure", async () => {
    // A parse is a single event in a session; a resident worker per file would be
    // a leak with a nicer name.
    const ok = parseCsvInWorker("a\n1\n", {}, spawn);
    const okWorker = FakeWorker.last;
    okWorker.reply({ id: 1, ok: true, result });
    await ok;
    expect(okWorker.terminated).toBe(1);

    const bad = parseCsvInWorker("", {}, spawn);
    const badWorker = FakeWorker.last;
    badWorker.reply({ id: 1, ok: false, error: "nope" });
    await expect(bad).rejects.toThrow();
    expect(badWorker.terminated).toBe(1);
  });

  it("gives each call its own worker, so two parses cannot interleave", async () => {
    const a = parseCsvInWorker("a\n1\n", { name: "a" }, spawn);
    const first = FakeWorker.last;
    const b = parseCsvInWorker("b\n2\n", { name: "b" }, spawn);
    const second = FakeWorker.last;
    expect(first).not.toBe(second);

    // Answering out of order settles the right promise, because there is no
    // shared table to get wrong.
    second.reply({ id: 1, ok: true, result });
    await expect(b).resolves.toBe(result);
    first.reply({ id: 1, ok: false, error: "a failed" });
    await expect(a).rejects.toThrow("a failed");
  });

  it("does not construct a worker until it is called", () => {
    const factory = vi.fn(spawn);
    expect(factory).not.toHaveBeenCalled();
    void parseCsvInWorker("a\n1\n", {}, factory).catch(() => {});
    expect(factory).toHaveBeenCalledOnce();
  });
});
