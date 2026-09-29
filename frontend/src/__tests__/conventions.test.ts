// The task-page conventions from #58, checked against the source.
//
// Each helper (useTaskSlots, RunButton, useCatalogueEntry, the route-test
// contract) replaced a pattern that had been copied 20–35 times. Nothing about
// a copy fails when a new route writes the old pattern again — it renders, and
// its tests pass — so the drift would be invisible until the next audit. These
// assertions make the old pattern fail in CI instead, with the file named.
//
// They read source text, deliberately: the property is "how the file is
// written", which no render can observe. Each one is narrow enough that a
// legitimate exception has to be added to its list below, by name.

import { describe, expect, it } from "vitest";

const routes = import.meta.glob<string>("../routes/*.tsx", {
  query: "?raw",
  import: "default",
  eager: true,
});
const routeTests = import.meta.glob<string>("./routes/*.test.tsx", {
  query: "?raw",
  import: "default",
  eager: true,
});
const hooks = import.meta.glob<string>("../hooks/*.ts", {
  query: "?raw",
  import: "default",
  eager: true,
});

const name = (path: string) => path.split("/").pop()!.replace(/\.(test\.)?tsx?$/, "");
const entries = (files: Record<string, string>) =>
  Object.entries(files).map(([path, src]) => [name(path), src] as const);

/** A route that lets the user pick a model to download is a task page. */
const taskRoutes = entries(routes).filter(([, src]) => src.includes("useModelSelection("));

describe("task-page conventions (#58)", () => {
  it("finds the routes it is meant to check", () => {
    // Guards the globs themselves: a moved directory would make every
    // assertion below pass vacuously.
    expect(Object.keys(routes).length).toBeGreaterThan(30);
    expect(taskRoutes.length).toBeGreaterThan(25);
    expect(Object.keys(hooks).length).toBeGreaterThan(20);
  });

  it("wires SELECT and LOAD through useTaskSlots on every task route", () => {
    const missing = taskRoutes
      .filter(([, src]) => !src.includes("useTaskSlots("))
      .map(([n]) => n);
    expect(missing).toEqual([]);
  });

  it("leaves the error split and the cache re-probe to useTaskSlots", () => {
    const handWritten = taskRoutes
      .filter(
        ([, src]) =>
          /\bconst loadError = [\w.]*status === "error"/.test(src) ||
          src.includes("useCacheRefresh("),
      )
      .map(([n]) => n);
    expect(handWritten).toEqual([]);
  });

  it("uses RunButton for a trigger, not a hand-written spinner button", () => {
    // The pattern RunButton replaced: a Button whose children switch between a
    // spinner + running label and an icon + verb on a run/training flag.
    // `/asr`'s Upload button spins on `decoding` — an input source, not a
    // trigger — so the flag is what the check keys on.
    // The props may hold `=>`, so they are matched up to the next <Button
    // rather than to the first ">".
    const spinnerTrigger =
      /<Button\b(?:(?!<\/?Button)[\s\S])*?>\s*\{[^?{}]*\b(running|training)\b[^?{}]*\?\s*\(\s*<>\s*<Loader2/;
    const handWritten = entries(routes)
      .filter(([, src]) => spinnerTrigger.test(src))
      .map(([n]) => n);
    expect(handWritten).toEqual([]);
  });

  it("looks catalogue entries up with useCatalogueEntry, not by hand", () => {
    const handWritten = entries(hooks)
      .filter(([n]) => !n.endsWith(".test"))
      .filter(([, src]) => /\.find\(\(m\) => m\.id === \w+\)\s*\?\?/.test(src))
      .map(([n]) => n);
    expect(handWritten).toEqual([]);
  });

  it("registers the §8 contract in every task route's test", () => {
    const tests = new Map(entries(routeTests));
    const missing = taskRoutes
      .map(([n]) => n)
      .filter((n) => !tests.get(n)?.includes("describeTaskPageContract("));
    expect(missing).toEqual([]);
  });
});
