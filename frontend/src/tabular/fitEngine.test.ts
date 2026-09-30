import { describe, expect, it, vi } from "vitest";

import { cpuMatmul } from "@/webgpu/linearModel";

import { parseCsv } from "./csv";
import { FAMILIES, familyInfo, REGRESSION_FAMILIES } from "./families";
import { createFitHandler } from "./fitEngine";
import { SAMPLES } from "./samples";
import type {
  Family,
  FitResult,
  PredictResult,
  TabularResponse,
} from "./types";

const deps = {
  // Rejecting is the honest stand-in for a machine with no WebGPU, and it is
  // the path a CI runner takes. The engine must still fit the whole ladder.
  gpuMatmul: () => Promise.reject(new Error("no device")),
  cpuMatmul: async (
    a: Float32Array,
    b: Float32Array,
    m: number,
    k: number,
    n: number,
  ) => cpuMatmul(a, b, m, k, n),
};

function harness() {
  const messages: TabularResponse[] = [];
  const handle = createFitHandler((m) => messages.push(m), deps);
  return { messages, handle };
}

const CSV = `x,y,label
1,1,a
2,1,a
3,1,a
1,9,b
2,9,b
3,9,b
1,2,a
2,2,a
3,8,b
1,8,b
2,8,b
3,2,a
`;

async function fitOn(csv: string, family: Family, extra: Record<string, unknown> = {}) {
  const { messages, handle } = harness();
  const { dataset } = parseCsv(csv, { name: "test" });
  await handle({ type: "load", dataset });
  const target = dataset.columns.length - 1;
  await handle({
    type: "run",
    id: 1,
    spec: {
      family,
      objective: "classification",
      targetIndex: target,
      featureIndices: dataset.columns.map((_, i) => i).filter((i) => i !== target),
      hp: familyInfo(family).defaults,
      seed: 42,
      testFraction: 0.3,
      ...extra,
    },
  });
  const result = messages.find((m) => m.type === "result");
  const error = messages.find((m) => m.type === "error");
  return { messages, handle, dataset, result, error };
}

describe("createFitHandler", () => {
  it("posts ready on load and carries the dataset name", async () => {
    const { messages, handle } = harness();
    const { dataset } = parseCsv(CSV, { name: "test" });
    await handle({ type: "load", dataset });
    expect(messages).toEqual([
      { type: "ready", model: "test", backend: "wasm" },
    ]);
  });

  it("fits every family on the ladder and beats the baseline", async () => {
    for (const family of FAMILIES) {
      const { result } = await fitOn(CSV, family.id);
      expect(result?.type, family.id).toBe("result");
      const fit = (result as { result: FitResult }).result;
      expect(fit.classification).toBeDefined();
      expect(fit.classification?.accuracy).toBeGreaterThanOrEqual(
        fit.classification?.baselineAccuracy ?? 1,
      );
    }
  });

  it("reports determinate iteration progress, never a byte count", async () => {
    const { messages } = await fitOn(CSV, "boosting");
    const partials = messages.filter((m) => m.type === "partial");
    expect(partials.length).toBeGreaterThan(0);
    for (const p of partials) {
      if (p.type !== "partial") continue;
      // A fit's total is known before it starts — epochs, trees and rows are
      // hyperparameters the user just set. `model/progress.ts`'s indeterminate
      // mode is the obvious reach and would be worse than the number we have.
      expect(p.partial.total).toBeGreaterThan(0);
      expect(p.partial.done).toBeLessThanOrEqual(p.partial.total);
      expect(p.partial.phase).toBeTruthy();
    }
    // Correlated to the request id, as the shared envelope requires.
    expect(partials.every((p) => p.type === "partial" && p.id === 1)).toBe(true);
  });

  it("reports the compute it actually used, not the one it wanted", async () => {
    // `gpuMatmul` rejects here. A silent fallback would make the one sentence
    // this category is about — "this half runs on your GPU and this half
    // deliberately does not" — quietly untrue.
    const { result } = await fitOn(CSV, "logistic");
    expect((result as { result: FitResult }).result.compute).toBe("cpu");
  });

  it("refuses a numeric target with a message that says what to do", async () => {
    const { error } = await fitOn(CSV, "forest", { targetIndex: 0 });
    expect(error?.type).toBe("error");
    expect((error as { error: string }).error).toMatch(/numeric/i);
  });

  it("fails the request, not the model, when a fit throws", async () => {
    const { messages } = await fitOn(CSV, "forest", { targetIndex: 0 });
    const error = messages.find((m) => m.type === "error");
    // `id != null` is the discriminator: Machine B failed and the worker still
    // holds the dataset. A load failure would carry no id and take the page to
    // `error`.
    expect(error && "id" in error && error.id).toBe(1);
  });

  it("predicts a single row against the fit in hand", async () => {
    const { handle, messages } = await fitOn(CSV, "forest");
    messages.length = 0;
    await handle({ type: "run", id: 2, predict: { values: [1, 9] } });
    const result = messages.find((m) => m.type === "result");
    const predicted = (result as { result: PredictResult }).result;
    expect(predicted.labels).toEqual(["a", "b"]);
    expect(predicted.scores).toHaveLength(2);
    expect(predicted.scores[0] + predicted.scores[1]).toBeCloseTo(1, 4);
  });

  it("refuses to predict before anything has been fitted", async () => {
    const { messages, handle } = harness();
    const { dataset } = parseCsv(CSV);
    await handle({ type: "load", dataset });
    await handle({ type: "run", id: 9, predict: { values: [1, 1] } });
    const error = messages.find((m) => m.type === "error");
    expect((error as { error: string }).error).toMatch(/fit a model/i);
  });

  it("serialises load and the run posted behind it", async () => {
    // The FIT button posts both in one tick. A fit that began before the
    // dataset landed would fail on an empty frame.
    const { messages, handle } = harness();
    const { dataset } = parseCsv(CSV);
    const target = dataset.columns.length - 1;
    const a = handle({ type: "load", dataset });
    const b = handle({
      type: "run",
      id: 1,
      spec: {
        family: "forest",
        objective: "classification",
        targetIndex: target,
        featureIndices: [0, 1],
        hp: familyInfo("forest").defaults,
        seed: 1,
        testFraction: 0.3,
      },
    });
    await Promise.all([a, b]);
    expect(messages.find((m) => m.type === "error")).toBeUndefined();
    expect(messages.find((m) => m.type === "result")).toBeDefined();
  });

  it("never trains a row whose categorical target was missing", async () => {
    // The dangerous half: a missing categorical code is 0, which is a real
    // class, so those rows would silently join whichever level was seen first
    // and the model would be fitted on labels nobody supplied.
    const withGaps = CSV.replace("1,1,a", "1,1,").replace("2,1,a", "2,1,");
    const { result } = await fitOn(withGaps, "forest");
    const fit = (result as { result: FitResult }).result;
    expect(fit.droppedRows).toBe(2);
    expect(fit.trainRows + fit.testRows).toBe(10);
  });

  it("ranks a genuinely unused column near zero", async () => {
    const withNoise = CSV.split("\n")
      .map((line, i) =>
        i === 0 ? `noise,${line}` : line ? `${(i * 37) % 11},${line}` : line,
      )
      .join("\n");
    const { result } = await fitOn(withNoise, "boosting");
    const fit = (result as { result: FitResult }).result;
    const noise = fit.importance.find((f) => f.name === "noise");
    expect(noise).toBeDefined();
    expect(Math.abs(noise?.drop ?? 1)).toBeLessThan(0.2);
  });
});

describe("the sample datasets", () => {
  it("make the ladder disagree", async () => {
    // A sample every family gets right demonstrates nothing about any of them,
    // and is satisfied by a page that fits one model and draws it four times.
    // The synthetic credit-risk file is built around an interaction no linear
    // boundary can express, and this is the assertion that keeps it that way.
    const sample = SAMPLES.find((s) => s.id === "credit-risk");
    expect(sample).toBeDefined();
    const csv = await sample!.load();
    const scores = new Map<Family, number>();
    for (const family of FAMILIES) {
      const { result } = await fitOn(csv, family.id);
      const fit = (result as { result: FitResult }).result;
      scores.set(family.id, fit.classification?.accuracy ?? 0);
    }
    const linear = scores.get("logistic") ?? 0;
    const trees = Math.max(scores.get("forest") ?? 0, scores.get("boosting") ?? 0);
    expect(trees - linear).toBeGreaterThan(0.05);
    // And the claim the page makes in words: the neural network does not win.
    expect(scores.get("mlp") ?? 1).toBeLessThan(trees);
  }, 120_000);
});

describe("the privacy claim", () => {
  it("is a behaviour, so it is asserted rather than stated", async () => {
    // Nothing in `src/tabular/` may reach the network, IndexedDB or
    // localStorage. A run record naming the user's columns would undo the
    // entire argument for the page, and `createInferenceRun` has no caller
    // anywhere in `src/` — this page must not become the first.
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const opened: string[] = [];
    const indexed = (globalThis as { indexedDB?: { open: unknown } }).indexedDB;
    if (indexed) {
      vi.spyOn(indexed as unknown as { open: () => unknown }, "open").mockImplementation(
        ((name: string) => {
          opened.push(name);
          throw new Error("not allowed");
        }) as never,
      );
    }
    const setItem = vi.spyOn(Storage.prototype, "setItem");

    await fitOn(CSV, "boosting");

    expect(fetchSpy).not.toHaveBeenCalled();
    expect(opened).toEqual([]);
    expect(setItem).not.toHaveBeenCalled();
    vi.restoreAllMocks();
  });
});

// --- The regression arm ------------------------------------------------------

const NUMERIC = `x,z,y
1,0,3.1
2,0,5.0
3,1,7.2
4,1,9.1
5,0,11.0
6,1,12.9
7,0,15.2
8,1,17.1
9,0,19.0
10,1,20.8
11,0,23.1
12,1,25.0
`;

async function regressOn(
  csv: string,
  family: Family,
  extra: Record<string, unknown> = {},
) {
  const { messages, handle } = harness();
  const { dataset } = parseCsv(csv, { name: "numeric" });
  await handle({ type: "load", dataset });
  const target = dataset.columns.length - 1;
  await handle({
    type: "run",
    id: 1,
    spec: {
      family,
      objective: "regression",
      targetIndex: target,
      featureIndices: dataset.columns.map((_, i) => i).filter((i) => i !== target),
      hp: familyInfo(family, "regression").defaults,
      seed: 7,
      testFraction: 0.3,
      ...extra,
    },
  });
  return {
    messages,
    handle,
    dataset,
    result: messages.find((m) => m.type === "result"),
    error: messages.find((m) => m.type === "error"),
  };
}

describe("the regression arm", () => {
  it("fits every regression rung and beats the train-mean baseline", async () => {
    for (const family of REGRESSION_FAMILIES) {
      const { result, error } = await regressOn(NUMERIC, family.id);
      expect(error, `${family.id}: ${(error as { error?: string })?.error}`).toBeUndefined();
      const fit = (result as { result: FitResult }).result;
      expect(fit.regression, family.id).toBeDefined();
      expect(fit.regression?.rmse ?? Infinity).toBeLessThan(
        fit.regression?.baselineRmse ?? 0,
      );
      // The baseline travels inside the metrics, in the target's own units.
      expect(fit.regression?.units).toBe("y");
    }
  });

  it("refuses a text target with a message that says what to do", async () => {
    const { error } = await regressOn("a,b\nx,1\ny,2\nx,3\n", "ridge", {
      targetIndex: 0,
      featureIndices: [1],
    });
    expect((error as { error: string }).error).toMatch(/text/i);
  });

  it("drops rows with no answer rather than imputing or refusing", async () => {
    // Imputing a target is inventing the thing being predicted; refusing the
    // whole file over two blank cells makes the page unusable on real data (the
    // Palmer penguins sample has exactly that). So the rows go, and the count
    // is reported.
    const { result, error } = await regressOn(
      "x,y\n1,1\n2,\n3,3\n4,4\n5,5\n6,6\n7,7\n8,8\n",
      "ridge",
    );
    expect(error).toBeUndefined();
    const fit = (result as { result: FitResult }).result;
    expect(fit.droppedRows).toBe(1);
    expect(fit.trainRows + fit.testRows).toBe(7);
  });

  it("says so when almost nothing is left after dropping", async () => {
    const { error } = await regressOn("x,y\n1,1\n2,\n3,\n4,\n5,\n6,\n", "ridge");
    expect((error as { error: string }).error).toMatch(/only 1 rows/i);
  });

  it("reports rank deficiency instead of returning a plausible answer", async () => {
    // Two identical columns. An inverse would return one of infinitely many
    // coefficient vectors and look entirely fine.
    const { result } = await regressOn(NUMERIC, "ridge", {
      featureIndices: [0, 0],
      hp: { ...familyInfo("ridge", "regression").defaults, lambda: 0 },
    });
    const fit = (result as { result: FitResult }).result;
    expect(fit.rankDeficient).toBe(true);
  });

  it("returns coefficients for ridge and importances for a tree", async () => {
    const ridge = (
      (await regressOn(NUMERIC, "ridge")).result as { result: FitResult }
    ).result;
    expect(ridge.coefficients?.length).toBeGreaterThan(0);
    const forest = (
      (await regressOn(NUMERIC, "forest")).result as { result: FitResult }
    ).result;
    expect(forest.coefficients).toBeUndefined();
    expect(forest.importance.length).toBeGreaterThan(0);
  });

  it("returns a band with its measured coverage", async () => {
    const { result } = await regressOn(NUMERIC, "quantile", {
      quantiles: [0.1, 0.5, 0.9],
    });
    const fit = (result as { result: FitResult }).result;
    expect(fit.quantilePredictions?.length).toBe(3 * fit.testRows);
    // A band of the wrong width looks entirely correct on screen, so the number
    // is carried rather than left to be eyeballed.
    expect(fit.quantileCoverage).toBeGreaterThanOrEqual(0);
    expect(fit.quantileCoverage).toBeLessThanOrEqual(1);
  });

  it("scores a logged fit in the target's own units, and reports log space separately", async () => {
    // The page's claim, and therefore a test: the comparable numbers are the
    // back-transformed ones, and the log-space ones are labelled and kept apart.
    const raw = ((await regressOn(NUMERIC, "ridge")).result as { result: FitResult }).result;
    const { result } = await regressOn(NUMERIC, "ridge", { logTarget: true });
    const logged = (result as { result: FitResult }).result;

    expect(logged.regression?.units).toBe("y");
    expect(logged.logSpaceRegression?.units).toBe("log(1 + y)");
    // Smaller for the same reason a logarithm is smaller — which is exactly why
    // it must not be rendered beside the raw fit's RMSE unlabelled.
    expect(logged.logSpaceRegression?.rmse ?? 1).toBeLessThan(
      logged.regression?.rmse ?? 0,
    );
    // Both fits are scored on the same scale, so they can be compared at all.
    expect(raw.regression?.units).toBe(logged.regression?.units);
  });

  it("predicts one row back in the target's units", async () => {
    const { handle, messages } = await regressOn(NUMERIC, "ridge", { logTarget: true });
    messages.length = 0;
    await handle({ type: "run", id: 2, predict: { values: [6, 1] } });
    const predicted = (messages.find((m) => m.type === "result") as { result: PredictResult })
      .result;
    expect(predicted.objective).toBe("regression");
    // ~13 on this line, not ~2.6 — a prediction left in log space would look
    // like a perfectly ordinary number.
    expect(predicted.scores[0]).toBeGreaterThan(8);
  });
});
