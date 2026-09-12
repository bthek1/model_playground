import { expect, test } from "../../fixtures/base";

// The /graph route, end to end in a real browser.
//
// Two things here cannot be tested anywhere else. The WGSL aggregation kernel
// needs a GPU device, which happy-dom does not have — so its agreement with the
// CPU reference is checked here, by running both and comparing. And the model's
// *correctness* needs a real training run: a wrong aggregation still produces a
// falling loss and a plausible-looking accuracy curve, which is exactly this
// category's documented failure mode, so the assertion is an **accuracy floor**
// rather than "a result appeared".
//
// Tagged @slow: a real 200-epoch run, and on the CPU path a slow one.

/** Evaluate in the page so the kernel runs in the app's own module graph. */
async function crossCheckKernel(page: import("@playwright/test").Page) {
  return page.evaluate(async () => {
    const { GraphAggregator } = await import("/src/webgpu/gnnRuntime.ts");
    const { makeCpuAggregate } = await import("/src/webgpu/gnn.ts");
    const { mulberry32 } = await import("/src/lib/random.ts");

    const rand = mulberry32(7);
    const nNodes = 97;
    const nFeat = 11;

    // A random symmetric graph with no self-loops — the two invariants the
    // shader header names, and the ones the transpose depends on.
    const neighbours: number[][] = Array.from({ length: nNodes }, () => []);
    for (let u = 0; u < nNodes; u++) {
      for (let v = u + 1; v < nNodes; v++) {
        if (rand() < 0.06) {
          neighbours[u].push(v);
          neighbours[v].push(u);
        }
      }
    }
    const rowPtr = new Uint32Array(nNodes + 1);
    for (let i = 0; i < nNodes; i++) {
      rowPtr[i + 1] = rowPtr[i] + neighbours[i].length;
    }
    const colIdx = new Uint32Array(rowPtr[nNodes]);
    let at = 0;
    for (const list of neighbours) for (const v of list) colIdx[at++] = v;

    const alpha = new Float32Array(nNodes);
    const beta = new Float32Array(nNodes);
    for (let i = 0; i < nNodes; i++) {
      alpha[i] = 1 / Math.sqrt(neighbours[i].length + 1);
      beta[i] = alpha[i] * (0.5 + rand());
    }
    const x = new Float32Array(nNodes * nFeat);
    for (let i = 0; i < x.length; i++) x[i] = rand() * 4 - 2;

    const gpu = await GraphAggregator.create(rowPtr, colIdx, nNodes, alpha, beta);
    const cpu = makeCpuAggregate(rowPtr, colIdx, nNodes, alpha, beta);
    try {
      let worst = 0;
      for (const transposed of [false, true]) {
        const a = await gpu.run(x, nFeat, transposed);
        const b = await cpu(x, nFeat, transposed);
        for (let i = 0; i < a.length; i++) {
          worst = Math.max(worst, Math.abs(a[i] - b[i]));
        }
      }
      return { worst, nEdges: colIdx.length };
    } finally {
      gpu.dispose();
    }
  });
}

test.describe("Graph machine learning", () => {
  test("the page loads its bundled graph without a download", async ({
    page,
    mockApi,
  }) => {
    await mockApi();

    // Nothing may reach the Hugging Face Hub or any model host: this route has
    // no checkpoint at all, and the dataset ships with the app.
    const external: string[] = [];
    await page.route("**/*", (route) => {
      const url = route.request().url();
      if (!url.includes("localhost") && !url.includes("127.0.0.1")) {
        external.push(url);
      }
      return route.continue();
    });

    await page.goto("/graph");
    await expect(
      page.getByRole("heading", { name: /graph machine learning/i }),
    ).toBeVisible();

    // All four bands, and OUTPUT says what you'll get rather than showing one.
    for (const step of [1, 2, 3, 4]) {
      await expect(page.getByTestId(`slot-${step}`)).toBeVisible();
    }
    await expect(page.getByTestId("output-empty")).toBeVisible();
    await expect(page.getByRole("button", { name: /^train$/i })).toBeDisabled();

    await page.getByRole("button", { name: /load graph/i }).click();
    await expect(page.getByTestId("model-ready")).toBeVisible({ timeout: 30_000 });
    await expect(page.getByTestId("model-ready")).toContainText("2708 nodes");
    await expect(page.getByTestId("model-ready")).toContainText("5278 edges");
    await expect(page.getByRole("button", { name: /^train$/i })).toBeEnabled();

    // Still nothing drawn — loading is not running.
    await expect(page.getByTestId("output-empty")).toBeVisible();
    expect(external).toEqual([]);
  });

  test("the aggregation kernel agrees with the CPU reference", async ({
    page,
    mockApi,
    webgpuStatus,
  }) => {
    test.skip(
      webgpuStatus !== "ready",
      `No GPU device in this browser (status: ${webgpuStatus}).`,
    );
    await mockApi();
    await page.goto("/graph");

    const { worst, nEdges } = await crossCheckKernel(page);
    expect(nEdges).toBeGreaterThan(100); // the random graph really has edges
    // f32 on both sides, summed in a different order — agreement to ~1e-5.
    expect(worst).toBeLessThan(1e-5);
  });

  test("@slow a 2-layer GCN learns Cora", async ({ page, mockApi }) => {
    test.setTimeout(300_000);
    await mockApi();
    await page.goto("/graph");

    await page.getByRole("button", { name: /load graph/i }).click();
    await expect(page.getByTestId("model-ready")).toBeVisible({ timeout: 30_000 });
    await page.getByRole("button", { name: /^train$/i }).click();

    await expect(page.getByTestId("graph-canvas")).toBeVisible({
      timeout: 120_000,
    });
    await expect(page.getByRole("button", { name: /^train$/i })).toBeEnabled({
      timeout: 240_000,
    });

    // The assertion that matters. A GCN on Cora reaches roughly 0.78 here; a
    // wrong aggregation — a missing self-loop, a transposed gather, a mis-scaled
    // normalisation — lands near 0.3 while still drawing a perfectly plausible
    // loss curve. Anything above chance-plus-a-lot is the real signal.
    const accuracy = await page
      .getByText("Test accuracy")
      .locator("xpath=following-sibling::dd[1]")
      .innerText();
    expect(Number.parseFloat(accuracy)).toBeGreaterThan(65);
  });

  test("@slow depth drives accuracy down and neighbour similarity up", async ({
    page,
    mockApi,
  }) => {
    // Two full 200-epoch runs, and the 8-layer one is roughly four times the
    // work of the 2-layer one. On a real GPU that is seconds; on the
    // SwiftShader fallback a CI runner uses it is minutes, and this budget is
    // sized for the slow case.
    test.setTimeout(1_800_000);
    await mockApi();
    await page.goto("/graph");
    await page.getByRole("button", { name: /load graph/i }).click();
    await expect(page.getByTestId("model-ready")).toBeVisible({ timeout: 30_000 });

    const read = async (label: string) =>
      Number.parseFloat(
        await page
          .getByText(label)
          .locator("xpath=following-sibling::dd[1]")
          .innerText(),
      );

    const runAt = async (layers: number) => {
      await page.getByLabel(/depth/i).fill(String(layers));
      await page.getByRole("button", { name: /^train$/i }).click();
      await expect(page.getByRole("button", { name: /^train$/i })).toBeEnabled({
        timeout: 780_000,
      });
      return {
        accuracy: await read("Test accuracy"),
        smoothness: await read("Neighbour similarity"),
      };
    };

    const shallow = await runAt(2);
    const deep = await runAt(8);

    // Oversmoothing, measured rather than asserted by eye: every node's
    // representation converges on its neighbours', and the model gets worse.
    expect(deep.smoothness).toBeGreaterThan(shallow.smoothness);
    expect(deep.accuracy).toBeLessThan(shallow.accuracy);
  });
});
