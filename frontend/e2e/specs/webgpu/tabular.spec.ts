import { expect, test } from "../../fixtures/base";

// The Tabular routes, end to end in a real browser.
//
// These run in the `webgpu` project because half the ladder is WGSL — the
// logistic and MLP rungs dispatch real compute shaders — and the project passes
// `--enable-unsafe-swiftshader` so a runner with no `/dev/dri` still executes
// them for real. They are not tagged `@slow`: there is nothing to download, so
// the whole file is seconds.
//
// The assertion that matters is **above the majority-class baseline**, not
// "a number appeared". A class prior is the easiest thing in any dataset to
// learn, so a page whose fit was silently broken still renders a confident
// accuracy — the `/graph-classification` lesson, and the reason the baseline is
// rendered beside every score rather than left for the reader to infer.

test.describe("Tabular classification", () => {
  test("fits nothing on arrival, and nothing leaves the machine", async ({
    page,
    mockApi,
  }) => {
    await mockApi();

    const outbound: string[] = [];
    await page.route("**/*", (route) => {
      const url = route.request().url();
      // Anything that is not the dev server itself. The page's entire claim is
      // that the CSV stays in the tab, so this is the assertion that claim
      // actually reduces to.
      if (!url.includes("localhost") && !url.includes("127.0.0.1")) {
        outbound.push(url);
      }
      return route.continue();
    });

    await page.goto("/tabular-classification");
    await expect(
      page.getByRole("heading", { name: /tabular classification/i }),
    ).toBeVisible();

    for (const step of [1, 2, 3, 4]) {
      await expect(page.getByTestId(`slot-${step}`)).toBeVisible();
    }
    await expect(page.getByTestId("output-empty")).toBeVisible();
    await expect(page.getByTestId("fit-button")).toBeDisabled();
    await expect(page.getByText(/never leaves this device/i)).toBeVisible();

    // Choosing a dataset and a target fits nothing.
    await page.getByRole("button", { name: /credit risk/i }).click();
    await expect(page.getByTestId("dataset-summary")).toContainText("1,800 rows");
    await expect(page.getByTestId("output-empty")).toBeVisible();

    expect(outbound).toEqual([]);
  });

  test("beats the majority baseline, and the MLP does not win", async ({
    page,
    mockApi,
  }) => {
    test.setTimeout(300_000);
    await mockApi();
    await page.goto("/tabular-classification");
    await page.getByRole("button", { name: /credit risk/i }).click();
    await expect(page.getByTestId("dataset-summary")).toBeVisible();

    /** Fit one rung and read the accuracy the page renders. */
    async function fit(family: RegExp): Promise<number> {
      await page.getByRole("button", { name: family }).click();
      await page.getByTestId("fit-button").click();
      await expect(page.getByTestId("model-ready")).toBeVisible({ timeout: 180_000 });
      const text = await page.getByTestId("metric-block").innerText();
      const match = /([\d.]+)%/.exec(text);
      expect(match, `no accuracy rendered for ${family}`).not.toBeNull();
      return Number(match?.[1]) / 100;
    }

    const boosting = await fit(/gradient boosting/i);
    const baselineText = await page.getByTestId("baseline-note").innerText();
    const baseline = Number(/scores ([\d.]+)%/.exec(baselineText)?.[1]) / 100;

    // The synthetic credit-risk sample is 56% "no", so "above 0.5" would pass
    // with the model having learned nothing but the prior.
    expect(baseline).toBeGreaterThan(0.5);
    expect(boosting).toBeGreaterThan(baseline + 0.05);

    // The claim the page makes in words, asserted: the neural network loses to
    // the trees. Shipping the MLP is only worth it if that is actually true on
    // the data in front of the user.
    const mlp = await fit(/neural network/i);
    expect(mlp).toBeLessThan(boosting);

    // …and the linear floor loses too, which is what makes the sample worth
    // having: a dataset every family gets right says nothing about any of them.
    const logistic = await fit(/logistic regression/i);
    expect(logistic).toBeLessThan(boosting);
  });

  test("the threshold slider re-reads the fit instead of refitting", async ({
    page,
    mockApi,
  }) => {
    test.setTimeout(180_000);
    await mockApi();
    await page.goto("/tabular-classification");
    await page.getByRole("button", { name: /credit risk/i }).click();
    await page.getByRole("button", { name: /random forest/i }).click();
    await page.getByTestId("fit-button").click();
    await expect(page.getByTestId("model-ready")).toBeVisible({ timeout: 180_000 });

    const fitted = await page.getByTestId("model-ready").innerText();
    const duration = /Fitted in [\d.]+s on [\d,]+ rows/.exec(fitted)?.[0];
    expect(duration).toBeTruthy();
    const before = await page.getByTestId("confusion-matrix").innerText();

    await page.getByLabel(/decision threshold/i).fill("0.2");
    await expect(page.getByTestId("confusion-matrix")).not.toHaveText(before);

    // The fit line still reports the *same* fit, which is what "it did not
    // refit" looks like from outside: a refit would post a new duration.
    await expect(page.getByTestId("model-ready")).toContainText(duration as string);
  });

  test("the linear rung really runs on the GPU", async ({ page, mockApi }) => {
    test.setTimeout(180_000);
    await mockApi();
    await page.goto("/tabular-classification");
    await page.getByRole("button", { name: /penguins/i }).click();
    await page.getByRole("button", { name: /logistic regression/i }).click();
    await expect(page.getByTestId("family-compute")).toContainText(/your GPU/i);
    await page.getByTestId("fit-button").click();
    await expect(page.getByTestId("model-ready")).toBeVisible({ timeout: 180_000 });
    // The page says which half ran where, so a silent CPU fallback would make
    // that sentence quietly untrue. On a runner this is SwiftShader — slow, and
    // numerically real.
    await expect(page.getByTestId("model-ready")).toContainText("GPU");
  });
});

test.describe("Tabular regression", () => {
  test("beats the train-mean baseline, and draws what a score cannot say", async ({
    page,
    mockApi,
  }) => {
    test.setTimeout(300_000);
    await mockApi();
    await page.goto("/tabular-regression");
    await expect(
      page.getByRole("heading", { name: /tabular regression/i }),
    ).toBeVisible();
    await expect(page.getByTestId("output-empty")).toBeVisible();

    await page.getByRole("button", { name: /palmer penguins/i }).click();
    await expect(page.getByTestId("dataset-summary")).toBeVisible();
    await page.getByRole("button", { name: /ridge regression/i }).click();
    await page.getByTestId("fit-button").click();
    await expect(page.getByTestId("model-ready")).toBeVisible({ timeout: 180_000 });

    // **An R² band above the train-mean baseline, not a floor.** A floor alone
    // passes on a model that predicts the mean — which is exactly what a broken
    // solve produces, since the intercept alone already gets there.
    const text = await page.getByTestId("regression-metrics").innerText();
    const r2 = Number(/R²\s+([\d.-]+)/.exec(text)?.[1]);
    expect(r2).toBeGreaterThan(0.5);
    expect(r2).toBeLessThanOrEqual(1);

    const baseline = await page.getByTestId("regression-baseline").innerText();
    const baselineRmse = Number(/scores\s+([\d.]+)/.exec(baseline)?.[1]);
    const rmse = Number(/RMSE\s+([\d.]+)/.exec(text)?.[1]);
    expect(rmse).toBeLessThan(baselineRmse);

    await expect(page.getByTestId("predicted-vs-actual")).toBeVisible();
    await expect(page.getByTestId("residual-plot")).toBeVisible();
    await expect(page.getByTestId("coefficients")).toBeVisible();

    // The page's own claim about where ridge runs.
    await expect(page.getByTestId("model-ready")).toContainText("GPU");
  });

  test("the log toggle costs a second fit, and the two metric blocks name their units", async ({
    page,
    mockApi,
  }) => {
    test.setTimeout(300_000);
    await mockApi();
    await page.goto("/tabular-regression");
    await page.getByRole("button", { name: /palmer penguins/i }).click();
    await expect(page.getByTestId("dataset-summary")).toBeVisible();
    await page.getByRole("button", { name: /ridge regression/i }).click();

    await page.getByTestId("fit-button").click();
    await expect(page.getByTestId("model-ready")).toBeVisible({ timeout: 180_000 });
    const rawUnits = await page.getByTestId("regression-metrics").innerText();
    expect(rawUnits).toContain("body_mass_g");
    await expect(page.getByTestId("log-space-metrics")).toHaveCount(0);

    // Flipping it runs nothing…
    await page.getByLabel(/fit on log/i).check();
    await expect(page.getByTestId("log-space-metrics")).toHaveCount(0);

    // …and the next Fit is a real second fit.
    await page.getByTestId("fit-button").click();
    await expect(page.getByTestId("log-space-metrics")).toBeVisible({ timeout: 180_000 });

    // Both blocks say which units they are in. That is the whole demonstration:
    // the log-space RMSE is smaller for the same reason a logarithm is smaller,
    // and reading it as an improvement is the mistake.
    await expect(page.getByTestId("log-space-metrics")).toContainText("log(1 + body_mass_g)");
    await expect(page.getByTestId("regression-metrics")).toContainText("body_mass_g");
    await expect(page.getByTestId("log-space-metrics")).toContainText(/not comparable/i);
  });

  test("the quantile band reports its measured coverage", async ({ page, mockApi }) => {
    test.setTimeout(300_000);
    await mockApi();
    await page.goto("/tabular-regression");
    await page.getByRole("button", { name: /credit risk/i }).click();
    await expect(page.getByTestId("dataset-summary")).toBeVisible();
    await page.getByRole("button", { name: /quantile regression/i }).click();
    await page.getByTestId("fit-button").click();
    await expect(page.getByTestId("model-ready")).toBeVisible({ timeout: 180_000 });

    // **Coverage, not "a band was drawn".** A band of the wrong width looks
    // entirely correct on the chart, so the assertion is the fraction of
    // held-out rows it actually contains.
    const note = await page.getByTestId("band-coverage").innerText();
    const coverage = Number(/([\d.]+)%/.exec(note)?.[1]) / 100;
    expect(coverage).toBeGreaterThan(0.6);
    expect(coverage).toBeLessThan(0.95);
  });
});
