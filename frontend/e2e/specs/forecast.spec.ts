import { expect, test } from "./../fixtures/base";

// /time-series-forecasting, end to end.
//
// It runs in the **default mocked suite** rather than behind `@slow` or the
// webgpu project, because there is genuinely nothing to download and no GPU
// involved: the whole page is arithmetic over one array on the main thread.
// That is also the first thing asserted — a page whose claim is "no model, no
// download, no worker" should have a test that would notice one appearing.
//
// The assertion that matters is **the spread against the single split**. "A
// chart appeared" would pass while the backtest reused one split's numbers for
// every window, which is this page's most plausible bug and would render as a
// perfectly flat strip.

test.describe("Time series forecasting", () => {
  test("has no model, no download and no worker", async ({ page, mockApi }) => {
    await mockApi();

    const outbound: string[] = [];
    await page.route("**/*", (route) => {
      const url = route.request().url();
      if (!url.includes("localhost") && !url.includes("127.0.0.1")) outbound.push(url);
      return route.continue();
    });

    // Count Workers the page constructs. Every other route in the app builds
    // one; this one must not, and a later refactor could quietly add one.
    await page.addInitScript(() => {
      const w = window as unknown as { __workers: number; Worker: unknown };
      w.__workers = 0;
      const Original = window.Worker;
      // @ts-expect-error - replacing the constructor for the count
      window.Worker = class extends Original {
        constructor(...args: ConstructorParameters<typeof Worker>) {
          (window as unknown as { __workers: number }).__workers++;
          super(...args);
        }
      };
    });

    await page.goto("/time-series-forecasting");
    await expect(
      page.getByRole("heading", { name: /time series forecasting/i }),
    ).toBeVisible();

    // Three bands, and the missing fourth explained rather than absent.
    await expect(page.getByTestId("slot-1")).toBeVisible();
    await expect(page.getByTestId("slot-2")).toBeVisible();
    await expect(page.getByTestId("slot-3")).toBeVisible();
    await expect(page.getByTestId("slot-4")).toHaveCount(0);
    await expect(page.getByTestId("no-fit-band")).toContainText(/no ONNX weights/i);

    // The honesty note is in OUTPUT's empty state, so it is read before there
    // is a forecast to believe.
    await expect(page.getByTestId("no-model-note")).toContainText(
      /no learned model on this page/i,
    );

    await page.getByRole("button", { name: /airline passengers/i }).click();
    await expect(page.getByTestId("series-summary")).toContainText("144 points");
    await expect(page.getByTestId("backtest-strip")).toBeVisible();

    expect(outbound).toEqual([]);
    expect(await page.evaluate(() => (window as unknown as { __workers: number }).__workers)).toBe(0);
  });

  test("the backtest spread is wide around the single-split number", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    await page.goto("/time-series-forecasting");
    await page.getByRole("button", { name: /airline passengers/i }).click();
    await expect(page.getByTestId("series-summary")).toBeVisible();

    // Seasonal naive on the airline series at a season of 12 — the pairing the
    // sample exists for.
    await page.getByRole("button", { name: /seasonal naive/i }).click();
    await page.getByTestId("backtest-strip").waitFor();

    const line = await page.getByTestId("backtest-spread").innerText();
    const [, single, lo, hi] =
      /one split ([\d.]+) · windows ([\d.]+)–([\d.]+)/.exec(line) ?? [];
    expect(single, line).toBeTruthy();

    const one = Number(single);
    const min = Number(lo);
    const max = Number(hi);

    // **This is the page's claim.** A backtest that reused one split's numbers
    // for every window — the bug most available here — would render min === max
    // and pass any "a chart appeared" assertion.
    expect(max).toBeGreaterThan(min * 1.5);
    // And the single split is one of those windows, so it sits inside the range
    // rather than beside it. A number outside it would mean the two were
    // computed from different runs, which is exactly what carrying them in one
    // result prevents.
    expect(one).toBeGreaterThanOrEqual(min - 1e-6);
    expect(one).toBeLessThanOrEqual(max + 1e-6);
  });

  test("the season length is a control, and getting it wrong is visible", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    await page.goto("/time-series-forecasting");
    await page.getByRole("button", { name: /airline passengers/i }).click();
    await page.getByRole("button", { name: /seasonal naive/i }).click();
    await expect(page.getByTestId("metric-table")).toBeVisible();

    const at12 = await page.getByTestId("metric-table").innerText();
    await page.getByLabel(/season length/i).fill("11");
    const at11 = await page.getByTestId("metric-table").innerText();

    // Off by one month on a series with a twelve-month period: the forecast is
    // confidently wrong by a phase, and only the numbers say so.
    expect(at11).not.toBe(at12);
    await expect(page.getByText(/Set by you, never detected/i)).toBeVisible();
  });

  test("gaps are reported rather than filled", async ({ page, mockApi }) => {
    await mockApi();
    await page.goto("/time-series-forecasting");
    await page
      .getByLabel(/paste a column/i)
      .fill("2024-01-01,1\n2024-01-02,2\n2024-01-05,3\n2024-01-06,4\n2024-01-07,5\n");
    await expect(page.getByTestId("series-gaps")).toContainText(/reported, not filled/i);
  });
});
