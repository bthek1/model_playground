import { expect, test } from "../../fixtures/base";

// The /link-prediction route, end to end in a real browser.
//
// The assertion that matters here is a **band**, not a floor, and that is the
// whole point of the spec. This page's one real bug is leakage — an encoder
// still allowed to aggregate over a citation it is later asked to predict — and
// leakage makes the number go *up*. A floor would pass with the bug in place and
// pass more comfortably than without it. So:
//
//   - above 0.85, because this page's own configuration measures 0.925 on the
//     CPU reference and 0.92 here, while a broken one sits at chance;
//   - below 0.985, because that is what a leaked split looks like.
//
// Tagged @slow: a real 250-epoch run, and on the CPU path a slow one.

test.describe("Link prediction", () => {
  test("loads its bundled graph, and splits nothing until asked", async ({
    page,
    mockApi,
  }) => {
    await mockApi();

    // No checkpoint exists for this task; the dataset ships with the app. A
    // single external request here would mean something was downloaded.
    const external: string[] = [];
    await page.route("**/*", (route) => {
      const url = route.request().url();
      if (!url.includes("localhost") && !url.includes("127.0.0.1")) {
        external.push(url);
      }
      return route.continue();
    });

    await page.goto("/link-prediction");
    await expect(
      page.getByRole("heading", { name: /link prediction/i }),
    ).toBeVisible();

    for (const step of [1, 2, 3, 4]) {
      await expect(page.getByTestId(`slot-${step}`)).toBeVisible();
    }
    await expect(page.getByTestId("output-empty")).toBeVisible();
    await expect(page.getByRole("button", { name: /^train$/i })).toBeDisabled();

    await page.getByRole("button", { name: /load graph/i }).click();
    await expect(page.getByTestId("model-ready")).toBeVisible({
      timeout: 60_000,
    });
    await expect(page.getByTestId("model-ready")).toContainText("citations kept");
    await expect(page.getByRole("button", { name: /^train$/i })).toBeEnabled();

    // Loading is not running: the graph is split and laid out, and nothing has
    // been scored.
    await expect(page.getByTestId("output-empty")).toBeVisible();
    expect(external).toEqual([]);
  });

  test("holds out the fraction it says it does", async ({ page, mockApi }) => {
    await mockApi();
    await page.goto("/link-prediction");
    await page.getByRole("button", { name: /load graph/i }).click();
    await expect(page.getByTestId("model-ready")).toBeVisible({
      timeout: 60_000,
    });

    const counts = async () => {
      const text = (await page.getByTestId("model-ready").innerText()).replace(
        /\s+/g,
        " ",
      );
      const kept = Number(/(\d+) citations kept/.exec(text)?.[1]);
      const held = Number(/· (\d+) test/.exec(text)?.[1]);
      return { kept, held };
    };

    const at10 = await counts();
    // 5278 undirected citations in Cora; 10 % of them, minus any kept back to
    // avoid isolating a paper.
    expect(at10.held).toBeGreaterThan(450);
    expect(at10.held).toBeLessThanOrEqual(528);

    await page.getByRole("button", { name: "30%" }).click();
    await expect(page.getByTestId("model-ready")).toBeVisible({
      timeout: 60_000,
    });
    const at30 = await counts();
    expect(at30.held).toBeGreaterThan(at10.held * 2);
    expect(at30.kept).toBeLessThan(at10.kept);
  });

  test("@slow a GCN scores the citations it never saw", async ({
    page,
    mockApi,
  }) => {
    test.setTimeout(420_000);
    await mockApi();
    await page.goto("/link-prediction");

    await page.getByRole("button", { name: /load graph/i }).click();
    await expect(page.getByTestId("model-ready")).toBeVisible({
      timeout: 60_000,
    });
    await page.getByRole("button", { name: /^train$/i }).click();

    await expect(page.getByTestId("graph-canvas")).toBeVisible({
      timeout: 360_000,
    });

    const auc = Number.parseFloat(
      await page
        .getByText("Test AUC", { exact: true })
        .locator("xpath=following-sibling::dd[1]")
        .innerText(),
    );

    // The band. Chance is 0.5 and a wrong aggregation lands there; a leaked
    // split lands above 0.99 and would sail past any floor.
    expect(auc).toBeGreaterThan(0.85);
    expect(auc).toBeLessThan(0.985);

    // The candidates are pairs Cora does not contain, drawn over the graph.
    await expect(page.getByText(/top 20/)).toBeVisible();
  });
});
