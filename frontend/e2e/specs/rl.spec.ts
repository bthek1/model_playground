import { expect, test } from "../fixtures/base";

// /rl, end to end in a real browser, with a real worker.
//
// Not `@slow` and not in the webgpu project: nothing is downloaded and there is
// no GPU involved — Phase 0 of #51 moved every step onto the CPU — so a full
// training run on the default grid is a few seconds. That is also why it runs
// in the default mocked suite rather than behind a flag: the page's claim is
// "it learns", and the only honest check of that is to let it learn.
//
// The assertions that matter are **the arrow in a named cell** and **agreement
// with value iteration**. "A curve went up" is what a broken update rule also
// produces — a wrong discount or a bootstrap through the terminal state still
// finds the goal on a 4×4 grid.

test.describe("Reinforcement learning", () => {
  test("trains nothing on arrival, and nothing leaves the machine", async ({ page, mockApi }) => {
    await mockApi();
    const outbound: string[] = [];
    await page.route("**/*", (route) => {
      const url = route.request().url();
      if (!url.includes("localhost") && !url.includes("127.0.0.1")) outbound.push(url);
      return route.continue();
    });

    await page.goto("/rl");
    await expect(page.getByRole("heading", { name: /reinforcement learning/i })).toBeVisible();

    // Three bands: the fourth is absent on purpose, and says so.
    for (const step of [1, 2, 3]) await expect(page.getByTestId(`slot-${step}`)).toBeVisible();
    await expect(page.getByTestId("slot-4")).toHaveCount(0);
    await expect(page.getByTestId("no-load-band")).toContainText(/the CPU, on purpose/i);

    await expect(page.getByTestId("output-empty")).toBeVisible();
    await expect(page.getByTestId("train-button")).toBeEnabled();

    // Choosing a grid, a hyperparameter and a speed trains nothing.
    await page.getByRole("button", { name: /8 × 8/ }).click();
    await page.getByLabel(/^seed$/i).fill("3");
    await page.getByRole("button", { name: /^max$/ }).click();
    await expect(page.getByTestId("output-empty")).toBeVisible();
    await expect(page.getByTestId("rl-progress")).toHaveCount(0);

    expect(outbound).toEqual([]);
  });

  test("learns the default grid — the optimal arrows, not just a rising curve", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    await page.goto("/rl");
    await page.getByRole("button", { name: /^max$/ }).click();
    await page.getByTestId("train-button").click();
    await expect(page.getByTestId("train-button")).toBeEnabled({ timeout: 30_000 });

    // Behaviour-policy success over the last 100 episodes: ε has decayed to
    // 0.05, so a solved table scores ~95% (measured 93–94% on seeds 1 and 2).
    // A floor, not a band: this number has no failure mode that pushes it up.
    const rate = Number((await page.getByTestId("success-rate").innerText()).replace("%", ""));
    expect(rate).toBeGreaterThanOrEqual(80);

    await expect(page.getByTestId("greedy-rate")).toHaveText("100%");
    await expect(page.getByTestId("policy-agreement")).toHaveText("11 of 11 cells");

    // A named cell: 14 is directly left of the goal, so its arrow must point
    // right. Read off the text rendering of the same table the canvas draws.
    await page.getByText("The policy as text").click();
    await expect(page.locator('[data-testid="policy-grid"] [data-state="14"]')).toHaveAttribute(
      "data-action",
      "right",
    );
  });

  test("never finds the goal at ε = 0, which the page says it will not", async ({ page, mockApi }) => {
    await mockApi();
    await page.goto("/rl");
    await page.getByLabel(/exploration ε/i).fill("0");
    await page.getByRole("button", { name: /^max$/ }).click();
    await page.getByTestId("train-button").click();
    await expect(page.getByTestId("train-button")).toBeEnabled({ timeout: 30_000 });
    await expect(page.getByTestId("success-rate")).toHaveText("0%");
    await expect(page.getByTestId("greedy-rate")).toHaveText("0%");
  });

  test("Stop ends the run, and a second Train starts cleanly", async ({ page, mockApi }) => {
    await mockApi();
    await page.goto("/rl");
    await page.getByRole("button", { name: /30 steps\/s/ }).click();
    await page.getByTestId("train-button").click();
    await expect(page.getByTestId("rl-progress")).toBeVisible();
    await page.getByRole("button", { name: /stop/i }).click();
    await expect(page.getByTestId("train-button")).toBeEnabled();

    await page.getByRole("button", { name: /^max$/ }).click();
    await page.getByTestId("train-button").click();
    await expect(page.getByTestId("train-button")).toBeEnabled({ timeout: 30_000 });
    // The second run is whole: every episode counted once, from zero.
    await expect(page.getByTestId("rl-progress")).toContainText("episode 1,000/1,000");
  });
});
