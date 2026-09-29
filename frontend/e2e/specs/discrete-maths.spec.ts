import { expect, test } from "./../fixtures/base";

// /discrete-maths, end to end, in the default mocked suite: nothing downloads,
// no GPU is involved, and a whole run is a handful of clicks.
//
// It asserts **known answers on known graphs**, never "a matrix appeared": a
// page that drew the BFS ball and the matrix power from the same (wrong)
// computation would render a confident ✓ over two identical mistakes. So the
// numbers below are the published ones — the Petersen graph has diameter 2, and
// on an even cycle a walk that must move every step cannot return to its start
// in an odd number of them. (The sidebar row itself is covered by
// navigation.spec.ts, which walks every mapped task.)

test.describe("Discrete maths", () => {
  test.beforeEach(async ({ page, mockApi }) => {
    await mockApi();
    await page.addInitScript(() => {
      const w = window as unknown as { __workers: number; Worker: unknown };
      w.__workers = 0;
      const Original = window.Worker;
      window.Worker = class extends Original {
        constructor(...args: ConstructorParameters<typeof Worker>) {
          (window as unknown as { __workers: number }).__workers++;
          super(...args);
        }
      };
    });
    await page.goto("/discrete-maths");
    await expect(page.getByRole("heading", { name: /discrete maths/i })).toBeVisible();
  });

  test("steps BFS on the Petersen graph to its diameter", async ({ page }) => {
    const output = page.getByTestId("slot-3");
    await expect(page.getByTestId("output-empty")).toBeVisible();

    await page.getByRole("button", { name: /petersen/i }).click();
    await page.getByRole("button", { name: /^step$/i }).click();
    await expect(page.getByTestId("stat-ball")).toHaveText("4 of 10");

    await page.getByRole("button", { name: /^step$/i }).click();
    await expect(page.getByTestId("stat-ball")).toHaveText("10 of 10");
    await expect(page.getByTestId("bfs-done")).toContainText("within 2 hops");
    await expect(page.getByRole("button", { name: /^step$/i })).toBeDisabled();
    await expect(output.getByTestId("verdict")).toHaveAttribute("data-match", "true");
  });

  test("on C₈, the self-loop is what lets the source hear itself", async ({ page }) => {
    const output = page.getByTestId("slot-3");
    await page.getByRole("button", { name: /^step$/i }).click();

    // A + I (the default): the row's support is the BFS ball, source included.
    await expect(output.getByTestId("verdict")).toHaveAttribute("data-match", "true");
    await expect(output.getByTestId("node-0")).toHaveAttribute("data-walks", "1");

    // A alone: one step from node 0 lands only on its two neighbours.
    await output.getByRole("button", { name: /a walk must move/i }).click();
    await expect(output.getByTestId("verdict")).toHaveAttribute("data-match", "false");
    await expect(output.getByTestId("walk-support")).toContainText("1, 7");
    await expect(output.getByTestId("node-0")).toHaveAttribute("data-walks", "0");
    await expect(page.getByTestId("step-count")).toHaveText("step 1 of 4"); // the switch did not step

    // The contrast graph: C₇ is one node shorter and odd, so parity breaks.
    await page.getByRole("button", { name: /cycle c₇/i }).click();
    await expect(page.getByTestId("output-empty")).toBeVisible(); // a choice clears, runs nothing
    await page.getByRole("button", { name: /run to end/i }).click();
    await expect(page.getByTestId("step-count")).toHaveText("step 3 of 3");
  });

  test("draws the karate club, and constructs no worker and makes no outbound request", async ({
    page,
  }) => {
    const outbound: string[] = [];
    page.on("request", (r) => {
      const url = r.url();
      if (!url.includes("localhost") && !url.includes("127.0.0.1")) outbound.push(url);
    });

    await page.getByRole("button", { name: /karate/i }).click();
    await page.getByRole("button", { name: /run to end/i }).click();
    await expect(page.getByTestId("stat-ball")).toHaveText("34 of 34");
    await expect(page.getByTestId("stat-ecc")).toHaveText("3"); // the instructor, per networkx

    // Every node is drawn inside the diagram's own box.
    const box = await page.getByTestId("slot-3").getByTestId("graph-diagram").boundingBox();
    expect(box).not.toBeNull();
    for (const v of [0, 16, 33]) {
      const node = await page.getByTestId("slot-3").getByTestId(`node-${v}`).boundingBox();
      expect(node!.x).toBeGreaterThanOrEqual(box!.x);
      expect(node!.x + node!.width).toBeLessThanOrEqual(box!.x + box!.width + 1);
    }

    expect(await page.evaluate(() => (window as unknown as { __workers: number }).__workers)).toBe(0);
    expect(outbound).toEqual([]);
  });

  test("keeps the page inside a phone's width", async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 800 });
    await page.getByRole("button", { name: /karate/i }).click();
    await page.getByRole("button", { name: /run to end/i }).click();
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflow).toBeLessThanOrEqual(0);
  });
});
