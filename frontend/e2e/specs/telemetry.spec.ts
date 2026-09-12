import { AppShell } from "../pages/AppShell";
import { ModelPageObject } from "../pages/ModelPage";
import { expect, test } from "../fixtures/base";

// The system panel, asserted where jsdom cannot help.
//
// Three of these four things are geometry — does the panel squeeze the
// workbench, does the result stay above the fold, does the page scroll sideways
// — and a unit test has no layout engine to ask. The fourth is that closing the
// panel actually *stops the sampling* rather than hiding the numbers, which is
// the whole observer-effect promise; `data-tick` on the panel body is how it is
// observable from outside.

const DOCKED = { width: 1440, height: 900 };
const NARROW = { width: 820, height: 900 };

test.describe("the system panel", () => {
  test("is closed until asked for, and then remembers", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const shell = new AppShell(page);
    await page.setViewportSize(DOCKED);
    await page.goto("/home");

    await expect(shell.systemPanel).toHaveCount(0);
    await expect(shell.toggleSystemPanelButton).toHaveAttribute(
      "aria-expanded",
      "false",
    );

    await shell.toggleSystemPanelButton.click();
    await expect(shell.systemPanel).toBeVisible();

    // The choice is a decision, so it survives a reload — unlike a model load.
    await page.reload();
    await expect(shell.systemPanel).toBeVisible();

    await shell.toggleSystemPanelButton.click();
    await expect(shell.systemPanel).toHaveCount(0);
    await page.reload();
    await expect(shell.systemPanel).toHaveCount(0);
  });

  test("opens and closes on Alt+Shift+M", async ({ page, mockApi }) => {
    await mockApi();
    const shell = new AppShell(page);
    await page.setViewportSize(DOCKED);
    await page.goto("/home");

    await page.keyboard.press("Alt+Shift+KeyM");
    await expect(shell.systemPanel).toBeVisible();

    await page.keyboard.press("Alt+Shift+KeyM");
    await expect(shell.systemPanel).toHaveCount(0);
  });

  test("renders all four cards, each saying what its number is", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const shell = new AppShell(page);
    await page.setViewportSize(DOCKED);
    await page.goto("/home");
    await shell.toggleSystemPanelButton.click();

    for (const card of ["gpu", "memory", "cpu", "storage"]) {
      await expect(page.getByTestId(`telemetry-${card}`)).toBeVisible();
    }
    await expect(
      page.getByText(/load and capacity, never utilisation/),
    ).toBeVisible();

    // Storage is the one measurement every browser makes, so it must be a
    // number here rather than a reason.
    await expect(page.getByTestId("telemetry-storage-unavailable")).toHaveCount(0);
  });

  test("samples while open and stops when closed", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const shell = new AppShell(page);
    await page.setViewportSize(DOCKED);
    await page.goto("/home");
    await shell.toggleSystemPanelButton.click();

    const body = page.getByTestId("system-panel-body");
    await expect(body).toHaveAttribute("data-tick", "1");
    // The loop is 1 Hz; a third sample proves it is running, not just seeded.
    await expect(body).toHaveAttribute("data-tick", /[3-9]/, { timeout: 6_000 });

    await shell.toggleSystemPanelButton.click();
    await expect(body).toHaveCount(0);

    // Re-opening restarts from the first sample: the loop was torn down, not
    // paused, and the history was cleared rather than drawn across the gap.
    await shell.toggleSystemPanelButton.click();
    await expect(page.getByTestId("system-panel-body")).toHaveAttribute(
      "data-tick",
      "1",
    );
  });
});

test.describe("the system panel beside a model page", () => {
  test.beforeEach(async ({ page }) => {
    // This spec must never pull weights.
    await page.route(
      (url) => url.hostname.endsWith("huggingface.co"),
      (r) => r.abort(),
    );
  });

  test("leaves the four slots intact and the output above the fold", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const shell = new AppShell(page);
    const model = new ModelPageObject(page);
    await page.setViewportSize(DOCKED);
    await page.goto("/image-classification");
    await shell.toggleSystemPanelButton.click();
    await expect(shell.systemPanel).toBeVisible();

    await expect(model.slots).toHaveCount(4);
    for (const step of [1, 2, 3, 4] as const) {
      await expect(model.slot(step)).toBeVisible();
    }

    // OUTPUT still starts on screen — the panel must not push the result down.
    const box = await model.outputPanel.boundingBox();
    expect(box).not.toBeNull();
    expect(box!.y).toBeLessThan(DOCKED.height);

    // And nothing scrolls sideways.
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - window.innerWidth,
    );
    expect(overflow).toBeLessThanOrEqual(1);
  });

  test("becomes an overlay on a narrow screen rather than squeezing the workbench", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const shell = new AppShell(page);
    const model = new ModelPageObject(page);
    await page.setViewportSize(NARROW);
    await page.goto("/image-classification");

    const runBefore = await model.slot(3).boundingBox();
    await shell.toggleSystemPanelButton.click();
    await expect(shell.systemPanel).toBeVisible();

    // An overlay, so the page underneath keeps its width.
    const runAfter = await model.slot(3).boundingBox();
    expect(runAfter!.width).toBeCloseTo(runBefore!.width, 0);

    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - window.innerWidth,
    );
    expect(overflow).toBeLessThanOrEqual(1);
  });
});
