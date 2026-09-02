import {
  taskCategories,
  tasksBySlug,
} from "../../src/components/layout/taskTaxonomy";
import { AppShell } from "../pages/AppShell";
import { expect, test } from "../fixtures/base";

// The sidebar is taxonomy-driven, so these tests are driven from the same data.
// Adding a task to taskTaxonomy.ts automatically extends this coverage.

// Real routes: everything whose `to` is not the /tasks/$slug placeholder.
const realTasks = Object.values(tasksBySlug).filter(
  (t) => !t.to.startsWith("/tasks/"),
);
// One placeholder task is enough to prove the fallback path.
const placeholderTask = Object.values(tasksBySlug).find((t) =>
  t.to.startsWith("/tasks/"),
)!;

test.describe("sidebar navigation", () => {
  test("renders every taxonomy category", async ({ page, mockApi }) => {
    await mockApi();
    const shell = new AppShell(page);
    await page.goto("/home");

    for (const category of taskCategories) {
      await expect(shell.category(category.label)).toBeVisible();
    }
  });

  test("expanding a category reveals its tasks", async ({ page, mockApi }) => {
    await mockApi();
    const shell = new AppShell(page);
    await page.goto("/home");

    const category = taskCategories.find((c) => c.label === "Theory")!;
    await expect(shell.taskLink(category.tasks[0].label)).toBeHidden();
    await shell.expandCategory("Theory");
    for (const item of category.tasks) {
      await expect(shell.taskLink(item.label)).toBeVisible();
    }
  });

  // Each implemented task must actually navigate to its real route.
  for (const item of realTasks) {
    test(`"${item.label}" navigates to ${item.to}`, async ({
      page,
      mockApi,
    }) => {
      await mockApi();
      await page.goto(item.to);
      await expect(page).toHaveURL(new RegExp(`${item.to}$`));
      // The route rendered something of its own, not the placeholder card.
      await expect(
        page.getByText("This task isn't available in the playground yet."),
      ).toHaveCount(0);
      await expect(page.getByRole("heading").first()).toBeVisible();
    });
  }

  test(`unimplemented task "${placeholderTask.label}" shows the placeholder`, async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    await page.goto(placeholderTask.to);
    // CardTitle renders a <div data-slot="card-title">, not a heading element.
    await expect(
      page.locator('[data-slot="card-title"]', {
        hasText: placeholderTask.label,
      }),
    ).toBeVisible();
    await expect(
      page.getByText("This task isn't available in the playground yet."),
    ).toBeVisible();
  });

  test("an unknown slug renders the unknown-task card, not a crash", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    await page.goto("/tasks/definitely-not-a-task");
    await expect(
      page.locator('[data-slot="card-title"]', { hasText: "Unknown task" }),
    ).toBeVisible();
    await expect(page.getByText("definitely-not-a-task")).toBeVisible();
  });

  test("clicking through the sidebar navigates and marks the link current", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const shell = new AppShell(page);
    await page.goto("/home");

    await shell.expandCategory("Theory");
    await shell.taskLink("Tensor Arithmetic").click();

    await expect(page).toHaveURL(/\/tensor$/);
    await expect(shell.taskLink("Tensor Arithmetic")).toHaveAttribute(
      "aria-current",
      "page",
    );
  });

  test("the active route's category auto-expands on deep link", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const shell = new AppShell(page);
    await page.goto("/tensor");

    await expect(shell.category("Theory")).toHaveAttribute(
      "aria-expanded",
      "true",
    );
    await expect(shell.taskLink("Tensor Arithmetic")).toBeVisible();
  });

  test("expanded-category state persists across navigation", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const shell = new AppShell(page);
    await page.goto("/home");

    await shell.expandCategory("Audio");
    await shell.taskLink("Text to Speech").click();
    await expect(page).toHaveURL(/\/text-to-speech$/);
    // Still open after the route change (store/ui.ts holds it).
    await expect(shell.category("Audio")).toHaveAttribute(
      "aria-expanded",
      "true",
    );
  });

  test("browser back and forward work", async ({ page, mockApi }) => {
    await mockApi();
    await page.goto("/home");
    await page.goto("/tensor");
    await expect(page).toHaveURL(/\/tensor$/);

    await page.goBack();
    await expect(page).toHaveURL(/\/home$/);
    await page.goForward();
    await expect(page).toHaveURL(/\/tensor$/);
  });
});
