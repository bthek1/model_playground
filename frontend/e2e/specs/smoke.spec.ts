import { AppShell } from "../pages/AppShell";
import { expect, test } from "../fixtures/base";

// Console messages that are pre-existing app issues, not regressions this suite
// introduced. Each entry needs a reason and a place to fix it — the list should
// shrink, never grow silently.
const KNOWN_CONSOLE_ISSUES = [
  // HeroBanner renders <Button render={<Link/>} />, so Base UI's default
  // nativeButton={true} lands on an <a>. Fix: pass nativeButton={false} at
  // src/components/home/HeroBanner.tsx:14-15, then drop this entry.
  /Base UI: A component that acts as a button expected a native <button>/,
];

function unexpected(messages: string[]): string[] {
  return messages.filter(
    (m) => !KNOWN_CONSOLE_ISSUES.some((known) => known.test(m)),
  );
}

// Runs on chromium AND firefox — this is the cross-browser subset.
test.describe("@smoke app shell", () => {
  test("landing page renders without console errors", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const errors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(msg.text());
    });
    page.on("pageerror", (err) => errors.push(err.message));

    await page.goto("/");
    await expect(page.locator("body")).toBeVisible();
    expect(unexpected(errors)).toEqual([]);
  });

  test("app shell renders navbar and sidebar on an in-app route", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const shell = new AppShell(page);

    await page.goto("/home");
    await expect(shell.navbar).toBeVisible();
    await expect(shell.sidebar).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "Model Playground" }),
    ).toBeVisible();
  });

  test("theme toggle cycles and the choice survives a reload", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const shell = new AppShell(page);
    await page.goto("/home");

    // Cycle is light → dark → system; land on dark deterministically.
    for (let i = 0; i < 3; i++) {
      if ((await shell.currentTheme()) === "dark") break;
      await shell.themeToggle.click();
    }
    expect(await shell.currentTheme()).toBe("dark");
    expect(await shell.isDarkClassApplied()).toBe(true);

    await page.reload();
    expect(await shell.currentTheme()).toBe("dark");
    expect(await shell.isDarkClassApplied()).toBe(true);
  });

  test("sidebar can be collapsed to the icon rail", async ({
    page,
    mockApi,
  }) => {
    await mockApi();
    const shell = new AppShell(page);
    await page.goto("/home");

    await expect(shell.sidebar).toContainText("Model Playground");
    await shell.toggleSidebarButton.click();
    // Collapsed rail shows the "MP" monogram instead of the full name.
    await expect(shell.sidebar).toContainText("MP");
  });
});
