import { AppShell } from "../pages/AppShell";
import { AUTH_FILE } from "../fixtures/paths";
import { expect, test } from "../fixtures/base";

// These start already signed in, using the storage state the `setup` project
// captured by logging in through the real UI once. That exercises the whole
// real-auth path — JWT issue, persistence, and reuse across a fresh context —
// without every spec paying for a login.
test.use({ storageState: AUTH_FILE });

test.describe("@backend signed-in session", () => {
  test("restores the session from saved storage state", async ({ page }) => {
    const shell = new AppShell(page);
    await page.goto("/home");

    // Sign-out only renders when /api/accounts/me/ resolved against real Django.
    await expect(shell.signOutButton).toBeVisible();
    const email = process.env.E2E_USER_EMAIL ?? "e2e@example.com";
    await expect(shell.navbar).toContainText(/Eee Tootoo|e2e@example\.com/);
    expect(email).toBeTruthy();
  });

  test("carries the token into a real registry call", async ({ page }) => {
    const response = page.waitForResponse(
      (r) => r.url().includes("/api/registry/models/") && r.status() === 200,
    );
    await page.goto("/home");
    const models = await (await response).json();
    expect(Array.isArray(models)).toBe(true);
    // Whatever the catalog holds, the card must not be in its error state.
    await expect(page.getByText("Could not load the catalog.")).toHaveCount(0);
  });

  test("the landing page bounces a signed-in user to /home", async ({
    page,
  }) => {
    await page.goto("/");
    // routes/index.tsx redirects once useMe() resolves.
    await expect(page).toHaveURL(/\/home$/);
  });
});
