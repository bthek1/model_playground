import { LoginPage } from "../pages/LoginPage";
import { AppShell } from "../pages/AppShell";
import { MOCK_USER } from "../fixtures/mockApi";
import { expect, test } from "../fixtures/base";

test.describe("authentication (mocked API)", () => {
  test("signs in and lands on /home", async ({ page, mockApi }) => {
    await mockApi({ authenticated: true });
    const login = new LoginPage(page);

    await login.goto();
    await login.signIn("e2e@example.com", "correct-horse");

    await expect(page).toHaveURL(/\/home$/);
    // Tokens were persisted by useLogin().
    const access = await page.evaluate(() =>
      localStorage.getItem("access_token"),
    );
    expect(access).toBe("mock.access");
  });

  test("shows an error and stays put on bad credentials", async ({ page }) => {
    await page.route("**/api/token/", (route) =>
      route.fulfill({
        status: 401,
        contentType: "application/json",
        body: JSON.stringify({ detail: "No active account found" }),
      }),
    );
    const login = new LoginPage(page);

    await login.goto();
    await login.signIn("e2e@example.com", "wrong");

    await expect(
      page.getByText("Invalid credentials. Please try again."),
    ).toBeVisible();
    await expect(page).toHaveURL(/\/login$/);
  });

  test("blocks submit on a malformed email", async ({ page }) => {
    const login = new LoginPage(page);
    await login.goto();
    await login.emailInput.fill("not-an-email");
    await login.passwordInput.fill("something");
    await login.submitButton.click();

    // The input is type="email", so the browser's own constraint validation
    // rejects it before react-hook-form runs — the Zod message never renders in
    // a real browser. (The Zod branch is covered by the unit tests, which use
    // happy-dom and therefore skip native validation.)
    await expect(login.emailInput).toHaveJSProperty("validity.valid", false);
    await expect(page).toHaveURL(/\/login$/);
  });

  test("surfaces the Zod message for an empty password", async ({ page }) => {
    const login = new LoginPage(page);
    await login.goto();
    await login.emailInput.fill("e2e@example.com");
    await login.submitButton.click();

    await expect(page.getByText("Password is required")).toBeVisible();
  });

  test("the sign-up link reaches /signup", async ({ page }) => {
    const login = new LoginPage(page);
    await login.goto();
    await login.signUpLink.click();
    await expect(page).toHaveURL(/\/signup$/);
  });

  test("shows the signed-in user and signs out", async ({
    page,
    mockApi,
    signedIn,
  }) => {
    await signedIn();
    await mockApi({ authenticated: true });
    const shell = new AppShell(page);

    await page.goto("/home");
    await expect(shell.navbar).toContainText("Eee Tootoo");

    await shell.signOutButton.click();
    await expect(page).toHaveURL(/\/login$/);
    const tokens = await page.evaluate(() => [
      localStorage.getItem("access_token"),
      localStorage.getItem("refresh_token"),
    ]);
    expect(tokens).toEqual([null, null]);
  });

  test("recovers from a 401 via the silent refresh in api/client.ts", async ({
    page,
    signedIn,
  }) => {
    await signedIn();

    let meCalls = 0;
    let refreshCalls = 0;

    await page.route("**/api/accounts/me/", (route) => {
      meCalls += 1;
      // Expire the access token exactly once; the retry must succeed.
      if (meCalls === 1) {
        return route.fulfill({
          status: 401,
          contentType: "application/json",
          body: JSON.stringify({ detail: "Token expired" }),
        });
      }
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(MOCK_USER),
      });
    });

    await page.route("**/api/token/refresh/", (route) => {
      refreshCalls += 1;
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ access: "refreshed.access" }),
      });
    });

    await page.route("**/api/registry/models/", (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: "[]",
      }),
    );

    const shell = new AppShell(page);
    await page.goto("/home");

    // The user is shown, which can only happen after the retried /me succeeded.
    await expect(shell.navbar).toContainText("Eee Tootoo");
    expect(refreshCalls).toBe(1);
    expect(meCalls).toBeGreaterThanOrEqual(2);
    // The refreshed token replaced the stale one in localStorage.
    const access = await page.evaluate(() =>
      localStorage.getItem("access_token"),
    );
    expect(access).toBe("refreshed.access");
  });

  test("a failed refresh clears both tokens", async ({ page, signedIn }) => {
    await signedIn();

    await page.route("**/api/accounts/me/", (route) =>
      route.fulfill({
        status: 401,
        contentType: "application/json",
        body: JSON.stringify({ detail: "Token expired" }),
      }),
    );
    await page.route("**/api/token/refresh/", (route) =>
      route.fulfill({
        status: 401,
        contentType: "application/json",
        body: JSON.stringify({ detail: "Refresh expired" }),
      }),
    );
    await page.route("**/api/registry/models/", (route) =>
      route.fulfill({ status: 200, contentType: "application/json", body: "[]" }),
    );

    await page.goto("/home");

    await expect
      .poll(() =>
        page.evaluate(() => localStorage.getItem("access_token")),
      )
      .toBeNull();
    expect(
      await page.evaluate(() => localStorage.getItem("refresh_token")),
    ).toBeNull();
  });
});

test.describe("@backend authentication (real Django)", () => {
  test("signs in against the real API", async ({ page }) => {
    const login = new LoginPage(page);
    await login.goto();
    await login.signIn(
      process.env.E2E_USER_EMAIL ?? "e2e@example.com",
      process.env.E2E_USER_PASSWORD ?? "e2e-password-123",
    );

    await expect(page).toHaveURL(/\/home$/);
    await expect(new AppShell(page).signOutButton).toBeVisible();
  });

  test("rejects a wrong password against the real API", async ({ page }) => {
    const login = new LoginPage(page);
    await login.goto();
    await login.signIn(
      process.env.E2E_USER_EMAIL ?? "e2e@example.com",
      "definitely-the-wrong-password",
    );

    await expect(
      page.getByText("Invalid credentials. Please try again."),
    ).toBeVisible();
  });
});
