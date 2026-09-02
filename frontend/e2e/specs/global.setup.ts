import { test as setup, expect } from "@playwright/test";

import { LoginPage } from "../pages/LoginPage";
import { AUTH_FILE } from "../fixtures/paths";

/**
 * Log in once through the real UI against the real Django API and save the
 * resulting storage state, so `@backend` specs can start already signed in.
 * Requires `just be-seed-e2e` to have created the user.
 */
setup("authenticate", async ({ page }) => {
  const login = new LoginPage(page);
  await login.goto();
  await login.signIn(
    process.env.E2E_USER_EMAIL ?? "e2e@example.com",
    process.env.E2E_USER_PASSWORD ?? "e2e-password-123",
  );

  await expect(page).toHaveURL(/\/home$/);
  await page.context().storageState({ path: AUTH_FILE });
});
