import type { Locator, Page } from "@playwright/test";

/**
 * /login. Note the identity field is **email** — the backend uses
 * AUTH_USER_MODEL = accounts.CustomUser with email as the username field.
 */
export class LoginPage {
  readonly emailInput: Locator;
  readonly passwordInput: Locator;
  readonly submitButton: Locator;
  readonly signUpLink: Locator;

  constructor(private readonly page: Page) {
    this.emailInput = page.getByLabel("Email");
    this.passwordInput = page.getByLabel("Password");
    this.submitButton = page.getByRole("button", { name: /Sign in|Signing in/ });
    this.signUpLink = page.getByRole("link", { name: "Sign up" });
  }

  async goto(): Promise<void> {
    await this.page.goto("/login");
  }

  async signIn(email: string, password: string): Promise<void> {
    await this.emailInput.fill(email);
    await this.passwordInput.fill(password);
    await this.submitButton.click();
  }
}
