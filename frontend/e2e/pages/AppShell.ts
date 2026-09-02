import type { Locator, Page } from "@playwright/test";

/**
 * The navbar + sidebar shell rendered by AppLayout for every non-public route.
 * Locators are role/label-based on purpose: the UI is built on Base UI (not
 * Radix), so there are no `data-state` conventions to lean on.
 */
export class AppShell {
  readonly sidebar: Locator;
  readonly navbar: Locator;
  readonly toggleSidebarButton: Locator;
  readonly themeToggle: Locator;
  readonly signOutButton: Locator;

  constructor(private readonly page: Page) {
    this.sidebar = page.locator("aside");
    this.navbar = page.locator("header");
    this.toggleSidebarButton = page.getByRole("button", {
      name: "Toggle sidebar",
    });
    this.themeToggle = page.getByRole("button", { name: /^Theme:/ });
    this.signOutButton = page.getByRole("button", { name: "Sign out" });
  }

  /** The expand/collapse button for a sidebar task category. */
  category(label: string): Locator {
    return this.sidebar.getByRole("button", { name: label, exact: true });
  }

  /** A task link inside the sidebar. Only visible once its category is open. */
  taskLink(label: string): Locator {
    return this.sidebar.getByRole("link", { name: label, exact: true });
  }

  async expandCategory(label: string): Promise<void> {
    const button = this.category(label);
    if ((await button.getAttribute("aria-expanded")) !== "true") {
      await button.click();
    }
  }

  /** Current theme as reported by the toggle's aria-label ("Theme: dark"). */
  async currentTheme(): Promise<string> {
    const label = await this.themeToggle.getAttribute("aria-label");
    return (label ?? "").replace("Theme: ", "");
  }

  async isDarkClassApplied(): Promise<boolean> {
    return this.page.evaluate(() =>
      document.documentElement.classList.contains("dark"),
    );
  }
}
