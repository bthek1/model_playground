import type { Locator, Page } from "@playwright/test";

/**
 * The navbar + sidebar shell rendered by AppLayout for every non-public route.
 * Locators are role/label-based on purpose: the UI is built on Base UI (not
 * Radix), so there are no `data-state` conventions to lean on.
 *
 * The sidebar is addressed by test id rather than by its `aside` tag: the
 * system panel is a complementary region too, so a bare tag locator becomes
 * ambiguous the moment the panel is open.
 */
export class AppShell {
  readonly sidebar: Locator;
  readonly navbar: Locator;
  readonly toggleSidebarButton: Locator;
  readonly themeToggle: Locator;
  readonly signOutButton: Locator;
  readonly systemPanel: Locator;
  readonly toggleSystemPanelButton: Locator;

  constructor(private readonly page: Page) {
    this.sidebar = page.getByTestId("sidebar");
    this.navbar = page.locator("header");
    this.toggleSidebarButton = page.getByRole("button", {
      name: "Toggle sidebar",
    });
    this.themeToggle = page.getByRole("button", { name: /^Theme:/ });
    this.signOutButton = page.getByRole("button", { name: "Sign out" });
    this.systemPanel = page.getByTestId("right-panel");
    this.toggleSystemPanelButton = page.getByRole("button", {
      name: "Toggle system panel",
    });
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
