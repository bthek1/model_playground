import type { Locator, Page } from "@playwright/test";

/**
 * The four-slot model page (docs/standards/model-page-pattern.md): every task
 * route renders Select → Load → Run → Output, in that order, always all four.
 *
 * This base holds what is true of *every* task page. `AudioPage` and
 * `TensorPage` extend it with what is specific to their modality.
 *
 * Everything is scoped to `<main>`: the sidebar carries task-category buttons
 * whose names collide with route buttons ("Computer Vision" vs "Compute").
 */
export class ModelPageObject {
  protected readonly main: Locator;

  constructor(protected readonly page: Page) {
    this.main = page.locator("main");
  }

  /** The four numbered bands. */
  get slots(): Locator {
    return this.main.locator("[data-testid^='slot-']");
  }

  slot(step: 1 | 2 | 3 | 4): Locator {
    return this.main.getByTestId(`slot-${step}`);
  }

  /** The OUTPUT slot's card. Present whether or not there is a result. */
  get outputPanel(): Locator {
    return this.main.getByTestId("output-panel");
  }

  /** The "here is what you'll get" state, before any run. */
  get emptyOutput(): Locator {
    return this.main.getByTestId("output-empty");
  }

  get runningOutput(): Locator {
    return this.main.getByTestId("output-running");
  }

  /** Any error, in whichever slot produced it (`ErrorNote` renders role=alert). */
  get error(): Locator {
    return this.main.getByRole("alert").first();
  }

  /** A button inside the route content, never the sidebar. */
  button(name: string | RegExp): Locator {
    return this.main.getByRole("button", { name });
  }
}
