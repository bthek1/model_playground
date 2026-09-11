import { expect, type Locator, type Page } from "@playwright/test";

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

  /** The LOAD slot's progress block, present only while loading. */
  get loadProgress(): Locator {
    return this.main.getByTestId("load-progress");
  }

  /** Cancel, offered beside the bar while a download is in flight. */
  get cancelLoad(): Locator {
    return this.main.getByTestId("load-cancel");
  }

  /** The "already downloaded" badge on a model row. */
  cachedBadge(modelId: string): Locator {
    return this.main.getByTestId(`model-cached-${modelId}`);
  }


  /** "…86M params · ≈172 MB on WebGPU · 88 MB on WASM" */
  get sizeNote(): Locator {
    return this.main.getByTestId("model-size-note");
  }

  /** The amber size-before-load guardrail, shown only for large models. */
  get largeModelWarning(): Locator {
    return this.main.getByTestId("model-size-warning");
  }

  /** "Model ready · running on WASM" */
  get readyStatus(): Locator {
    return this.main.getByTestId("model-ready");
  }

  /** The LOAD slot's action, shown while the model is `idle`. */
  get loadButton(): Locator {
    return this.main.getByRole("button", { name: /^Load model$/ });
  }

  /** Shown in place of the load action after a failed load. */
  get retryButton(): Locator {
    return this.main.getByRole("button", { name: /^Retry$/ });
  }

  /**
   * Load the model from the LOAD slot. Nothing downloads before this — `idle` is
   * every weight-downloading route's default state.
   */
  async load(): Promise<void> {
    await this.loadButton.click();
  }

  /**
   * Press the RUN slot's trigger, whatever this route calls it.
   *
   * Choosing an input never starts an inference (model-page-pattern.md §1.6):
   * a sample click, a file drop and a recording all land in INPUT and stop, so
   * every spec that wants a result asks for one here.
   *
   * Waits for *enabled*, not merely present: the button exists from the first
   * render and only lights up once a model is ready and an input is held, and
   * a Playwright click on a disabled button times out with a far less useful
   * message than "still disabled".
   */
  async run(name: string | RegExp): Promise<void> {
    const trigger = this.button(name);
    await trigger.waitFor();
    await expect(trigger).toBeEnabled();
    await trigger.click();
  }

  /**
   * Block every Hugging Face request so a spec can assert the page's shell and
   * its failure path without pulling hundreds of megabytes.
   *
   * Uses a URL predicate, never a glob: `**\/api/**`-style globs also match the
   * dev server's own module URLs and stop the app booting (see e2e-testing.md).
   */
  async blockModelDownloads(): Promise<void> {
    await this.page.route(
      (url) => url.hostname.endsWith("huggingface.co"),
      (route) => route.abort(),
    );
  }

  /** Wait out a real model download + warm-up. Only for `@slow` specs. */
  async waitForReady(timeout = 8 * 60 * 1000): Promise<void> {
    await this.readyStatus.waitFor({ timeout });
  }

  /** The backend the model actually loaded on, from the ready line. */
  async backend(): Promise<string> {
    const text = await this.readyStatus.innerText();
    // `\s+`, not a literal space: the ready line is an `inline-flex` row and the
    // backend sits in its own span, so Chromium's `innerText` puts a newline
    // between "running on" and the name. A single-space pattern silently
    // returns "" and every backend assertion becomes vacuous.
    return (text.match(/running on\s+(\w+)/i)?.[1] ?? "").toLowerCase();
  }

  /** A button inside the route content, never the sidebar. */
  button(name: string | RegExp): Locator {
    return this.main.getByRole("button", { name });
  }
}
