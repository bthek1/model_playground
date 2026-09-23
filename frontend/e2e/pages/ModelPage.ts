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

  /**
   * The LOAD slot's action, shown while the model is `idle`.
   *
   * Matches the cached label too. The button reads "Load model (cached)" once
   * the weights are in the browser cache, and a pattern anchored to the
   * uncached wording silently stops matching the moment a `@slow` spec runs
   * second on the same profile — `load()` then cannot click the only control
   * that starts a download. A spec asserting one *specific* wording still says
   * so itself, via `button(/^Load model \(cached\)$/)`.
   */
  get loadButton(): Locator {
    return this.main.getByRole("button", {
      name: /^Load model( \(cached\))?$/,
    });
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

  /**
   * Wait out a real model download + warm-up. Only for `@slow` specs.
   *
   * Fails fast if the page falls back to `idle` mid-load rather than spending
   * the whole timeout on a page that is never going to become ready. The dev
   * server can reload itself while re-optimising dependencies (see the
   * `ensureMounted` note in `fixtures/base.ts`), and a reload resets Machine A
   * — so the symptom is an eight-minute wait ending in a screenshot of an
   * untouched LOAD slot, which says nothing about why.
   */
  async waitForReady(timeout = 8 * 60 * 1000): Promise<void> {
    const ready = this.readyStatus.waitFor({ timeout }).then(() => "ready");
    // Never rejects: whichever of the two loses the race is abandoned, and an
    // abandoned rejection would surface as an unhandled promise rejection that
    // fails the run for the wrong reason.
    const reset = this.loadButton
      .waitFor({ state: "visible", timeout })
      .then(() => "idle")
      .catch(() => "never");
    const first = await Promise.race([
      ready,
      // Only treat a *return* to idle as a reset — the click itself takes a
      // moment to move the machine out of idle, so give the load a head start.
      new Promise<void>((r) => setTimeout(r, 15_000)).then(() => reset),
    ]);
    if (first === "idle") {
      throw new Error(
        "The LOAD slot went back to idle mid-load — the page reloaded " +
          "underneath the download (dev-server dependency re-optimisation is " +
          "the usual cause). The model never reached ready.",
      );
    }
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
