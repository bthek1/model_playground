import type { Locator } from "@playwright/test";

import { ModelPageObject } from "./ModelPage";

/**
 * Shared page object for the in-browser audio routes (`/asr`,
 * `/audio-classification`, `/text-to-speech`, `/text-to-audio`,
 * `/audio-to-audio`). They share the four-slot model page
 * (docs/standards/model-page-pattern.md): a `ModelPicker` in SELECT, a
 * `ModelStatus` in LOAD, transport in RUN, an `OutputPanel` in OUTPUT — so one
 * page object serves all of them.
 */
export class AudioPage extends ModelPageObject {
  modelButton(label: string | RegExp): Locator {
    return this.button(label);
  }

  /** "…74M params · ≈140 MB on WebGPU · 221 MB on WASM" */
  get sizeNote(): Locator {
    return this.main.getByTestId("model-size-note");
  }

  /** The amber size-before-load guardrail, shown only for large models. */
  get largeModelWarning(): Locator {
    return this.main.getByTestId("model-size-warning");
  }

  /** "Model ready · running on WASM" */
  get readyStatus(): Locator {
    return this.main.getByText(/Model ready/);
  }

  get warmingStatus(): Locator {
    return this.main.getByText(/Warming up the model/);
  }

  get loadingStatus(): Locator {
    return this.main.getByText(/Loading model/);
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
    return (text.match(/running on (\w+)/i)?.[1] ?? "").toLowerCase();
  }
}
