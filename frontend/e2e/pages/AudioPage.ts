import type { Locator, Page } from "@playwright/test";

/**
 * Shared page object for the three in-browser audio routes (`/asr`,
 * `/audio-classification`, `/text-to-speech`). They deliberately share a shell:
 * a `ModelPicker` (buttons + the size-before-load note and warning) above a
 * `ModelStatus` line, so one page object serves all three.
 */
export class AudioPage {
  constructor(private readonly page: Page) {}

  /** Route content only — the sidebar also contains category buttons. */
  private get main(): Locator {
    return this.page.locator("main");
  }

  modelButton(label: string | RegExp): Locator {
    return this.main.getByRole("button", { name: label });
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

  /** The route's error banner (load failure, decode failure, …). */
  get error(): Locator {
    return this.main.locator(".text-destructive").first();
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
