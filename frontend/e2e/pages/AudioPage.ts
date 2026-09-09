import type { Locator } from "@playwright/test";

import { ModelPageObject } from "./ModelPage";

/**
 * Shared page object for the in-browser audio routes (`/asr`,
 * `/audio-classification`, `/text-to-speech`, `/text-to-audio`,
 * `/audio-to-audio`). They share the four-slot model page
 * (docs/standards/model-page-pattern.md): a `ModelPicker` in SELECT, a
 * `ModelStatus` in LOAD, transport in RUN, an `OutputPanel` in OUTPUT — so one
 * page object serves all of them.
 *
 * Everything that is true of *any* model page — the size note, the ready line,
 * `load()`, `waitForReady()`, `backend()` — lives on `ModelPageObject`, so the
 * vision routes get it without a second copy. What is left here is what only
 * the audio pages say.
 */
export class AudioPage extends ModelPageObject {
  modelButton(label: string | RegExp): Locator {
    return this.button(label);
  }

  get warmingStatus(): Locator {
    return this.main.getByText(/Warming up the model/);
  }

  get loadingStatus(): Locator {
    return this.main.getByText(/Loading model/);
  }
}
