import { installMockApi } from "../fixtures/mockApi";
import { AudioPage } from "../pages/AudioPage";
import { silenceSpeechSilenceWav } from "../utils/wav";
import { expect, test } from "../fixtures/base";

// @slow — the live loop's silence gate (#55), with a real Whisper-base and a
// real microphone stream: Chromium's fake capture device playing
// 6 s silence → JFK (11 s) → 10 s silence, once.
//
// Everything the gate touches is mocked away somewhere else: the hook test fakes
// MediaRecorder and the decode, and the mocked E2E run never loads a model, so
// "Start listening" is never even enabled there. This is the one place the whole
// loop runs — Opus in, decode, gate, worker, transcript out.
//
// What it asserts is what the user can see: the page *says* it is skipping
// before anyone speaks, the words arrive when JFK does, the page says it is
// skipping again once he stops, and the final pass still transcribes the take.
// It does not count worker requests — the count on screen is the hook's own
// `skippedTicks`, and on a slow machine busy ticks drop anyway, so an exact
// number would pin the hardware rather than the gate.

const LEAD_SECONDS = 6;
const TAIL_SECONDS = 10;
const BUDGET_MS = 10 * 60 * 1000;

test.describe("@slow live ASR silence gate", () => {
  test.describe.configure({ timeout: BUDGET_MS });

  test("/asr skips silent updates while listening and still transcribes speech", async ({
    playwright,
    baseURL,
  }) => {
    // The fake mic reads its file at launch, so the file comes first and the
    // browser second — which is why this spec launches its own.
    const wav = await silenceSpeechSilenceWav(LEAD_SECONDS, TAIL_SECONDS);
    const browser = await playwright.chromium.launch({
      args: [
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
        `--use-file-for-fake-audio-capture=${wav}%noloop`,
      ],
    });
    try {
      const context = await browser.newContext({
        baseURL,
        ignoreHTTPSErrors: true,
        permissions: ["microphone"],
      });
      const page = await context.newPage();
      await installMockApi(page);
      const audio = new AudioPage(page);

      await page.goto("/asr");
      // The base fixture's mount check is bound to its own `page`; this one
      // re-does the part that matters (the dev server's rare empty first load).
      const heading = page.getByRole("heading", { name: /automatic speech recognition/i });
      if (!(await heading.isVisible({ timeout: 8_000 }).catch(() => false))) await page.reload();
      await expect(heading).toBeVisible();

      await audio.load();
      await audio.waitForReady(BUDGET_MS);
      await expect(page.getByTestId("skip-silence-toggle")).toBeChecked();

      await page.getByRole("button", { name: /start listening/i }).click();

      // 1. The lead-in: nothing said, so the page says it is skipping.
      const silent = page.getByTestId("asr-silent");
      await expect(silent).toBeVisible({ timeout: LEAD_SECONDS * 1000 });
      await expect(page.getByText(/ask not/i)).toHaveCount(0);

      // 2. JFK: the gate opens and the words arrive.
      await expect(page.getByText(/ask not what your country/i).first()).toBeVisible({
        timeout: 60_000,
      });

      // 3. The tail: silent again, and the transcript holds still rather than
      //    growing "you" lines.
      await expect(silent).toBeVisible({ timeout: 30_000 });

      // 4. Stop: the final pass is never gated.
      await page.getByRole("button", { name: /^stop$/i }).click();
      await expect(silent).toHaveCount(0);
      await expect(page.getByText(/ask not what your country/i).first()).toBeVisible({
        timeout: 60_000,
      });
    } finally {
      await browser.close();
    }
  });
});
