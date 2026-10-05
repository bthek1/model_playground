/**
 * Build-time feature flags, read from the environment in exactly one place.
 *
 * Components import the constant rather than checking `import.meta.env`
 * themselves, so a test can flip a flag with one `vi.mock("@/lib/features")`
 * and a grep for the flag finds every behaviour it changes.
 */

/**
 * Whether this build has a Django backend behind `/api` (#57).
 *
 * `VITE_BACKEND=off` is the static deploy (S3 + CloudFront): every model page
 * runs in the browser and fetches its weights from the Hugging Face CDN, so the
 * app is complete without a backend — but anything that calls `/api` would get
 * the SPA's `index.html` back with a 200. With the flag off the app never makes
 * that call: no `useMe`, no login/signup, no registry catalogue.
 *
 * Anything other than `off` (including unset) keeps today's behaviour.
 */
export const BACKEND_ENABLED: boolean = import.meta.env.VITE_BACKEND !== "off";

/**
 * PostHog's **project** key (`phc_…`) — public by design and write-only (#60).
 * Set only on CI's shipped build, as a repository *variable*; unset in dev,
 * Vitest and the dev-server E2E run, which is what keeps every "no outbound
 * request" assertion meaning what it says. Never the personal `phx_` key:
 * `scripts/check-bundle.mjs` fails the build if one reaches `dist/`.
 */
export const POSTHOG_KEY: string = import.meta.env.VITE_POSTHOG_KEY ?? "";

/** Where events go. `/ingest` is the same-origin CloudFront behaviour. */
export const POSTHOG_HOST: string = import.meta.env.VITE_POSTHOG_HOST || "/ingest";

/**
 * Anonymous product analytics (#60). Off unless the build carries a key; when
 * off, `src/analytics/` imports nothing and sends nothing.
 */
export const ANALYTICS_ENABLED: boolean = POSTHOG_KEY !== "";
