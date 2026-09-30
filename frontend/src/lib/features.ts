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
