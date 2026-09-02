import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// Resolved from this file's own location, NOT from process.cwd(). Playwright
// resolves a relative `storageState` against the working directory, so a run
// launched from the repo root instead of `frontend/` would silently write the
// saved session to `<repo>/e2e/.auth/` and every authenticated spec would then
// fail to find it.
const FIXTURES_DIR = dirname(fileURLToPath(import.meta.url));

/** Where the `setup` project saves the signed-in session. */
export const AUTH_FILE = resolve(FIXTURES_DIR, "../.auth/user.json");
