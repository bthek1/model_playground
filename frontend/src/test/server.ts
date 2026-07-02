import { setupServer } from "msw/node"

import { handlers } from "./handlers"

/** Shared MSW server. Import `server` in tests to add handlers via `server.use(...)`. */
export const server = setupServer(...handlers)
