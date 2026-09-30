/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** `off` builds the static, backend-free app — read via `@/lib/features`. */
  readonly VITE_BACKEND?: "on" | "off";
  readonly VITE_API_BASE_URL?: string;
  readonly VITE_API_PROXY_TARGET?: string;
}
