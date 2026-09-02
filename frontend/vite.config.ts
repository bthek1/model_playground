import { defineConfig } from "vitest/config";
import { loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import { TanStackRouterVite } from "@tanstack/router-vite-plugin";
import tailwindcss from "@tailwindcss/vite";
import basicSsl from "@vitejs/plugin-basic-ssl";
import { resolve } from "path";

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, __dirname, "");
  return {
    plugins: [
      react(),
      TanStackRouterVite({ routeFileIgnorePattern: ".(test|spec).(tsx?|js)" }),
      tailwindcss(),
      // Serve the dev server over HTTPS (self-signed cert). WebGPU (navigator.gpu)
      // is only exposed in a secure context — HTTPS or localhost — so a LAN IP
      // like https://192.168.2.106:5180 needs TLS for the GPU API to appear.
      basicSsl(),
    ],
    // Pre-bundle the ONNX Runtime entry the enhancement worker imports. Left to
    // discovery, Vite first meets it inside a Web Worker mid-session, re-optimises,
    // and triggers a full page reload — which resets a route that had just started
    // loading a model. Naming it here means the optimiser has it before the server
    // accepts a request. (`@huggingface/transformers` imports the same subpath, so
    // this is one shared bundle, not a second copy — see audio/enhance/session.ts.)
    optimizeDeps: {
      include: ["onnxruntime-web/webgpu"],
    },
    resolve: {
      alias: {
        "@": resolve(__dirname, "./src"),
      },
    },
    server: {
      host: "0.0.0.0",
      port: 5180,
      // Proxy the API through the dev server so the browser only ever talks to
      // the origin it loaded the page from. Hitting http://localhost:8006
      // directly from an https://192.168.x.x page is blocked three ways: mixed
      // content, CORS, and the browser's Local Network Access gate.
      proxy: {
        "/api": {
          target: env.VITE_API_PROXY_TARGET || "http://localhost:8006",
          changeOrigin: true,
          secure: false,
        },
      },
    },
    test: {
      environment: "happy-dom",
      setupFiles: ["./src/test/setup.ts"],
      globals: true,
      // Vitest's default `include` glob matches `**/*.spec.ts`, which would
      // sweep up the Playwright specs in `e2e/` and fail on importing
      // @playwright/test. E2E belongs to Playwright; Vitest owns `src/` only.
      include: ["src/**/*.{test,spec}.?(c|m)[jt]s?(x)"],
      exclude: ["node_modules/**", "dist/**", "e2e/**"],
    },
  };
});
