import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import { TanStackRouterVite } from "@tanstack/router-vite-plugin";
import tailwindcss from "@tailwindcss/vite";
import basicSsl from "@vitejs/plugin-basic-ssl";
import { resolve } from "path";

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    TanStackRouterVite({ routeFileIgnorePattern: ".(test|spec).(tsx?|js)" }),
    tailwindcss(),
    // Serve the dev server over HTTPS (self-signed cert). WebGPU (navigator.gpu)
    // is only exposed in a secure context — HTTPS or localhost — so a LAN IP
    // like https://192.168.2.106:5174 needs TLS for the GPU API to appear.
    basicSsl(),
  ],
  resolve: {
    alias: {
      "@": resolve(__dirname, "./src"),
    },
  },
  server: {
    host: "0.0.0.0",
    port: 5174,
  },
  test: {
    environment: "happy-dom",
    setupFiles: ["./src/test/setup.ts"],
    globals: true,
  },
});
