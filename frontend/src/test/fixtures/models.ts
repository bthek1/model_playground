// Shared registry fixtures, imported by BOTH the MSW handlers (src/test/handlers.ts)
// and the Playwright mock-API fixture (e2e/fixtures/mockApi.ts). Keeping one
// source of truth means the unit-test mocks and the E2E mocks cannot drift out
// of sync with each other — or with the shapes in src/types/models.ts.

import type { ModelCard } from "@/types/models"

export const mockModels: ModelCard[] = [
  {
    id: "11111111-1111-4111-8111-111111111111",
    slug: "tiny-matmul",
    name: "Tiny Matmul",
    task: "custom",
    description: "A minimal WGSL matmul kernel used for smoke tests.",
    weights_url: "https://models.example.com/tiny-matmul.bin",
    config: { dims: 64 },
    size_bytes: 16384,
    license: "MIT",
    is_public: true,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  },
  {
    id: "22222222-2222-4222-8222-222222222222",
    slug: "whisper-tiny-en",
    name: "Whisper Tiny (en)",
    task: "audio",
    description: "Speech recognition running via Transformers.js.",
    weights_url: "https://models.example.com/whisper-tiny-en",
    config: { quantized: true },
    size_bytes: 39000000,
    license: "Apache-2.0",
    is_public: true,
    created_at: "2026-01-02T00:00:00Z",
    updated_at: "2026-01-02T00:00:00Z",
  },
]
