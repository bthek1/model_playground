// Mirrors apps/registry serializers. See docs/standards/api-contracts.md.

export type ModelTask = "llm" | "vision" | "embedding" | "audio" | "custom";

export interface ModelCard {
  id: string;
  slug: string;
  name: string;
  task: ModelTask;
  description: string;
  weights_url: string;
  config: Record<string, unknown>;
  size_bytes: number | null;
  license: string;
  is_public: boolean;
  created_at: string;
  updated_at: string;
}

export type InferenceRunStatus = "completed" | "failed";

export interface InferenceRun {
  id: string;
  model: string;
  model_slug: string;
  status: InferenceRunStatus;
  params: Record<string, unknown>;
  metrics: Record<string, unknown>;
  created_at: string;
}

export interface CreateInferenceRunInput {
  model: string;
  status?: InferenceRunStatus;
  params?: Record<string, unknown>;
  metrics?: Record<string, unknown>;
}
