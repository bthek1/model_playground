import type {
  CreateInferenceRunInput,
  InferenceRun,
  ModelCard,
} from "@/types/models";

import { apiClient } from "./client";

export async function listModels(): Promise<ModelCard[]> {
  const { data } = await apiClient.get<ModelCard[]>("/api/registry/models/");
  return data;
}

export async function getModel(slug: string): Promise<ModelCard> {
  const { data } = await apiClient.get<ModelCard>(
    `/api/registry/models/${slug}/`,
  );
  return data;
}

export async function createInferenceRun(
  input: CreateInferenceRunInput,
): Promise<InferenceRun> {
  const { data } = await apiClient.post<InferenceRun>(
    "/api/registry/runs/",
    input,
  );
  return data;
}
