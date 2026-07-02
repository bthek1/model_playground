import { useQuery } from "@tanstack/react-query";

import { listModels } from "@/api/models";
import { queryKeys } from "@/api/queryKeys";

/** Fetch the browser-runnable model catalog from the registry. */
export function useModels() {
  return useQuery({
    queryKey: queryKeys.models.all,
    queryFn: listModels,
  });
}
