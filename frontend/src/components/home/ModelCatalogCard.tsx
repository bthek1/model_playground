import { Layers } from "lucide-react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useModels } from "@/hooks/useModels";

function ModelCatalog() {
  const { data: models, isLoading, isError } = useModels();

  if (isLoading) {
    return <p className="text-sm text-muted-foreground">Loading catalog…</p>;
  }
  if (isError) {
    return (
      <p className="text-sm text-destructive">Could not load the catalog.</p>
    );
  }
  if (!models || models.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        No models yet. Add one via the Django admin or{" "}
        <code>POST /api/registry/models/</code>. See
        docs/guides/adding-a-model.md.
      </p>
    );
  }

  return (
    <ul className="divide-y">
      {models.map((model) => (
        <li key={model.id} className="flex items-baseline gap-3 py-2 text-sm">
          <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs uppercase">
            {model.task}
          </span>
          <span className="font-medium">{model.name}</span>
          <span className="truncate text-muted-foreground">
            {model.description}
          </span>
        </li>
      ))}
    </ul>
  );
}

/** Dashboard card listing registered models from the backend registry. */
export function ModelCatalogCard() {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Layers className="size-4" /> Model Catalog
        </CardTitle>
        <CardDescription>
          Registered models, served from the backend registry.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ModelCatalog />
      </CardContent>
    </Card>
  );
}
