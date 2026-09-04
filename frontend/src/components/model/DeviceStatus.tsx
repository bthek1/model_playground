// The LOAD slot for routes whose "load" is a device probe and a shader compile
// rather than a weight download — the raw-WebGPU pages (tensor arithmetic,
// training).
//
// Nothing here is deferred: acquiring a GPUDevice and compiling WGSL is fast and
// free, so these routes auto-load (model-page-pattern.md §7). The band still
// renders, so a raw-WebGPU page keeps the same four-band rhythm as a page that
// downloads 200 MB — and it answers the question the user actually has, which is
// whether this will run on the GPU or not at all.

import { CheckCircle2, Cpu, Loader2 } from "lucide-react";

import { ErrorNote } from "@/components/model/ErrorNote";
import type { WebGPUCapabilities } from "@/webgpu/types";

/** Why a status means the page can't compute, in the user's terms. */
const UNAVAILABLE: Record<string, string> = {
  unsupported:
    "This browser doesn't expose WebGPU. It needs a secure context (HTTPS or localhost) — and on Firefox, dom.webgpu.enabled in about:config.",
  "no-adapter":
    "WebGPU is available but no GPU adapter was offered, so there is nothing to compute on.",
  "no-device":
    "A GPU adapter was found but requesting a device failed. Another tab may hold the GPU, or the driver refused.",
};

export function DeviceStatus({
  capabilities,
  loading,
}: {
  capabilities: WebGPUCapabilities | null;
  loading: boolean;
}) {
  if (loading) {
    return (
      <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <Loader2 className="size-3.5 animate-spin" />
        Probing the GPU…
      </p>
    );
  }

  const status = capabilities?.status ?? "unsupported";

  if (status !== "ready") {
    return <ErrorNote message={UNAVAILABLE[status] ?? UNAVAILABLE.unsupported} />;
  }

  const adapter = capabilities?.adapter;
  const name =
    [adapter?.vendor, adapter?.architecture].filter(Boolean).join(" ") ||
    adapter?.description ||
    "GPU";

  return (
    <p
      data-testid="device-ready"
      className="flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground"
    >
      <CheckCircle2 className="size-3.5 text-emerald-600 dark:text-emerald-500" />
      GPU ready · kernels compile on first use ·
      <span className="inline-flex items-center gap-1 font-medium">
        <Cpu className="size-3.5" />
        {name}
      </span>
      {capabilities?.isFallbackAdapter && (
        <span className="text-amber-600 dark:text-amber-500">
          (software fallback — expect slow compute)
        </span>
      )}
    </p>
  );
}
