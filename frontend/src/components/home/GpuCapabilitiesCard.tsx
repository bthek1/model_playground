import { CheckCircle2, Cpu, Loader2, XCircle } from "lucide-react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useWebGPU } from "@/hooks/useWebGPU";
import { detectBrowser } from "@/lib/browser";
import { formatBytes, titleCase } from "@/lib/format";
import type { WebGPUCapabilities, WebGPUStatus } from "@/webgpu/types";

// Human-readable summary for each capability status, keyed off the outcome of
// detectWebGPU() (which confirms a real GPUDevice can be acquired).
const STATUS_INFO: Record<
  WebGPUStatus,
  { label: string; hint: string; ok: boolean }
> = {
  ready: {
    label: "GPU accessible",
    hint: "This browser can acquire a GPU device and run WebGPU compute.",
    ok: true,
  },
  "no-device": {
    label: "No GPU access",
    hint: "An adapter was found but a GPU device could not be acquired — the driver may be blocked, out of resources, or unavailable.",
    ok: false,
  },
  "no-adapter": {
    label: "No GPU access",
    hint: "WebGPU is present but no adapter was offered — likely no compatible GPU or driver in this environment.",
    ok: false,
  },
  unsupported: {
    label: "WebGPU unsupported",
    hint: "This browser doesn't expose WebGPU. Use Chrome/Edge 113+, Firefox 141+, or Safari 26+.",
    ok: false,
  },
};

function CapabilityPanel() {
  const { capabilities, loading } = useWebGPU();

  if (loading) {
    return (
      <p className="flex items-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" /> Checking GPU access…
      </p>
    );
  }

  const status = capabilities?.status ?? "unsupported";
  const info = STATUS_INFO[status];
  const browser = detectBrowser();

  // WebGPU is only exposed in a secure context (HTTPS or localhost). On a plain
  // HTTP origin Chrome removes `navigator.gpu` entirely, which surfaces here as
  // "unsupported" — a far more common cause than an actually-incapable browser.
  const insecureContext =
    status === "unsupported" &&
    typeof window !== "undefined" &&
    window.isSecureContext === false;
  // Firefox on Linux/macOS still ships WebGPU behind a flag even on versions
  // that enable it by default on Windows — point those users at the flag.
  const firefoxNeedsFlag =
    status === "unsupported" &&
    !insecureContext &&
    /firefox/i.test(browser.name);

  let hint = info.hint;
  if (insecureContext) {
    hint =
      "This page isn't a secure context, so the browser hides WebGPU. Open it over HTTPS or via http://localhost (e.g. an SSH tunnel) — not a plain-HTTP IP address.";
  } else if (firefoxNeedsFlag) {
    hint =
      "Firefox hides WebGPU behind a flag on Linux and macOS. Open about:config, set dom.webgpu.enabled to true, then restart Firefox.";
  }

  return (
    <div className="space-y-4 text-sm">
      <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-1">
        <dt className="text-muted-foreground">Browser</dt>
        <dd>
          {browser.name}
          {browser.version ? ` ${browser.version}` : ""}
        </dd>
      </dl>

      <div
        className={`flex items-start gap-2 rounded-md border p-3 ${
          info.ok
            ? "border-emerald-500/30 bg-emerald-500/10"
            : "border-destructive/30 bg-destructive/10"
        }`}
      >
        {info.ok ? (
          <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-emerald-600 dark:text-emerald-400" />
        ) : (
          <XCircle className="mt-0.5 size-4 shrink-0 text-destructive" />
        )}
        <div>
          <p className="font-medium">
            {insecureContext
              ? "WebGPU hidden (insecure context)"
              : firefoxNeedsFlag
                ? "WebGPU disabled (Firefox flag)"
                : info.label}
          </p>
          <p className="text-muted-foreground">{hint}</p>
        </div>
      </div>

      {capabilities?.status === "ready" && (
        <CapabilityDetails capabilities={capabilities} />
      )}
    </div>
  );
}

// Friendly label + unit for each reported limit. `bytes: true` means the raw
// value is a byte count and should be shown in IEC units.
const LIMIT_META: Record<string, { label: string; bytes?: boolean }> = {
  maxBufferSize: { label: "Max buffer size", bytes: true },
  maxStorageBufferBindingSize: { label: "Max storage binding", bytes: true },
  maxComputeWorkgroupStorageSize: { label: "Workgroup storage", bytes: true },
  maxComputeInvocationsPerWorkgroup: { label: "Invocations / workgroup" },
  maxComputeWorkgroupsPerDimension: { label: "Workgroups / dimension" },
};

// Features worth calling out — they change what kernels can do.
const NOTABLE_FEATURES = new Set(["shader-f16", "timestamp-query"]);

function formatUnknown(value: string | undefined): string {
  if (!value || value === "unknown") return "—";
  return titleCase(value);
}

function CapabilityDetails({
  capabilities,
}: {
  capabilities: WebGPUCapabilities;
}) {
  const { adapter, features, limits, isFallbackAdapter } = capabilities;
  const vendor = formatUnknown(adapter?.vendor);
  const architecture = formatUnknown(adapter?.architecture);

  return (
    <div className="space-y-4 text-sm">
      <div>
        <p className="mb-1.5 font-medium">Adapter</p>
        <dl className="grid grid-cols-[10rem_1fr] gap-y-1">
          <dt className="text-muted-foreground">Vendor</dt>
          <dd>{vendor}</dd>
          <dt className="text-muted-foreground">Architecture</dt>
          <dd>{architecture}</dd>
          <dt className="text-muted-foreground">Type</dt>
          <dd>
            {isFallbackAdapter ? (
              <span className="text-amber-600 dark:text-amber-400">
                Software (fallback)
              </span>
            ) : (
              "Hardware"
            )}
          </dd>
        </dl>
      </div>

      <div>
        <p className="mb-1.5 font-medium">Limits</p>
        <dl className="grid grid-cols-[10rem_1fr] gap-y-1">
          {Object.entries(limits).map(([key, value]) => {
            const meta = LIMIT_META[key];
            return (
              <div key={key} className="contents">
                <dt className="text-muted-foreground">{meta?.label ?? key}</dt>
                <dd className="font-mono text-xs" title={value.toLocaleString()}>
                  {meta?.bytes ? formatBytes(value) : value.toLocaleString()}
                </dd>
              </div>
            );
          })}
        </dl>
      </div>

      {features.length > 0 && (
        <div>
          <p className="mb-1.5 font-medium">
            Features{" "}
            <span className="font-normal text-muted-foreground">
              ({features.length})
            </span>
          </p>
          <div className="flex flex-wrap gap-1">
            {features.map((f) => (
              <span
                key={f}
                className={`rounded px-1.5 py-0.5 font-mono text-xs ${
                  NOTABLE_FEATURES.has(f)
                    ? "bg-primary/15 text-primary"
                    : "bg-muted"
                }`}
              >
                {f}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/** Dashboard card summarising what WebGPU this browser exposes. */
export function GpuCapabilitiesCard() {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Cpu className="size-4" /> GPU Capabilities
        </CardTitle>
        <CardDescription>What this browser exposes.</CardDescription>
      </CardHeader>
      <CardContent>
        <CapabilityPanel />
      </CardContent>
    </Card>
  );
}
