import { useEffect, useState } from "react";

import { detectWebGPU } from "@/webgpu/capabilities";
import type { WebGPUCapabilities } from "@/webgpu/types";

interface UseWebGPUResult {
  capabilities: WebGPUCapabilities | null;
  loading: boolean;
  supported: boolean;
}

/**
 * Probe WebGPU capabilities once on mount. `supported` is true only once the
 * adapter has been acquired successfully (`status === "ready"`).
 */
export function useWebGPU(): UseWebGPUResult {
  const [capabilities, setCapabilities] = useState<WebGPUCapabilities | null>(
    null,
  );
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    detectWebGPU()
      .then((caps) => {
        if (active) setCapabilities(caps);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  return {
    capabilities,
    loading,
    supported: capabilities?.status === "ready",
  };
}
