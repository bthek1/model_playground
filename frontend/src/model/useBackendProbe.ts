// What backend *would* a model load on, asked before anything downloads.
//
// `pickBackend()` already answers this, but until now it was only ever called
// inside a worker at load time — which is too late for the one thing the SELECT
// slot needs it for: a catalogue entry that declares `backends: ["webgpu"]`
// (Florence-2, RT-DETR r50) is unusable on a CPU-only machine, and offering it
// anyway turns a known limitation into a failed download.
//
// One probe per component, resolved once. It is a `requestAdapter()` call, not a
// download, so it is cheap — but it is async, and `null` means "not answered
// yet", never "no GPU". A picker that treats the undecided state as WASM would
// grey out the WebGPU models for a frame on every page load.

import { useEffect, useState } from "react";

import { pickBackend, type Backend } from "./backend";

/** The backend a load would resolve to, or `null` until the probe answers. */
export function useBackendProbe(): Backend | null {
  const [backend, setBackend] = useState<Backend | null>(null);

  useEffect(() => {
    let live = true;
    void pickBackend().then((next) => {
      if (live) setBackend(next);
    });
    return () => {
      live = false;
    };
  }, []);

  return backend;
}
