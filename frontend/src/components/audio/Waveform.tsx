// Audio waveform canvases for the audio routes, per the model-visualization
// standard (docs/standards/model-visualization.md): drawn from the real
// Float32Array (never a stock image) and theme-aware by construction — the bars
// are painted in the canvas's own resolved `color`, so a Tailwind text class
// (default `text-primary`) themes them in both light and dark.
//
// Two variants share the drawing grammar:
//   <Waveform>     — a static clip (recorded take or uploaded file)
//   <LiveWaveform> — a scrolling mic waveform while recording, fed by an
//                    AnalyserNode on the live MediaStream

import { useEffect, useRef } from "react";

import { computePeaks } from "@/audio/waveform";

const HEIGHT = 96; // CSS pixels

/** Paint interleaved [min, max] peak pairs as symmetric bars around a midline. */
function paintPeaks(
  canvas: HTMLCanvasElement,
  peaks: Float32Array,
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return; // e.g. happy-dom in tests

  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 300;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(HEIGHT * dpr);
  ctx.scale(dpr, dpr);

  const color = getComputedStyle(canvas).color;
  const mid = HEIGHT / 2;
  const amp = mid - 2;

  ctx.clearRect(0, 0, width, HEIGHT);

  // Faint midline so silence still reads as "empty track", not a blank card.
  ctx.globalAlpha = 0.25;
  ctx.fillStyle = color;
  ctx.fillRect(0, mid - 0.5, width, 1);

  ctx.globalAlpha = 0.9;
  const buckets = peaks.length / 2;
  for (let b = 0; b < buckets; b++) {
    const min = peaks[b * 2];
    const max = peaks[b * 2 + 1];
    const x = (b / buckets) * width;
    const top = mid - max * amp;
    const bottom = mid - min * amp;
    // ≥1px tall so quiet-but-present audio stays visible.
    ctx.fillRect(x, top, Math.max(1, width / buckets - 0.5), Math.max(1, bottom - top));
  }
  ctx.globalAlpha = 1;
}

/**
 * Static waveform of a mono `Float32Array` clip. One [min, max] bucket per
 * ~2 CSS px, redrawn on resize. Color rides on `currentColor` — override with
 * any text-color class.
 */
export function Waveform({
  samples,
  className,
}: {
  samples: Float32Array;
  className?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const draw = () =>
      paintPeaks(canvas, computePeaks(samples, Math.max(1, Math.floor((canvas.clientWidth || 300) / 2))));
    draw();

    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(draw);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [samples]);

  return (
    <canvas
      ref={canvasRef}
      role="img"
      aria-label="Audio waveform"
      style={{ height: HEIGHT }}
      className={`w-full rounded-md border bg-muted/30 text-primary ${className ?? ""}`}
    />
  );
}

/** How many rolling peak buckets the live view keeps (~20 s at 60 fps ÷ 4). */
const LIVE_BUCKETS = 300;

/**
 * Scrolling live waveform of a `MediaStream` (the mic, while recording). An
 * `AnalyserNode` samples the time-domain signal each animation frame; each
 * frame's [min, max] is pushed into a rolling ring drawn right-to-left, so the
 * newest audio is at the right edge. The tap is analysis-only — nothing is
 * connected to the speakers, so there's no feedback.
 */
export function LiveWaveform({
  stream,
  className,
}: {
  stream: MediaStream;
  className?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || typeof AudioContext === "undefined") return;

    let ctx: AudioContext | null = null;
    let analyser: AnalyserNode;
    let source: MediaStreamAudioSourceNode;
    try {
      ctx = new AudioContext();
      analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      source = ctx.createMediaStreamSource(stream);
      source.connect(analyser);
    } catch {
      void ctx?.close();
      return; // stream already ended / env without real Web Audio
    }
    const audioCtx = ctx;

    const data = new Float32Array(analyser.fftSize);
    const ring = new Float32Array(LIVE_BUCKETS * 2); // interleaved [min, max]
    let head = 0;
    let raf = 0;

    const tick = () => {
      analyser.getFloatTimeDomainData(data);
      let min = 0;
      let max = 0;
      for (let i = 0; i < data.length; i++) {
        const v = data[i];
        if (v < min) min = v;
        if (v > max) max = v;
      }
      ring[head * 2] = min;
      ring[head * 2 + 1] = max;
      head = (head + 1) % LIVE_BUCKETS;

      // Unroll the ring so oldest→newest reads left→right.
      const ordered = new Float32Array(LIVE_BUCKETS * 2);
      for (let b = 0; b < LIVE_BUCKETS; b++) {
        const src = ((head + b) % LIVE_BUCKETS) * 2;
        ordered[b * 2] = ring[src];
        ordered[b * 2 + 1] = ring[src + 1];
      }
      paintPeaks(canvas, ordered);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
      source.disconnect();
      void audioCtx.close();
    };
  }, [stream]);

  return (
    <canvas
      ref={canvasRef}
      role="img"
      aria-label="Live microphone waveform"
      style={{ height: HEIGHT }}
      className={`w-full rounded-md border bg-muted/30 text-destructive ${className ?? ""}`}
    />
  );
}
