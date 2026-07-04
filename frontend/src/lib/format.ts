// Small display formatters shared across the UI.

/**
 * Format a byte count with binary (IEC) units — B, KiB, MiB, GiB, TiB.
 * GPU limits are reported in bytes and are far more legible as "2 GiB" than
 * "2,147,483,644". Non-finite or negative input is returned as-is (stringified).
 */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return String(bytes);

  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit++;
  }

  // Whole bytes stay integers; larger units get up to 2 decimals, trimmed.
  const text =
    unit === 0
      ? String(value)
      : value
          .toFixed(2)
          .replace(/\.00$/, "")
          .replace(/(\.\d)0$/, "$1");
  return `${text} ${units[unit]}`;
}

/** Capitalise the first letter of each word (e.g. "turing" → "Turing"). */
export function titleCase(value: string): string {
  return value.replace(/\b\w/g, (c) => c.toUpperCase());
}
