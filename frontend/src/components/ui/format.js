export function formatBytes(n = 0) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 ** 2).toFixed(2)} MB`;
}

export function downloadJson(obj, name) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(obj, null, 2)], { type: 'application/json' }));
  const a = document.createElement('a');
  a.href = url; a.download = name; a.click();
  URL.revokeObjectURL(url);
}

/** A value from the API, or "—" when it is missing. Never a made-up 0. */
export function orDash(value, format) {
  if (value == null || Number.isNaN(value)) return '—';
  return format ? format(value) : value;
}

/** A 0–1 share as a whole percent, or "—" when it is missing. */
export function percent(share) {
  return orDash(share, (s) => `${Math.round(s * 100)}%`);
}
