import { DATA_VERDICTS, MODEL_VERDICTS } from './verdicts';

/**
 * Scrolls sideways inside its own box on narrow screens instead of widening the page.
 * Inside a grid or flex item, give that item min-w-0, or the item grows to the table's minWidth.
 */
export function Table({ minWidth = 560, className = '', children }) {
  return (
    <div className={`relative overflow-x-auto ${className}`}>
      <table className="data-table" style={{ minWidth }}>{children}</table>
    </div>
  );
}

const TONES = {
  safe: 'var(--status-safe)', info: 'var(--status-info)', warn: 'var(--status-warn)',
  critical: 'var(--status-critical)', flag: 'var(--flag)', neutral: 'var(--text-muted)', accent: 'var(--uv)',
};

/** A small status pill. Pass a `tone` from the palette, or a `color` for verdict colours. */
export function Badge({ tone = 'neutral', color, dot = true, className = '', children, ...props }) {
  const c = color || TONES[tone];
  return (
    <span
      {...props}
      className={`inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full px-2.5 py-0.5 text-[12px] font-medium leading-5 ${className}`}
      style={{ color: c, background: `color-mix(in srgb, ${c} 11%, transparent)`, boxShadow: `inset 0 0 0 1px color-mix(in srgb, ${c} 32%, transparent)` }}
    >
      {dot && <span className="h-1.5 w-1.5 rounded-full" style={{ background: c }} aria-hidden />}
      {children}
    </span>
  );
}

/** The backend's verdict code as a badge. Unknown codes show as-is in a neutral badge, never as "clean". */
export function VerdictBadge({ verdict, kind = 'data', short = true }) {
  const map = kind === 'model' ? MODEL_VERDICTS : DATA_VERDICTS;
  const v = map[verdict];
  if (!v) return <Badge tone="neutral">{verdict ? String(verdict).replace(/_/g, ' ').toLowerCase() : 'no verdict'}</Badge>;
  return <Badge color={v.color} title={v.label}>{short ? v.short || v.label : v.label}</Badge>;
}

/** One measured value with its label. Shows "—" when the value is missing. */
export function Readout({ label, value, unit, color, note }) {
  return (
    <div className="min-w-0">
      <div className="rule-label">{label}</div>
      <div className="mt-1 font-mono text-[28px] font-bold leading-none tabular" style={{ color: color || 'var(--text-primary)' }}>
        {value ?? '—'}{unit && <span className="ml-1 text-sm font-normal text-textMuted">{unit}</span>}
      </div>
      {note && <div className="mt-1.5 text-xs text-textMuted">{note}</div>}
    </div>
  );
}
