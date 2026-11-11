/* eslint-disable react-refresh/only-export-components */
import { createElement, useEffect, useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { motion } from 'framer-motion';
import { Loader2, UploadCloud, FileCheck2, FlaskConical, Trash2, X } from 'lucide-react';

export const DATA_VERDICTS = {
  CONFIRMED_POISONED: { label: 'Confirmed poisoned', short: 'Confirmed', color: 'var(--status-critical)', p: 'p ≤ 1e-4', blurb: 'The incoming rows differ from the trusted reference far beyond what clean batches do. Review the flagged rows before training.' },
  SUSPICIOUS: { label: 'Suspicious', short: 'Suspicious', color: 'var(--status-warn)', p: 'p ≤ 0.01', blurb: 'Stronger evidence than 99% of clean batches. Review the ranked rows first.' },
  LOW_RISK: { label: 'Low risk', short: 'Low risk', color: 'var(--status-info)', p: 'p ≤ 0.05', blurb: 'Weak evidence; about 7% of clean batches reach this level. Spot-check the top rows.' },
  CLEAN: { label: 'Clean', short: 'Clean', color: 'var(--status-safe)', p: 'p > 0.05', blurb: 'Nothing here beyond what clean batches show. Small contamination (~1%) can still hide below this.' },
};

export const MODEL_VERDICTS = {
  DANGEROUS: { label: 'Dangerous', color: 'var(--status-critical)', blurb: 'Loading this file can run code or reach the system. Do not load it.' },
  REVIEW: { label: 'Needs review', color: 'var(--status-warn)', blurb: 'Something could not be verified. A person should look before this file is loaded.' },
  NO_CODE_EXECUTION_FOUND: { label: 'No code execution found', color: 'var(--status-info)', blurb: 'Every import is a known data container. That is not a guarantee of safety.' },
  SAFE_FORMAT: { label: 'Safe format', color: 'var(--status-safe)', blurb: 'This format stores weights only and cannot contain code.' },
};

export function PageHeader({ title, accent, lede, children, preview }) {
  return (
    <header className="flex flex-col gap-5 md:flex-row md:items-end md:justify-between pb-8 mb-10 border-b border-borderHairline">
      <div className="max-w-3xl">
        <h1 className="font-display font-medium text-[clamp(2.6rem,5.6vw,5rem)] leading-[1] text-textPrimary">
          {title}{accent && <> <span className="serif-accent">{accent}</span></>}
          {preview && <PreviewTag className="ml-4 align-middle" />}
        </h1>
        {lede && <p className="mt-4 text-textSecondary text-[17px] leading-relaxed max-w-[62ch]">{lede}</p>}
      </div>
      {children && <div className="flex flex-wrap gap-3">{children}</div>}
    </header>
  );
}

export function PreviewTag({ className = '' }) {
  return (
    <span
      title="Heuristic output, not measured on the benchmark"
      className={`inline-flex items-center gap-1.5 rounded-full border border-statusWarn/50 px-3 py-1 text-[12px] font-medium text-statusWarn align-middle ${className}`}
    >
      <FlaskConical className="w-3 h-3" aria-hidden /> Research preview
    </span>
  );
}

const BUTTON_VARIANTS = {
  primary: 'bg-white text-ink-900 hover:bg-[#EDC5FC]',
  uv: 'bg-white text-ink-900 hover:bg-[#EDC5FC] shadow-uv',
  ghost: 'bg-white/[0.06] text-textPrimary border border-transparent hover:border-white/40',
  danger: 'bg-transparent text-statusCritical border border-statusCritical/50 hover:bg-statusCritical/10',
};
const BUTTON_SIZES = { md: 'px-6 py-3.5', sm: 'px-4 py-2' };

/** primary (white) for the one main action, ghost for secondary, danger for irreversible ones. */
export function Button({ children, variant = 'primary', size = 'md', loading, className = '', type = 'button', ...props }) {
  return (
    <button
      type={type}
      {...props}
      disabled={props.disabled || loading}
      aria-busy={loading || undefined}
      className={`group relative inline-flex items-center justify-center gap-2 rounded-2xl text-sm font-medium transition-[background,border-color,transform,box-shadow] duration-200 active:translate-y-px disabled:opacity-45 disabled:cursor-not-allowed disabled:active:translate-y-0 ${BUTTON_SIZES[size]} ${BUTTON_VARIANTS[variant]} ${className}`}
    >
      {loading && <Loader2 className="w-4 h-4 animate-spin" aria-hidden />}
      {children}
    </button>
  );
}

/** A square icon-only button. `label` is required: it is the button's accessible name and its tooltip. */
export function IconButton({ icon, label, tone = 'default', className = '', ...props }) {
  const tones = {
    default: 'text-textMuted hover:bg-white/[0.06] hover:text-textPrimary',
    danger: 'text-textMuted hover:bg-statusCritical/10 hover:text-statusCritical',
  };
  return (
    <button type="button" aria-label={label} title={label} {...props}
      className={`inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl transition-colors disabled:cursor-not-allowed disabled:opacity-35 ${tones[tone]} ${className}`}>
      {createElement(icon, { className: 'h-4 w-4', 'aria-hidden': true })}
    </button>
  );
}

/** The standard page container under the app shell. */
export function Page({ children, className = '' }) {
  return <div className={`mx-auto max-w-[1360px] px-5 py-10 md:px-10 md:py-14 ${className}`}>{children}</div>;
}

/**
 * A labelled form control. `as` picks input, select or textarea; `mono` is for data values (column names,
 * dates, keys). `hint` and `error` are announced with the control.
 */
export function Field({ label, hint, error, as = 'input', mono, hideLabel, className = '', children, id: givenId, ...props }) {
  const autoId = useId();
  const id = givenId || autoId;
  const describedBy = [hint && `${id}-hint`, error && `${id}-error`].filter(Boolean).join(' ') || undefined;
  return (
    <div className={className}>
      <label htmlFor={id} className={hideLabel ? 'sr-only' : 'rule-label'}>{label}</label>
      {createElement(as, {
        id,
        ...props,
        'aria-invalid': error ? true : undefined,
        'aria-describedby': describedBy,
        className: `field-input ${hideLabel ? '' : 'mt-1.5'} ${mono ? 'font-mono text-sm' : ''}`,
      }, children)}
      {hint && !error && <p id={`${id}-hint`} className="mt-1.5 text-xs text-textMuted">{hint}</p>}
      {error && <p id={`${id}-error`} className="mt-1.5 text-xs text-[#FFB4B5]">{error}</p>}
    </div>
  );
}

/** A radio group drawn as cards, for choices that need a sentence each. options: [{ id, label, hint }]. */
export function RadioCards({ name, label, options, value, onChange, stacked, hideLabel, className = '' }) {
  return (
    <fieldset className={className}>
      <legend className={hideLabel ? 'sr-only' : 'rule-label'}>{label}</legend>
      <div className={`grid gap-2 ${hideLabel ? '' : 'mt-2'} ${stacked ? '' : '[grid-template-columns:repeat(auto-fit,minmax(min(100%,200px),1fr))]'}`}>
        {options.map((o) => (
          <label key={o.id} className={`flex cursor-pointer gap-3 rounded-2xl border p-3.5 transition-colors has-[:focus-visible]:outline has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-uv ${value === o.id ? 'border-violet bg-violet/[0.08]' : 'border-borderHairline hover:border-borderStrong hover:bg-white/[0.02]'}`}>
            <input type="radio" name={name} value={o.id} checked={value === o.id} onChange={() => onChange(o.id)} className="mt-1 accent-[#B9A3FF] focus-visible:outline-none" />
            <span>
              <span className="block text-sm font-medium">{o.label}</span>
              {o.hint && <span className="block text-xs text-textMuted">{o.hint}</span>}
            </span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}

/**
 * A row of toggle buttons for filters and pickers (aria-pressed). Use Tabs instead when the choice
 * swaps a whole panel. options: [{ id, label, icon?, count? }].
 */
export function Segmented({ options, value, onChange, label, className = '' }) {
  const layoutId = useId();
  return (
    <div role="group" aria-label={label} className={`inline-flex max-w-full flex-wrap gap-1 rounded-[20px] bg-white/[0.04] p-1 ring-1 ring-inset ring-white/[0.06] ${className}`}>
      {options.map((o) => (
        <SegmentButton key={String(o.id)} option={o} active={value === o.id} layoutId={layoutId} onClick={() => onChange(o.id)} aria-pressed={value === o.id} />
      ))}
    </div>
  );
}

function SegmentButton({ option, active, layoutId, ...props }) {
  return (
    <button
      type="button"
      {...props}
      className={`relative inline-flex min-h-[40px] items-center gap-2 rounded-2xl px-4 py-2 text-sm font-medium transition-colors duration-200 ${active ? 'text-ink-900' : 'text-textSecondary hover:bg-white/[0.06] hover:text-textPrimary'}`}
    >
      {active && <motion.span layoutId={layoutId} className="absolute inset-0 rounded-2xl bg-white" transition={{ type: 'spring', stiffness: 520, damping: 42 }} aria-hidden />}
      <span className="relative inline-flex items-center gap-2">
        {option.icon && createElement(option.icon, { className: 'h-4 w-4', 'aria-hidden': true })}
        {option.label}
        {option.count != null && <span className={`tabular text-xs ${active ? 'text-ink-900/60' : 'text-textMuted'}`}>{option.count}</span>}
      </span>
    </button>
  );
}

/**
 * Tabs that swap a panel: arrow keys, Home and End move between them. Render each panel with
 * <TabPanel idBase={…} value={…}> using the same idBase.
 */
export function Tabs({ tabs, value, onChange, label, idBase, className = '' }) {
  const layoutId = useId();
  const refs = useRef({});
  const onKeyDown = (e) => {
    const i = tabs.findIndex((t) => t.id === value);
    const next = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 }[e.key];
    if (next === undefined) return;
    e.preventDefault();
    const t = tabs[(next + tabs.length) % tabs.length];
    onChange(t.id);
    refs.current[t.id]?.focus();
  };
  return (
    <div role="tablist" aria-label={label} onKeyDown={onKeyDown} className={`inline-flex max-w-full flex-wrap gap-1 rounded-[20px] bg-white/[0.04] p-1 ring-1 ring-inset ring-white/[0.06] ${className}`}>
      {tabs.map((t) => (
        <SegmentButton
          key={t.id}
          ref={(el) => { refs.current[t.id] = el; }}
          option={t}
          active={value === t.id}
          layoutId={layoutId}
          role="tab"
          id={`${idBase}-tab-${t.id}`}
          aria-selected={value === t.id}
          aria-controls={`${idBase}-panel-${t.id}`}
          tabIndex={value === t.id ? 0 : -1}
          onClick={() => onChange(t.id)}
        />
      ))}
    </div>
  );
}

export function TabPanel({ idBase, value, children }) {
  return <div role="tabpanel" id={`${idBase}-panel-${value}`} aria-labelledby={`${idBase}-tab-${value}`} tabIndex={0} className="focus-visible:outline-offset-8">{children}</div>;
}

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

/** Inline feedback. critical = something failed, safe = something finished, info = worth knowing. */
export function Notice({ tone = 'info', onDismiss, className = '', children }) {
  if (!children) return null;
  const styles = {
    critical: 'border-statusCritical/40 bg-statusCritical/[0.08] text-[#FFB4B5]',
    safe: 'border-statusSafe/40 bg-statusSafe/[0.07] text-statusSafe',
    info: 'border-borderStrong bg-white/[0.04] text-textSecondary',
  };
  return (
    <div role={tone === 'critical' ? 'alert' : 'status'} className={`flex items-start justify-between gap-3 rounded-2xl border px-4 py-3 text-sm ${styles[tone]} ${className}`}>
      <div className="min-w-0">{children}</div>
      {onDismiss && (
        <button type="button" onClick={onDismiss} aria-label="Dismiss" className="-my-1.5 -mr-2 inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-xl opacity-80 hover:bg-white/[0.06] hover:opacity-100">
          <X className="h-4 w-4" aria-hidden />
        </button>
      )}
    </div>
  );
}

/** A quiet loading line that screen readers announce. */
export function Loading({ children = 'Loading…', className = '' }) {
  return (
    <p role="status" className={`flex items-center gap-2.5 text-sm text-textMuted ${className}`}>
      <span className="relative flex h-2 w-2" aria-hidden>
        <span className="absolute inset-0 animate-ping rounded-full bg-uv/60 motion-reduce:animate-none" />
        <span className="relative h-2 w-2 rounded-full bg-uv" />
      </span>
      {children}
    </p>
  );
}

/**
 * A modal for decisions that need the page held still: irreversible actions that also ask for input.
 * Built on <dialog>, so focus stays inside, Escape closes it and the page behind is inert.
 */
export function Dialog({ open, onClose, title, description, children }) {
  const ref = useRef(null);
  const titleId = useId();
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) {
      d.showModal();
      // showModal focuses the first button (Close); the first field is the useful place to start.
      d.querySelector('input:not([type=hidden]), select, textarea')?.focus();
    }
    if (!open && d.open) d.close();
  }, [open]);
  // Portalled to <body> so the parent's spacing utilities (space-y-*) cannot shift it off centre.
  return createPortal(
    <dialog ref={ref} className="veritas-dialog" aria-labelledby={titleId} onClose={onClose} onCancel={(e) => { e.preventDefault(); onClose?.(); }}
      onClick={(e) => { if (e.target === ref.current) onClose?.(); }}>
      {open && (
        <div className="edge-card quiet">
          <div className="edge-inner p-7 sm:p-8">
            <div className="flex items-start justify-between gap-4">
              <h2 id={titleId} className="font-display text-[26px] font-medium leading-tight">{title}</h2>
              <IconButton icon={X} label="Close" onClick={onClose} className="-mr-2 -mt-1" />
            </div>
            {description && <p className="mt-2 text-sm text-textSecondary">{description}</p>}
            <div className="mt-6">{children}</div>
          </div>
        </div>
      )}
    </dialog>,
    document.body,
  );
}

/**
 * Two-step destructive action without a modal: the first click asks, the second does it.
 * size="sm" is an icon-and-word control for table rows; "md" is a full danger button.
 */
export function ConfirmButton({ label, confirmLabel, onConfirm, disabled, title, icon = Trash2, size = 'md', cancelLabel = 'Cancel' }) {
  const [asking, setAsking] = useState(false);
  const [busy, setBusy] = useState(false);
  const confirm = async () => {
    setBusy(true);
    try { await onConfirm(); } finally { setBusy(false); setAsking(false); }
  };
  if (!asking) {
    return size === 'sm' ? (
      <button type="button" disabled={disabled} title={title} aria-label={title || label} onClick={() => setAsking(true)}
        className="inline-flex min-h-[36px] items-center gap-1.5 rounded-xl px-3 py-1.5 text-sm text-textMuted transition-colors hover:bg-statusCritical/10 hover:text-statusCritical disabled:cursor-not-allowed disabled:opacity-30">
        {createElement(icon, { className: 'h-4 w-4', 'aria-hidden': true })} <span className="sr-only sm:not-sr-only">{label}</span>
      </button>
    ) : (
      <Button variant="danger" disabled={disabled} title={title} onClick={() => setAsking(true)}>{createElement(icon, { className: 'h-4 w-4', 'aria-hidden': true })} {label}</Button>
    );
  }
  return (
    <span className="inline-flex flex-wrap items-center gap-2">
      <Button variant="danger" size={size} loading={busy} onClick={confirm} autoFocus>{busy ? 'Working…' : confirmLabel}</Button>
      <Button variant="ghost" size={size} disabled={busy} onClick={() => setAsking(false)}>{cancelLabel}</Button>
    </span>
  );
}

export function Panel({ title, aside, children, className = '' }) {
  return (
    <section className={`panel p-6 ${className}`}>
      {(title || aside) && (
        <div className="flex items-baseline justify-between gap-4 mb-5">
          {title && <h2 className="font-display font-bold text-[17px] text-textPrimary">{title}</h2>}
          {aside}
        </div>
      )}
      {children}
    </section>
  );
}

/**
 * The reference's step card: #1A1A1A, 28px corners, a big display number. With an `icon` the number sits
 * bottom-right beside a round icon chip; without one it sits under the label. Only pass real values;
 * use "—" while loading.
 */
export function StatCard({ label, value, unit, note, icon, accent }) {
  const number = (
    <span className={`font-display font-medium leading-none tracking-[-0.05em] tabular ${icon ? 'text-right text-[clamp(2.4rem,3.6vw,3.4rem)]' : 'text-[clamp(2.2rem,3.2vw,3rem)]'}`}>
      {value}{unit && <span className="ml-1 text-[0.42em] tracking-normal text-textSecondary">{unit}</span>}
    </span>
  );
  return (
    <div className="flex h-full flex-col rounded-[28px] border border-white/[0.07] bg-[#1A1A1A] p-6">
      <div className="flex items-start justify-between gap-3">
        <span className="text-sm font-medium text-textSecondary">{label}</span>
        {accent && <span className="rounded-lg bg-uv px-2 py-0.5 text-[10px] font-medium text-ink-900">{accent}</span>}
      </div>
      {icon ? (
        <div className="mt-auto flex items-end justify-between gap-3 pt-8">
          <span className="icon-chip h-11 w-11 shrink-0">{createElement(icon, { className: 'h-4 w-4 text-white', strokeWidth: 1.5, 'aria-hidden': true })}</span>
          {number}
        </div>
      ) : <div className="mt-6">{number}</div>}
      {note && <div className={`mt-2 text-xs text-textMuted ${icon ? 'mt-3 text-right' : ''}`}>{note}</div>}
    </div>
  );
}

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

export function Dropzone({ accept, onFile, file, hint, label = 'Drop a file or browse', icon = UploadCloud }) {
  const ref = useRef();
  const [over, setOver] = useState(false);
  return (
    <div
      role="button"
      tabIndex={0}
      onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), ref.current?.click())}
      onClick={() => ref.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); onFile(e.dataTransfer.files[0]); }}
      className={`relative flex min-h-[240px] cursor-pointer flex-col items-center justify-center gap-3 rounded-[28px] border border-dashed p-10 text-center transition-colors duration-300
        ${over ? 'border-uv bg-uv/[0.06] scan-sweep' : file ? 'border-statusSafe/60 bg-statusSafe/[0.04]' : 'border-borderStrong bg-ink-950/50 hover:border-violet hover:bg-violet/[0.05]'}`}
    >
      <input ref={ref} type="file" accept={accept} aria-label="Choose a file" className="sr-only" tabIndex={-1} onChange={(e) => onFile(e.target.files[0])} />
      {/* slide corner marks */}
      {['top-2 left-2 border-l border-t', 'top-2 right-2 border-r border-t', 'bottom-2 left-2 border-l border-b', 'bottom-2 right-2 border-r border-b'].map((c) => (
        <span key={c} className={`absolute h-3 w-3 rounded-[2px] border-uv/70 ${c}`} aria-hidden />
      ))}
      {file ? <FileCheck2 className="h-10 w-10 text-statusSafe" aria-hidden /> : createElement(icon, { className: `h-10 w-10 ${over ? 'text-uv' : 'text-textMuted'}`, 'aria-hidden': true })}
      {file ? (
        <>
          <div className="font-mono text-base text-statusSafe break-all">{file.name}</div>
          <div className="text-xs text-textMuted">{formatBytes(file.size)} · click or drop to replace</div>
        </>
      ) : (
        <>
          <div className="font-display text-lg font-bold text-textPrimary">{label}</div>
          {hint && <div className="text-xs text-textMuted max-w-[46ch]">{hint}</div>}
        </>
      )}
    </div>
  );
}

export function ErrorNote({ children, className }) {
  return <Notice tone="critical" className={className}>{children}</Notice>;
}

export function Empty({ title, children, action }) {
  return (
    <div className="panel flex flex-col items-start gap-3 p-8">
      <div className="font-display text-xl font-bold">{title}</div>
      <p className="text-textSecondary max-w-[60ch]">{children}</p>
      {action}
    </div>
  );
}

export function Reveal({ children, delay = 0, className = '' }) {
  return (
    <motion.div
      className={className}
      // Starts readable: if the animation never runs (stalled script, throttled tab), the content is still there.
      initial={{ opacity: 0.55, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-40px' }}
      transition={{ duration: 0.8, delay, ease: [0.19, 1, 0.22, 1] }}
    >
      {children}
    </motion.div>
  );
}

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
