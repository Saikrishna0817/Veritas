/* eslint-disable react-refresh/only-export-components */
import { createElement, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Loader2, UploadCloud, FileCheck2, FlaskConical } from 'lucide-react';

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
      className={`inline-flex items-center gap-1.5 rounded-lg border border-statusWarn/50 px-2 py-1 font-mono text-[10px] font-bold uppercase tracking-wider text-statusWarn align-middle ${className}`}
      style={{ fontStretch: 'normal', letterSpacing: '0.08em' }}
    >
      <FlaskConical className="w-3 h-3" aria-hidden /> Research preview
    </span>
  );
}

export function Button({ children, variant = 'primary', loading, className = '', ...props }) {
  const styles = {
    primary: 'bg-white text-ink-900 hover:bg-[#EDC5FC]',
    uv: 'bg-white text-ink-900 hover:bg-[#EDC5FC] shadow-uv',
    ghost: 'bg-white/[0.06] text-textPrimary border border-transparent hover:border-white/40',
    danger: 'bg-transparent text-statusCritical border border-statusCritical/50 hover:bg-statusCritical/10',
  };
  return (
    <button
      {...props}
      disabled={props.disabled || loading}
      className={`group relative inline-flex items-center justify-center gap-2 rounded-2xl px-6 py-3.5 text-sm font-medium transition-[background,transform,box-shadow,filter] duration-200 active:translate-y-px disabled:opacity-45 disabled:cursor-not-allowed ${styles[variant]} ${className}`}
    >
      {loading && <Loader2 className="w-4 h-4 animate-spin" aria-hidden />}
      {children}
    </button>
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
      <input ref={ref} type="file" accept={accept} className="sr-only" tabIndex={-1} onChange={(e) => onFile(e.target.files[0])} />
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

export function ErrorNote({ children }) {
  if (!children) return null;
  return (
    <div role="alert" className="rounded-2xl border border-statusCritical/40 bg-statusCritical/[0.08] px-4 py-3 text-sm text-[#FFB4B5]">
      {children}
    </div>
  );
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
      initial={{ opacity: 0, y: 16, filter: 'blur(6px)' }}
      whileInView={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
      viewport={{ once: true, margin: '-40px' }}
      transition={{ duration: 0.7, delay, ease: [0.16, 1, 0.3, 1] }}
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
