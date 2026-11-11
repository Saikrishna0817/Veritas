import { createElement, useState } from 'react';
import { Loader2, Trash2 } from 'lucide-react';

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
