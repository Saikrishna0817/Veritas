import { createElement } from 'react';
import { CircleAlert, CloudOff, FlaskConical, Hourglass, Lock, RotateCw, SearchX, ServerOff, X } from 'lucide-react';
import { describeError } from '../../services/errors';
import { Button } from './actions';

/** Marks a heuristic feature: output that is not measured on the benchmark. */
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

/** A failed action, inline. For a failed page load with nothing to show, use ErrorState. */
export function ErrorNote({ children, className }) {
  return <Notice tone="critical" className={className}>{children}</Notice>;
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

// Empty, error and unavailable share one card so a page's "nothing to show" moments look like one family.
function StateCard({ icon, iconClass = 'text-textSecondary', title, children, action, role, className = '' }) {
  return (
    <div role={role} className={`panel flex flex-col items-start gap-4 p-8 ${className}`}>
      {icon && <span className="icon-chip h-11 w-11 shrink-0">{createElement(icon, { className: `h-4 w-4 ${iconClass}`, strokeWidth: 1.5, 'aria-hidden': true })}</span>}
      <div>
        <div className="font-display text-xl font-bold">{title}</div>
        {children && <p className="mt-1.5 max-w-[60ch] text-textSecondary">{children}</p>}
      </div>
      {action && <div className="flex flex-wrap items-center gap-3">{action}</div>}
    </div>
  );
}

/** Nothing to show yet. Say why, and give the one action that fixes it. */
export function Empty({ title, children, action, icon, className }) {
  return <StateCard icon={icon} title={title} action={action} className={className}>{children}</StateCard>;
}

/**
 * The server cannot do this here: the feature is switched off on this deployment (501/503) or the
 * server is not answering. Not the person's fault and not a bug, so it is quiet, not red.
 */
export function Unavailable({ title = 'Not available on this server', children, action, icon = ServerOff, className }) {
  return <StateCard role="status" icon={icon} title={title} action={action} className={className}>{children}</StateCard>;
}

const ERROR_LOOK = {
  offline: { icon: CloudOff, iconClass: 'text-textSecondary' },
  timeout: { icon: Hourglass, iconClass: 'text-textSecondary' },
  unavailable: { icon: ServerOff, iconClass: 'text-textSecondary' },
  rate_limited: { icon: Hourglass, iconClass: 'text-statusWarn' },
  forbidden: { icon: Lock, iconClass: 'text-statusWarn' },
  not_found: { icon: SearchX, iconClass: 'text-textSecondary' },
  failed: { icon: CircleAlert, iconClass: 'text-statusCritical' },
};

/**
 * A page or section that failed to load and has nothing else to show. Pass the thrown error from
 * services/api; the wording follows its status, and the server's own message is kept when it has one.
 * `onRetry` adds a Try again button.
 */
export function ErrorState({ error, onRetry, title, children, className }) {
  const d = describeError(error);
  if (d.kind === 'unavailable' || d.kind === 'offline' || d.kind === 'timeout') {
    return (
      <Unavailable title={title || d.title} icon={ERROR_LOOK[d.kind].icon} className={className}
        action={onRetry && <Button variant="ghost" size="sm" onClick={onRetry}><RotateCw className="h-4 w-4" aria-hidden /> Try again</Button>}>
        {children || d.message}
      </Unavailable>
    );
  }
  return (
    <StateCard role="alert" {...ERROR_LOOK[d.kind]} title={title || d.title} className={className}
      action={onRetry && d.kind !== 'forbidden' && <Button variant="ghost" size="sm" onClick={onRetry}><RotateCw className="h-4 w-4" aria-hidden /> Try again</Button>}>
      {children || d.message}
    </StateCard>
  );
}
