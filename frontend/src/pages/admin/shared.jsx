import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Skeleton } from '../../components/Skeleton';

// Shared pieces of the admin console: data loading, tabs, an inline confirm step, load states and notices.
/* eslint-disable react-refresh/only-export-components */

/** Loads one API resource and keeps loading / error / data apart, so a failed load never looks like an empty list. */
export function useResource(fetcher) {
  const [state, setState] = useState({ data: undefined, error: null, loading: true });
  const settle = useCallback((promise) => promise
    .then((data) => setState({ data, error: null, loading: false }))
    .catch((error) => setState((s) => ({ data: s.data, error, loading: false }))), []);
  const reload = useCallback(() => {
    setState((s) => ({ ...s, loading: true, error: null }));
    return settle(fetcher());
  }, [fetcher, settle]);
  useEffect(() => { settle(fetcher()); }, [fetcher, settle]);   // first load: the initial state already says "loading"
  return { ...state, reload };
}

/**
 * A resource whose list the API pages with a cursor. page(before) resolves to { items, next }.
 * data is every page loaded so far; loadMore() fetches the next one; reload() starts again from the newest.
 */
export function usePagedResource(page) {
  const [more, setMore] = useState({ items: [], next: null, busy: false, error: null });
  const first = useCallback(() => page().then((r) => { setMore({ items: [], next: r.next, busy: false, error: null }); return r.items; }), [page]);
  const res = useResource(first);
  const loadMore = async () => {
    setMore((m) => ({ ...m, busy: true, error: null }));
    try {
      const r = await page(more.next);
      setMore((m) => ({ items: [...m.items, ...r.items], next: r.next, busy: false, error: null }));
    } catch (error) {
      setMore((m) => ({ ...m, busy: false, error }));
    }
  };
  const data = useMemo(() => (res.data ? [...res.data, ...more.items] : res.data), [res.data, more.items]);
  return { ...res, data, next: more.next, loadingMore: more.busy, moreError: more.error, loadMore };
}

/** "Load more" under a paged list; renders nothing on the last page. */
export function LoadMore({ resource, what }) {
  const { next, loadingMore, moreError, loadMore } = resource;
  if (!next) return null;
  return (
    <div className="in-actions">
      <button type="button" className="in-btn in-btn-quiet" onClick={loadMore} disabled={loadingMore}>{loadingMore ? 'Loading…' : `Load more ${what}`}</button>
      {moreError && <span className="in-error" role="alert">More {what} could not be loaded: {moreError.message}</span>}
    </div>
  );
}

export const when = (iso) => (iso ? new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }) : '—');
export const shortId = (id) => (id ? `${String(id).slice(0, 8)}…` : '—');

/** Tabs with arrow-key, Home and End navigation. tabs: [{ id, label, count? }] */
export function Tabs({ tabs, value, onChange, label }) {
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
    <div className="in-tabs" role="tablist" aria-label={label} onKeyDown={onKeyDown}>
      {tabs.map((t) => (
        <button
          key={t.id}
          ref={(el) => { refs.current[t.id] = el; }}
          type="button"
          role="tab"
          id={`admin-tab-${t.id}`}
          aria-selected={value === t.id}
          aria-controls={`admin-panel-${t.id}`}
          tabIndex={value === t.id ? 0 : -1}
          className="in-tab"
          onClick={() => onChange(t.id)}
        >
          {t.label}
          {t.count != null && <span className="in-tab-count in-num"><span className="in-sr"> (</span>{t.count}<span className="in-sr">)</span></span>}
        </button>
      ))}
    </div>
  );
}

export function TabPanel({ id, children }) {
  return <div role="tabpanel" id={`admin-panel-${id}`} aria-labelledby={`admin-tab-${id}`} tabIndex={0} className="in-tabpanel">{children}</div>;
}

/**
 * A two-step action: the first button opens a sentence that says exactly what will happen, the second does it.
 * Focus moves to the confirming button, and Escape or Cancel closes the step.
 */
export function ConfirmAction({ label, confirmLabel, message, onConfirm, disabled, danger = false, quiet = true, disabledReason }) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const confirmRef = useRef(null);
  useEffect(() => { if (open) confirmRef.current?.focus(); }, [open]);
  const run = async () => {
    setBusy(true);
    try { await onConfirm(); } finally { setBusy(false); setOpen(false); }
  };
  if (!open) {
    return (
      <button type="button" className={`in-btn${quiet ? ' in-btn-quiet' : ''} in-btn-sm`} disabled={disabled} title={disabled ? disabledReason : undefined} onClick={() => setOpen(true)}>
        {label}
      </button>
    );
  }
  return (
    <div className="in-confirm" role="group" aria-label={label} onKeyDown={(e) => e.key === 'Escape' && setOpen(false)}>
      <p>{message}</p>
      <div className="in-actions">
        <button ref={confirmRef} type="button" className={`in-btn in-btn-sm${danger ? ' in-btn-danger' : ''}`} disabled={busy} onClick={run}>
          {busy ? 'Working…' : confirmLabel}
        </button>
        <button type="button" className="in-btn in-btn-quiet in-btn-sm" disabled={busy} onClick={() => setOpen(false)}>Cancel</button>
      </div>
    </div>
  );
}

/** Loading, failed and empty states for one resource. Renders children only when there is data to show. */
export function LoadState({ resource, what, empty, isEmpty, children }) {
  const { data, error, loading, reload } = resource;
  if (data === undefined && loading) return <Skeleton label={`Loading ${what}…`} lines={0} rows={4} />;
  if (data === undefined && error) {
    return (
      <div className="in-state" role="alert">
        <p><b>{what.charAt(0).toUpperCase() + what.slice(1)} could not be loaded.</b> {error.message}</p>
        <button type="button" className="in-btn in-btn-quiet in-btn-sm" onClick={reload}>Try again</button>
      </div>
    );
  }
  return (
    <>
      {error && <p className="in-error" role="alert">Refreshing {what} failed: {error.message} The list below may be out of date.</p>}
      {isEmpty(data) ? <p className="in-empty">{empty}</p> : children}
    </>
  );
}

/** Outcome of the last action: role=status for success, role=alert for a failure. */
export function Notice({ notice, onClose }) {
  if (!notice) return null;
  return (
    <div className={`in-notice${notice.error ? ' in-notice-error' : ''}`} role={notice.error ? 'alert' : 'status'}>
      <span>{notice.text}</span>
      <button type="button" className="in-btn in-btn-quiet in-btn-sm" onClick={onClose}>Dismiss</button>
    </div>
  );
}
