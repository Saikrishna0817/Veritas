import { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Copy, Check, KeyRound, Trash2 } from 'lucide-react';
import { api } from '../services/api';
import { useAuth } from '../hooks/useAuth';
import { PageHeader, Panel, Button, ErrorNote, ErrorState, Notice, Field, ConfirmButton, Dialog, Loading, Page, EASE, DURATION } from '../components/ui';

const USAGE = [
  ['analyses', 'Analyses today', (n) => n.toLocaleString()],
  ['rows', 'Rows checked today', (n) => n.toLocaleString()],
  ['scan_mb', 'Model files scanned today', (n) => `${n.toLocaleString()} MB`],
];

function day(iso) {
  return iso ? new Date(iso).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' }) : 'never';
}

function Usage() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const load = () => api.getUsage().then((d) => { setData(d); setError(null); }).catch(setError);
  useEffect(() => { load(); }, []);
  return (
    <Panel title="Usage" aside={<span className="text-xs text-textMuted">Resets at 00:00 UTC</span>}>
      {error && <ErrorState error={error} onRetry={() => { setError(null); load(); }} title="Usage could not load" className="border-0 bg-transparent p-0" />}
      {!data && !error && <Loading>Loading usage…</Loading>}
      {data && (
        <>
          <div className="space-y-6">
            {USAGE.map(([key, label, fmt], i) => {
              const used = data.used?.[key] ?? 0;
              const limit = data.limits?.[key] ?? 0;
              const share = limit ? Math.min(used / limit, 1) : 0;
              return (
                <div key={key}>
                  <div className="flex items-baseline justify-between gap-3">
                    <span className="font-medium">{label}</span>
                    <span className="font-mono text-sm text-textSecondary">
                      {fmt(used)} <span className="text-textMuted">{limit ? `of ${fmt(limit)}` : '· no limit'}</span>
                    </span>
                  </div>
                  <div className="mt-2 h-3 overflow-hidden rounded-full bg-white/[0.07]">
                    <motion.div
                      className="h-full origin-left rounded-full"
                      style={{ width: `${share * 100}%`, background: share >= 0.9 ? 'var(--flag)' : 'var(--uv)' }}
                      initial={{ scaleX: 0 }}
                      animate={{ scaleX: 1 }}
                      transition={{ duration: DURATION.grow, delay: i * 0.1, ease: EASE }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
          <p className="mt-6 text-sm text-textMuted">
            {data.retention_days > 0
              ? `Results and scans are deleted automatically after ${data.retention_days} days.`
              : 'Results and scans are kept until you delete them.'}
          </p>
        </>
      )}
    </Panel>
  );
}

function ApiKeys() {
  const [keys, setKeys] = useState(null);
  const [name, setName] = useState('');
  const [fresh, setFresh] = useState(null);
  const [copied, setCopied] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const load = () => api.listApiKeys().then((d) => setKeys(d.keys || [])).catch((e) => setError(e.message));
  useEffect(() => { load(); }, []);

  const create = async (e) => {
    e.preventDefault();
    setBusy(true); setError(''); setCopied(false);
    try {
      setFresh(await api.createApiKey(name.trim()));
      setName('');
      load();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };
  const revoke = async (id) => {
    setError('');
    try {
      await api.revokeApiKey(id);
      if (fresh?.id === id) setFresh(null);   // a revoked key is no longer worth copying
      load();
    } catch (err) { setError(err.message); }
  };
  const copy = async () => {
    try { await navigator.clipboard.writeText(fresh.key); setCopied(true); } catch { setError('Copy failed. Select the key and copy it by hand.'); }
  };

  const active = (keys || []).filter((k) => !k.revoked_at);
  const revoked = (keys || []).filter((k) => k.revoked_at);

  return (
    <Panel title="API keys" aside={<span className="text-xs text-textMuted">{active.length} of 10 active</span>}>
      <p className="-mt-2 mb-5 max-w-[60ch] text-sm text-textSecondary">
        For the <span className="font-mono">veritas</span> command-line tool and CI jobs. A key can do everything you can, except create other keys.
      </p>

      {fresh && (
        <div className="glass-card mb-6" role="status">
          <div className="p-5">
            <div className="font-medium">Copy this key now. It will not be shown again.</div>
            <div className="mt-3 flex flex-col gap-3 sm:flex-row sm:items-center">
              <code className="min-w-0 flex-1 break-all rounded-2xl bg-black/40 px-4 py-3 font-mono text-sm text-uv">{fresh.key}</code>
              <Button variant="primary" onClick={copy}>{copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}{copied ? 'Copied' : 'Copy'}</Button>
            </div>
            <button type="button" onClick={() => setFresh(null)} className="mt-1 inline-flex min-h-[44px] items-center text-sm text-textMuted underline hover:text-textPrimary">I have saved it</button>
          </div>
        </div>
      )}

      <form onSubmit={create} className="flex flex-col gap-3 sm:flex-row">
        <Field label="Key name" hideLabel className="flex-1" value={name} onChange={(e) => setName(e.target.value)} placeholder="Key name, e.g. ci-pipeline" maxLength={64} required />
        <Button type="submit" variant="primary" loading={busy} disabled={!name.trim() || active.length >= 10}><KeyRound className="h-4 w-4" /> Create key</Button>
      </form>
      <div className="mt-4"><ErrorNote>{error}</ErrorNote></div>

      {keys === null && !error && <Loading className="mt-5">Loading keys…</Loading>}
      {keys && keys.length === 0 && <p className="mt-5 text-sm text-textSecondary">No keys yet. Create one to use the command-line tool.</p>}
      {active.length > 0 && (
        <ul className="mt-5 divide-y divide-borderHairline">
          {active.map((k) => (
            <li key={k.id} className="flex flex-wrap items-center justify-between gap-3 py-3.5">
              <div className="min-w-0">
                <div className="truncate font-medium">{k.name}</div>
                <div className="font-mono text-xs text-textMuted">{k.prefix}… · created {day(k.created_at)} · last used {day(k.last_used_at)}</div>
              </div>
              <Button variant="danger" size="sm" onClick={() => revoke(k.id)} aria-label={`Revoke key ${k.name}`}>Revoke</Button>
            </li>
          ))}
        </ul>
      )}
      {revoked.length > 0 && (
        <p className="mt-4 text-xs text-textMuted">{revoked.length} revoked {revoked.length === 1 ? 'key' : 'keys'}: {revoked.map((k) => k.name).join(', ')}</p>
      )}
    </Panel>
  );
}

function DangerZone() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [dataMsg, setDataMsg] = useState('');
  const [asking, setAsking] = useState(false);
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');

  const deleteData = async () => {
    setError(''); setDataMsg('');
    try {
      const { deleted } = await api.deleteMyData();
      const n = (deleted.analysis_results || 0) + (deleted.model_scans || 0);
      setDataMsg(`Deleted ${n} ${n === 1 ? 'result' : 'results'} and everything attached to them.`);
    } catch (e) {
      setError(e.message);
    }
  };
  const closeDialog = () => { setAsking(false); setPassword(''); setError(''); };

  const deleteAccount = async (e) => {
    e.preventDefault();
    setBusy('account'); setError('');
    try {
      await api.deleteAccount(password);
      await logout();
      navigate('/', { replace: true });
    } catch (err) {
      setError(err.message);
      setBusy('');
    }
  };

  return (
    <Panel title="Delete data">
      <div className="space-y-8">
        <div>
          <div className="font-medium">Delete my results</div>
          <p className="mt-1 max-w-[60ch] text-sm text-textSecondary">
            Erases every analysis, model scan, review case, defense action and usage counter stored for you. Your account and API keys stay. This cannot be undone.
          </p>
          <Notice tone="safe" className="mt-3">{dataMsg}</Notice>
          <div className="mt-4">
            <ConfirmButton label="Delete my results…" confirmLabel="Yes, delete all my results" cancelLabel="Keep them" onConfirm={deleteData} />
          </div>
        </div>

        {user?.role !== 'admin' && (
          <div className="border-t border-borderHairline pt-8">
            <div className="font-medium">Delete my account</div>
            <p className="mt-1 max-w-[60ch] text-sm text-textSecondary">
              Permanently deletes the account <span className="font-mono">{user?.name}</span>, all results, API keys and the audit log.
            </p>
            <div className="mt-4">
              <Button variant="danger" onClick={() => setAsking(true)}><Trash2 className="h-4 w-4" /> Delete my account…</Button>
            </div>
          </div>
        )}
        {!asking && <ErrorNote>{error}</ErrorNote>}
        <Dialog
          open={asking}
          onClose={closeDialog}
          title="Delete your account?"
          description={`This deletes ${user?.name || 'the account'}, every result, API key and audit entry for it. It cannot be undone.`}
        >
          <form onSubmit={deleteAccount} className="space-y-5">
            <Field label="Your password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" required autoFocus error={error || undefined} />
            <div className="flex flex-wrap justify-end gap-3">
              <Button variant="ghost" onClick={closeDialog} disabled={busy === 'account'}>Keep my account</Button>
              <Button type="submit" variant="danger" loading={busy === 'account'} disabled={!password}>Delete permanently</Button>
            </div>
          </form>
        </Dialog>
      </div>
    </Panel>
  );
}

// Audit actions the backend records, in words. An action not listed here is shown as its code.
const ACTIONS = {
  'account.signup': 'Account created',
  'account.password_reset': 'Password reset',
  'account.data_deleted': 'Results deleted',
  'apikey.created': 'API key created',
  'apikey.revoked': 'API key revoked',
  'analysis.uploaded': 'Batch examined',
  'analysis.model_scanned': 'Model file scanned',
  'analysis.deleted': 'Stored analysis deleted',
  'defense.quarantine': 'Rows quarantined',
  'hitl.decided': 'Review decision made',
};

function detailOf(e) {
  const d = e.details || {};
  return d.filename || d.name || d.decision?.replace(/_/g, ' ') || (d.samples_affected != null ? `${d.samples_affected} rows` : '');
}

/** GET /audit/events: a regular user gets their own actions; an admin gets everyone's, so it is narrowed to theirs. */
function Activity({ userId, isAdmin }) {
  const [events, setEvents] = useState(null);
  const [error, setError] = useState(null);
  const load = useCallback(() => api.getAuditEvents(50)
    .then((d) => { setEvents((d.events || []).filter((e) => !isAdmin || e.actor_id === userId)); setError(null); })
    .catch(setError), [userId, isAdmin]);
  useEffect(() => { load(); }, [load]);

  return (
    <Panel title="Recent activity" aside={<span className="text-xs text-textMuted">newest 50, from the audit log</span>}>
      {error ? (
        <ErrorState error={error} onRetry={() => { setError(null); load(); }} title="Activity could not load" className="border-0 bg-transparent p-0" />
      ) : events === null ? (
        <Loading>Loading activity…</Loading>
      ) : events.length === 0 ? (
        <p className="text-sm text-textSecondary">Nothing recorded yet. Examining a batch, scanning a file or managing keys shows up here.</p>
      ) : (
        // Beside the shorter left column, 50 rows would run the page far past it: on wide screens the list scrolls in its card.
        // It is focusable so the keyboard can scroll it too.
        <ul tabIndex={0} aria-label="Recent account activity" className="-mx-2 divide-y divide-borderHairline px-2 lg:max-h-[600px] lg:overflow-y-auto">
          {events.map((e) => (
            <li key={e.id} className="grid grid-cols-[minmax(0,1fr)_auto] items-baseline gap-3 py-2.5 text-sm">
              <span className="min-w-0">
                <span className={ACTIONS[e.action] ? '' : 'font-mono text-[13px]'}>{ACTIONS[e.action] || e.action}</span>
                {detailOf(e) && <span className="ml-2 truncate font-mono text-[13px] text-textMuted">{detailOf(e)}</span>}
              </span>
              <time className="whitespace-nowrap text-xs text-textMuted" dateTime={e.created_at}>{new Date(e.created_at).toLocaleString()}</time>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

export default function AccountPage() {
  const { user } = useAuth();
  return (
    <Page>
      <PageHeader title="Your" accent="account" lede={`Signed in as ${user?.name || 'you'}. Check today's usage, manage keys for the command-line tool, and delete what Veritas stores for you.`} />
      <div className="grid gap-8 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <div className="space-y-8">
          <Usage />
          <DangerZone />
        </div>
        <div className="space-y-8">
          <ApiKeys />
          <Activity userId={user?.id} isAdmin={user?.role === 'admin'} />
        </div>
      </div>
    </Page>
  );
}
