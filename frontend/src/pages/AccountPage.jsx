import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Copy, Check, KeyRound, Trash2 } from 'lucide-react';
import { api } from '../services/api';
import { useAuth } from '../hooks/useAuth';
import { PageHeader, Panel, Button, ErrorNote } from '../components/ui';

const EASE = [0.19, 1, 0.22, 1];
const USAGE = [
  ['analyses', 'Analyses today', (n) => n.toLocaleString()],
  ['rows', 'Rows checked today', (n) => n.toLocaleString()],
  ['scan_mb', 'Model files scanned today', (n) => `${n.toLocaleString()} MB`],
];
const inputClass = 'w-full rounded-2xl border border-transparent bg-white/[0.06] px-4 py-3 text-[15px] transition-colors focus:border-uv focus:outline-none';

function day(iso) {
  return iso ? new Date(iso).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' }) : 'never';
}

function Usage() {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  useEffect(() => { api.getUsage().then(setData).catch((e) => setError(e.message)); }, []);
  return (
    <Panel title="Usage" aside={<span className="text-xs text-textMuted">Resets at 00:00 UTC</span>}>
      <ErrorNote>{error}</ErrorNote>
      {!data && !error && <p className="text-sm text-textMuted" role="status">Loading…</p>}
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
                      className="h-full rounded-full"
                      style={{ background: share >= 0.9 ? 'var(--flag)' : 'var(--uv)' }}
                      initial={{ width: 0 }}
                      animate={{ width: `${share * 100}%` }}
                      transition={{ duration: 1.4, delay: i * 0.1, ease: EASE }}
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
            <button onClick={() => setFresh(null)} className="mt-3 text-sm text-textMuted underline hover:text-textPrimary">I have saved it</button>
          </div>
        </div>
      )}

      <form onSubmit={create} className="flex flex-col gap-3 sm:flex-row">
        <label className="flex-1">
          <span className="sr-only">Key name</span>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Key name, e.g. ci-pipeline" maxLength={64} required className={inputClass} />
        </label>
        <Button type="submit" variant="primary" loading={busy} disabled={!name.trim() || active.length >= 10}><KeyRound className="h-4 w-4" /> Create key</Button>
      </form>
      <div className="mt-4"><ErrorNote>{error}</ErrorNote></div>

      {keys === null && !error && <p className="mt-5 text-sm text-textMuted" role="status">Loading…</p>}
      {keys && keys.length === 0 && <p className="mt-5 text-sm text-textSecondary">No keys yet. Create one to use the command-line tool.</p>}
      {active.length > 0 && (
        <ul className="mt-5 divide-y divide-borderHairline">
          {active.map((k) => (
            <li key={k.id} className="flex flex-wrap items-center justify-between gap-3 py-3.5">
              <div className="min-w-0">
                <div className="truncate font-medium">{k.name}</div>
                <div className="font-mono text-xs text-textMuted">{k.prefix}… · created {day(k.created_at)} · last used {day(k.last_used_at)}</div>
              </div>
              <Button variant="danger" className="rounded-2xl px-4 py-2" onClick={() => revoke(k.id)} aria-label={`Revoke key ${k.name}`}>Revoke</Button>
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
  const [confirmData, setConfirmData] = useState(false);
  const [dataMsg, setDataMsg] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');

  const deleteData = async () => {
    setBusy('data'); setError(''); setDataMsg('');
    try {
      const { deleted } = await api.deleteMyData();
      const n = (deleted.analysis_results || 0) + (deleted.model_scans || 0);
      setDataMsg(`Deleted ${n} ${n === 1 ? 'result' : 'results'} and everything attached to them.`);
      setConfirmData(false);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy('');
    }
  };

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
          {dataMsg && <p role="status" className="mt-3 text-sm text-statusSafe">{dataMsg}</p>}
          <div className="mt-4 flex flex-wrap gap-3">
            {confirmData ? (
              <>
                <Button variant="danger" loading={busy === 'data'} onClick={deleteData}><Trash2 className="h-4 w-4" /> Yes, delete all my results</Button>
                <Button variant="ghost" onClick={() => setConfirmData(false)}>Keep them</Button>
              </>
            ) : (
              <Button variant="danger" onClick={() => { setConfirmData(true); setDataMsg(''); }}>Delete my results…</Button>
            )}
          </div>
        </div>

        {user?.role !== 'admin' && (
          <form onSubmit={deleteAccount} className="border-t border-borderHairline pt-8">
            <div className="font-medium">Delete my account</div>
            <p className="mt-1 max-w-[60ch] text-sm text-textSecondary">
              Permanently deletes the account <span className="font-mono">{user?.name}</span>, all results, API keys and the audit log. Enter your password to confirm.
            </p>
            <div className="mt-4 flex flex-col gap-3 sm:flex-row">
              <label className="flex-1 sm:max-w-xs">
                <span className="sr-only">Password</span>
                <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Your password" autoComplete="current-password" required className={inputClass} />
              </label>
              <Button type="submit" variant="danger" loading={busy === 'account'} disabled={!password}>Delete account permanently</Button>
            </div>
          </form>
        )}
        <ErrorNote>{error}</ErrorNote>
      </div>
    </Panel>
  );
}

export default function AccountPage() {
  const { user } = useAuth();
  return (
    <div className="mx-auto max-w-[1360px] px-5 py-10 md:px-10 md:py-14">
      <PageHeader title="Your" accent="account" lede={`Signed in as ${user?.name || 'you'}. Check today's usage, manage keys for the command-line tool, and delete what Veritas stores for you.`} />
      <div className="grid gap-8 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <div className="space-y-8">
          <Usage />
          <DangerZone />
        </div>
        <ApiKeys />
      </div>
    </div>
  );
}
