import { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';
import { useAuth } from '../hooks/useAuth';
import { fmt } from '../components/results';
import { Skeleton } from '../components/Skeleton';
import { PageIntro, PageTitle } from '../ui/Reveal';

/**
 * The signed-in person's own account: today's usage against the daily limits, how long results are kept, API keys
 * for scripts and CI, and the two deletions (results only, or the whole account). Every number comes from the API.
 */
const USAGE = [['analyses', 'Analyses'], ['rows', 'Rows checked'], ['scan_mb', 'Megabytes scanned']];
const when = (iso) => (iso ? new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }) : '');

function Usage() {
  const [u, setU] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => { api.getUsage().then(setU).catch((e) => setError(e.message)); }, []);
  if (error) return <p className="in-error" role="alert">{error}</p>;
  if (!u) return <Skeleton label="Loading your usage…" lines={0} rows={3} />;
  return (
    <>
      <div className="in-usage">
        {USAGE.map(([k, label]) => {
          const used = u.used?.[k] || 0;
          const limit = u.limits?.[k];
          return (
            <div key={k} className="in-usage-row">
              <span className="in-usage-name">{label}</span>
              {limit ? (
                <>
                  <div className="in-track" role="img" aria-label={`${label}: ${fmt(used)} of ${fmt(limit)} today`}>
                    <div className="in-usage-fill" style={{ width: `${Math.min(100, (used / limit) * 100)}%` }} />
                  </div>
                  <span className="in-num in-usage-val">{fmt(used)} <small>of {fmt(limit)}</small></span>
                </>
              ) : <span className="in-hint in-usage-val">{fmt(used)} today, no daily limit</span>}
            </div>
          );
        })}
      </div>
      <p className="in-hint">Counts start again each day (UTC). {u.retention_days ? `Results are kept for ${fmt(u.retention_days)} days, then deleted.` : 'Results are kept until you delete them.'}</p>
    </>
  );
}

function ApiKeys() {
  const [keys, setKeys] = useState(null);
  const [name, setName] = useState('');
  const [fresh, setFresh] = useState(null);       // the new key, shown once
  const [confirm, setConfirm] = useState(null);   // id of the key whose revoke awaits a second click
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const load = useCallback(() => api.listApiKeys().then((d) => setKeys(d.keys || [])).catch((e) => setError(e.message)), []);
  useEffect(() => { load(); }, [load]);

  const create = async (e) => {
    e.preventDefault();
    setBusy(true); setError(null); setCopied(false);
    try { setFresh(await api.createApiKey(name.trim())); setName(''); await load(); }
    catch (err) { setError(err.message); }
    finally { setBusy(false); }
  };
  const revoke = async (id) => {
    if (confirm !== id) return setConfirm(id);
    setConfirm(null); setError(null);
    try { await api.revokeApiKey(id); await load(); } catch (err) { setError(err.message); }
  };
  const copy = async () => {
    try { await navigator.clipboard.writeText(fresh.key); setCopied(true); } catch { setCopied(false); }
  };

  return (
    <>
      <form className="in-actions" onSubmit={create}>
        <input className="in-input" placeholder="Key name, e.g. CI pipeline" aria-label="New key name" value={name} onChange={(e) => setName(e.target.value)} required maxLength={64} />
        <button className="in-btn" type="submit" disabled={busy || !name.trim()}>{busy ? 'Creating…' : 'Create key'}</button>
      </form>
      {fresh && (
        <div className="in-key-once" role="status">
          <p><b>{fresh.name}</b>: copy this key now. It is shown once; only its hash is stored.</p>
          <code className="in-code">{fresh.key}</code>
          <div className="in-actions">
            <button type="button" className="in-btn in-btn-quiet" onClick={copy}>{copied ? 'Copied' : 'Copy key'}</button>
            <button type="button" className="in-btn in-btn-quiet" onClick={() => setFresh(null)}>I have stored it</button>
          </div>
        </div>
      )}
      {error && <p className="in-error" role="alert">{error}</p>}
      {!keys && !error && <Skeleton label="Loading your API keys…" lines={0} rows={2} />}
      {keys && !keys.length && <p className="in-empty">No API keys. Create one to use the command-line tool or CI.</p>}
      {keys?.length > 0 && (
        <div className="in-table-wrap">
          <table className="in-table in-table-stack">
            <thead><tr><th>Name</th><th>Starts with</th><th>Created</th><th>Last used</th><th>Status</th><th><span className="in-sr">Action</span></th></tr></thead>
            <tbody>{keys.map((k) => (
              <tr key={k.id}>
                <td data-label="Name">{k.name}</td>
                <td className="in-path" data-label="Starts with">{k.prefix}…</td>
                <td className="in-tag" data-label="Created">{when(k.created_at)}</td>
                <td className="in-tag" data-label="Last used">{k.last_used_at ? when(k.last_used_at) : 'Never'}</td>
                <td className="in-tag" data-label="Status">{k.revoked_at ? `Revoked ${when(k.revoked_at)}` : 'Active'}</td>
                <td data-label="Action">{!k.revoked_at && (
                  <button className="in-btn in-btn-quiet" onClick={() => revoke(k.id)} onBlur={() => setConfirm(null)}>{confirm === k.id ? 'Confirm revoke' : 'Revoke'}</button>
                )}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      )}
    </>
  );
}

function DeleteData() {
  const [step, setStep] = useState('idle');   // idle → confirm → done
  const [counts, setCounts] = useState(null);
  const [error, setError] = useState(null);
  const run = async () => {
    setError(null);
    try { const r = await api.deleteMyData(); setCounts(r.deleted || {}); setStep('done'); }
    catch (e) { setError(e.message); setStep('idle'); }
  };
  const total = counts ? Object.values(counts).reduce((a, b) => a + (Number(b) || 0), 0) : 0;
  return (
    <div className="in-danger">
      <h3 className="in-h3">Delete my data</h3>
      <p>Erases every result, scan, review case and usage counter stored for you. Your account, API keys and audit log stay. This cannot be undone.</p>
      {step === 'done' ? <p className="in-ok" role="status">Deleted {fmt(total)} stored records.</p> : (
        <div className="in-actions">
          {step === 'idle' && <button className="in-btn in-btn-quiet" onClick={() => setStep('confirm')}>Delete my data…</button>}
          {step === 'confirm' && (
            <>
              <button className="in-btn in-btn-danger" onClick={run}>Yes, delete all my results</button>
              <button className="in-btn in-btn-quiet" onClick={() => setStep('idle')}>Keep them</button>
            </>
          )}
        </div>
      )}
      {error && <p className="in-error" role="alert">{error}</p>}
    </div>
  );
}

function DeleteAccount({ admin }) {
  const { logout } = useAuth();
  const navigate = useNavigate();
  const [password, setPassword] = useState('');
  const [sure, setSure] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const run = async (e) => {
    e.preventDefault();
    setBusy(true); setError(null);
    try {
      await api.deleteAccount(password);
      await logout();
      navigate('/login', { replace: true });
    } catch (err) { setError(err.message); setBusy(false); }
  };
  return (
    <div className="in-danger">
      <h3 className="in-h3">Delete my account</h3>
      <p>Erases everything above plus the account itself, its API keys, pending email links, job records and audit log. You are signed out. This cannot be undone.</p>
      {admin ? <p className="in-hint">Administrator accounts cannot be deleted here.</p> : (
        <form className="in-auth-form" onSubmit={run} style={{ maxWidth: 400 }}>
          <label>Your password<input className="in-input" type="password" value={password} onChange={(e) => setPassword(e.target.value)} required autoComplete="current-password" /></label>
          <label className="in-radio"><input type="checkbox" checked={sure} onChange={(e) => setSure(e.target.checked)} /><span>I understand my account and everything stored for it will be deleted.</span></label>
          {error && <p className="in-error" role="alert" style={{ margin: 0 }}>{error}</p>}
          <div><button className="in-btn in-btn-danger" type="submit" disabled={busy || !sure || !password}>{busy ? 'Deleting…' : 'Delete my account'}</button></div>
        </form>
      )}
    </div>
  );
}

export default function AccountPage() {
  const { user } = useAuth();
  const admin = user?.role === 'admin';
  return (
    <main className="in-main"><div className="in-page">
      <p className="in-h3">Account</p>
      <PageTitle>{user?.name || 'Your account'}</PageTitle>
      <PageIntro>Your usage today, your API keys, and what is stored for you.</PageIntro>
      <section className="in-section">
        <h2 className="in-h2">Today</h2>
        <Usage />
      </section>
      <section className="in-section">
        <h2 className="in-h2">API keys</h2>
        <p className="in-sub">For the command-line tool and CI. At most 10 active keys. A key signs in as you; it cannot create other keys.</p>
        <ApiKeys />
      </section>
      <section className="in-section">
        <h2 className="in-h2">Your data</h2>
        <DeleteData />
        <DeleteAccount admin={admin} />
      </section>
    </div></main>
  );
}
