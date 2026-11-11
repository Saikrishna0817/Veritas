import { useCallback, useEffect, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Users, UserPlus, Trash2, RefreshCw, ShieldAlert, ScrollText, Database, ChevronDown, X } from 'lucide-react';
import { api } from '../services/api';
import { useAuth } from '../hooks/useAuth';
import { DATA_VERDICTS, MODEL_VERDICTS, PageHeader, Panel, Button, ErrorNote, Empty } from '../components/ui';

const ROLES = [
  { id: 'user', label: 'User', note: 'Runs checks and scans; sees only their own results.' },
  { id: 'analyst', label: 'Analyst', note: 'Same access as User today.' },
  { id: 'admin', label: 'Administrator', note: 'This console: accounts, audit log, every stored result.' },
];
const inputClass = 'w-full rounded-2xl border border-transparent bg-white/[0.06] px-4 py-3 text-[15px] transition-colors focus:border-uv focus:outline-none';

function when(iso) {
  return iso ? new Date(iso).toLocaleString() : '—';
}

/** Two-step destructive action: the first click asks, the second does it. */
function ConfirmButton({ label, confirmLabel, onConfirm, disabled, title }) {
  const [asking, setAsking] = useState(false);
  const [busy, setBusy] = useState(false);
  if (!asking) {
    return (
      <button
        type="button"
        disabled={disabled}
        title={title}
        aria-label={title || label}
        onClick={() => setAsking(true)}
        className="inline-flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-sm text-textMuted transition-colors hover:bg-statusCritical/10 hover:text-statusCritical disabled:cursor-not-allowed disabled:opacity-30"
      >
        <Trash2 className="h-4 w-4" aria-hidden /> <span className="sr-only sm:not-sr-only">{label}</span>
      </button>
    );
  }
  return (
    <span className="inline-flex items-center gap-1">
      <button
        type="button"
        disabled={busy}
        onClick={async () => { setBusy(true); try { await onConfirm(); } finally { setBusy(false); setAsking(false); } }}
        className="rounded-xl border border-statusCritical/50 px-3 py-1.5 text-sm text-statusCritical hover:bg-statusCritical/10 disabled:opacity-50"
      >
        {busy ? 'Working…' : confirmLabel}
      </button>
      <button type="button" onClick={() => setAsking(false)} className="rounded-xl px-2 py-1.5 text-sm text-textMuted hover:text-textPrimary" aria-label="Cancel">
        <X className="h-4 w-4" aria-hidden />
      </button>
    </span>
  );
}

function Count({ label, value, note }) {
  return (
    <div className="rounded-[28px] border border-white/[0.07] bg-[#1A1A1A] p-6">
      <div className="text-sm font-medium text-textSecondary">{label}</div>
      <div className="font-display mt-6 text-[clamp(2.2rem,3.2vw,3rem)] font-medium leading-none tracking-[-0.05em] tabular">{value}</div>
      {note && <div className="mt-2 text-xs text-textMuted">{note}</div>}
    </div>
  );
}

function verdictOf(v) {
  return DATA_VERDICTS[v] || MODEL_VERDICTS[v] || { label: v || '—', short: v || '—', color: 'var(--text-muted)' };
}

// ── Accounts ─────────────────────────────────────────────────────────────────

function NewAccount({ onCreated, onClose }) {
  const [form, setForm] = useState({ username: '', password: '', role: 'user' });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value });

  const submit = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError('');
    try {
      await api.createUser(form);
      onCreated(`Account ${form.username} created as ${ROLES.find((r) => r.id === form.role).label.toLowerCase()}.`);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <motion.form
      onSubmit={submit}
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: 'auto' }}
      exit={{ opacity: 0, height: 0 }}
      transition={{ duration: 0.4, ease: [0.19, 1, 0.22, 1] }}
      className="overflow-hidden"
    >
      <div className="glass-card mb-6">
        <div className="space-y-5 p-6">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h3 className="text-lg font-medium">New account</h3>
              <p className="mt-1 text-sm text-textSecondary">The person signs in with these details. No confirmation email is sent.</p>
            </div>
            <button type="button" onClick={onClose} aria-label="Close" className="rounded-xl p-1.5 text-textMuted hover:bg-white/[0.06] hover:text-textPrimary"><X className="h-4 w-4" /></button>
          </div>
          <div className="grid gap-4 md:grid-cols-2">
            <label className="block">
              <span className="rule-label">Username</span>
              <input value={form.username} onChange={set('username')} required minLength={3} maxLength={64} autoComplete="off" placeholder="3 to 64 characters" className={`${inputClass} mt-1.5`} />
            </label>
            <label className="block">
              <span className="rule-label">Password</span>
              <input type="password" value={form.password} onChange={set('password')} required minLength={10} maxLength={72} autoComplete="new-password" placeholder="At least 10 characters" className={`${inputClass} mt-1.5`} />
            </label>
          </div>
          <fieldset>
            <legend className="rule-label">Role</legend>
            <div className="mt-2 grid gap-2 md:grid-cols-3">
              {ROLES.map((r) => (
                <label key={r.id} className={`flex cursor-pointer gap-3 rounded-2xl border p-3.5 transition-colors ${form.role === r.id ? 'border-violet bg-violet/[0.08]' : 'border-borderHairline hover:border-borderStrong'}`}>
                  <input type="radio" name="role" value={r.id} checked={form.role === r.id} onChange={set('role')} className="mt-1 accent-[#B9A3FF]" />
                  <span><span className="block text-sm font-medium">{r.label}</span><span className="block text-xs text-textMuted">{r.note}</span></span>
                </label>
              ))}
            </div>
          </fieldset>
          <ErrorNote>{error}</ErrorNote>
          <Button type="submit" variant="primary" loading={busy}><UserPlus className="h-4 w-4" /> Create account</Button>
        </div>
      </div>
    </motion.form>
  );
}

function AccountsTab({ users, loading, me, onChanged, notify, fail }) {
  const [adding, setAdding] = useState(false);
  return (
    <Panel title="Accounts" aside={!adding && <Button variant="primary" className="px-4 py-2.5" onClick={() => setAdding(true)}><UserPlus className="h-4 w-4" /> New account</Button>}>
      <AnimatePresence>
        {adding && <NewAccount onClose={() => setAdding(false)} onCreated={(msg) => { setAdding(false); notify(msg); onChanged(); }} />}
      </AnimatePresence>
      {loading ? <p className="text-sm text-textMuted" role="status">Loading accounts…</p> : users.length === 0 ? (
        <p className="text-sm text-textSecondary">No accounts yet.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[560px] text-sm">
            <thead>
              <tr className="rule-label border-b border-borderHairline text-left">
                <th className="py-3 pr-4 font-normal">Account</th>
                <th className="py-3 pr-4 font-normal">Role</th>
                <th className="py-3 pr-4 font-normal">Created</th>
                <th className="py-3 text-right font-normal"><span className="sr-only">Actions</span></th>
              </tr>
            </thead>
            <tbody>
              {users.map((u) => {
                const self = u.id === me?.id;
                return (
                  <tr key={u.id} className="border-b border-borderHairline last:border-0">
                    <td className="py-3.5 pr-4">
                      <div className="flex items-center gap-3">
                        <span className="icon-chip h-9 w-9 text-sm font-medium uppercase">{u.username.charAt(0)}</span>
                        <div>
                          <div className="font-medium">{u.username}{self && <span className="ml-2 rounded-lg bg-white/[0.08] px-2 py-0.5 text-[11px] text-textSecondary">you</span>}</div>
                          <div className="font-mono text-[11px] text-textMuted">{u.id}</div>
                        </div>
                      </div>
                    </td>
                    <td className="py-3.5 pr-4">
                      <span className={`rounded-lg px-2 py-1 text-xs ${u.role === 'admin' ? 'bg-uv/15 text-uv' : 'bg-white/[0.06] text-textSecondary'}`}>
                        {ROLES.find((r) => r.id === u.role)?.label || u.role}
                      </span>
                    </td>
                    <td className="py-3.5 pr-4 text-textMuted">{when(u.created_at)}</td>
                    <td className="py-3.5 text-right">
                      <ConfirmButton
                        label="Delete"
                        confirmLabel={`Delete ${u.username}`}
                        disabled={self}
                        title={self ? 'You cannot delete the account you are signed in with' : `Delete account ${u.username}`}
                        onConfirm={async () => {
                          try { await api.deleteUser(u.id); notify(`Account ${u.username} deleted.`); onChanged(); } catch (e) { fail(e.message); }
                        }}
                      />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}

// ── Stored analyses ──────────────────────────────────────────────────────────

function AnalysesTab({ rows, loading, onChanged, notify, fail }) {
  return (
    <Panel title="Every stored analysis" aside={<span className="text-xs text-textMuted">all accounts · newest 100</span>}>
      <p className="-mt-2 mb-5 max-w-[70ch] text-sm text-textSecondary">Batches and model scans from every account. Deleting one removes it for its owner too, and cannot be undone.</p>
      {loading ? <p className="text-sm text-textMuted" role="status">Loading…</p> : rows.length === 0 ? (
        <p className="text-sm text-textSecondary">Nothing stored yet.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] text-sm">
            <thead>
              <tr className="rule-label border-b border-borderHairline text-left">
                <th className="py-3 pr-4 font-normal">File</th>
                <th className="py-3 pr-4 font-normal">Source</th>
                <th className="py-3 pr-4 font-normal">Verdict</th>
                <th className="py-3 pr-4 font-normal">When</th>
                <th className="py-3 text-right font-normal"><span className="sr-only">Actions</span></th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const v = verdictOf(r.verdict);
                const name = r.filename || r.model_filename || r.id.slice(0, 8);
                return (
                  <tr key={`${r.source}-${r.id}`} className="border-b border-borderHairline last:border-0">
                    <td className="max-w-[260px] truncate py-3 pr-4 font-mono">{name}</td>
                    <td className="py-3 pr-4 text-textSecondary">{(r.source || '').replace('_', ' ')}</td>
                    <td className="py-3 pr-4"><span className="text-xs font-medium" style={{ color: v.color }}>{v.short || v.label}</span></td>
                    <td className="py-3 pr-4 text-textMuted">{when(r.created_at)}</td>
                    <td className="py-3 text-right">
                      <ConfirmButton
                        label="Delete"
                        confirmLabel="Delete for good"
                        title={`Delete ${name}`}
                        onConfirm={async () => {
                          try { await api.deleteAnalysisHistory(r.id); notify(`${name} deleted.`); onChanged(); } catch (e) { fail(e.message); }
                        }}
                      />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}

// ── Audit log ────────────────────────────────────────────────────────────────

function AuditTab({ events, loading, users }) {
  const [open, setOpen] = useState(null);
  const nameOf = (id) => users.find((u) => u.id === id)?.username || id;
  return (
    <Panel title="Audit log" aside={<span className="text-xs text-textMuted">newest 100</span>}>
      {loading ? <p className="text-sm text-textMuted" role="status">Loading…</p> : events.length === 0 ? (
        <p className="text-sm text-textSecondary">No events recorded yet.</p>
      ) : (
        <ul className="divide-y divide-borderHairline">
          {events.map((e) => {
            const expanded = open === e.id;
            return (
              <li key={e.id}>
                <button
                  type="button"
                  onClick={() => setOpen(expanded ? null : e.id)}
                  aria-expanded={expanded}
                  className="grid w-full grid-cols-[1fr_auto] items-center gap-3 py-3 text-left text-sm hover:bg-white/[0.02] sm:grid-cols-[180px_1fr_160px_auto]"
                >
                  <span className="text-textMuted">{when(e.created_at)}</span>
                  <span className="hidden font-mono sm:block">{e.action}</span>
                  <span className="hidden truncate text-textSecondary sm:block">{nameOf(e.actor_id)}</span>
                  <ChevronDown className={`h-4 w-4 text-textMuted transition-transform ${expanded ? 'rotate-180' : ''}`} aria-hidden />
                  <span className="col-span-2 font-mono text-xs text-textSecondary sm:hidden">{e.action} · {nameOf(e.actor_id)}</span>
                </button>
                {expanded && (
                  <pre className="mb-3 overflow-x-auto rounded-2xl bg-black/40 p-4 font-mono text-xs text-textSecondary">{JSON.stringify(e, null, 2)}</pre>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </Panel>
  );
}

// ── Review queue and quarantine (this admin account's own analyses) ──────────

function ReviewTab({ cases, status, loading, onChanged, notify, fail }) {
  const [busy, setBusy] = useState('');
  const decide = async (caseId, decision) => {
    setBusy(caseId);
    try {
      await api.submitReviewDecision(caseId, decision);
      notify(decision === 'approve_quarantine' ? 'Case approved: its rows were quarantined.' : 'Case marked safe.');
      onChanged();
    } catch (e) {
      fail(e.message);
    } finally {
      setBusy('');
    }
  };
  return (
    <div className="space-y-6">
      <p className="max-w-[75ch] text-sm text-textSecondary">
        These controls act on analyses run with <span className="text-textPrimary">this admin account</span>. Each analyst's queue and quarantine are on their own Response desk.
      </p>
      <div className="grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <Panel title="Quarantine">
          <dl className="grid grid-cols-2 gap-5">
            <div><dt className="rule-label">Rows quarantined</dt><dd className="font-display mt-1 text-3xl font-medium tracking-[-0.04em]">{status?.total_quarantined ?? '—'}</dd></div>
            <div><dt className="rule-label">Defense actions</dt><dd className="font-display mt-1 text-3xl font-medium tracking-[-0.04em]">{status?.n_defense_actions ?? '—'}</dd></div>
          </dl>
          <p className="mt-5 text-sm text-textSecondary">
            Quarantines the rows your latest dataset analysis flagged, capped at 5% of that batch. It needs a CSV or reference-dataset result with row flags.
          </p>
          <div className="mt-5">
            <ConfirmButtonLarge
              label="Quarantine flagged rows"
              confirmLabel="Yes, quarantine them"
              onConfirm={async () => {
                try {
                  const r = await api.triggerQuarantine();
                  notify(`${r.samples_affected ?? 0} rows quarantined.`);
                  onChanged();
                } catch (e) { fail(e.message); }
              }}
            />
          </div>
        </Panel>
        <Panel title="Review queue" aside={<span className="text-xs text-textMuted">{cases.length} waiting</span>}>
          {loading ? <p className="text-sm text-textMuted" role="status">Loading…</p> : cases.length === 0 ? (
            <p className="text-sm text-textSecondary">Nothing waiting. A case opens when a batch lands between suspicious and confirmed.</p>
          ) : (
            <ul className="divide-y divide-borderHairline">
              {cases.map((c) => (
                <li key={c.case_id} className="flex flex-wrap items-center justify-between gap-3 py-3.5">
                  <div className="min-w-0">
                    <div className="font-mono text-sm">{c.case_id.slice(0, 12)}…</div>
                    <div className="text-xs text-textMuted">suspicion {Math.round((c.suspicion_score || 0) * 100)}% · {c.n_samples} rows · {when(c.created_at)}</div>
                  </div>
                  <div className="flex gap-2">
                    <Button variant="danger" className="px-4 py-2" disabled={busy === c.case_id} onClick={() => decide(c.case_id, 'approve_quarantine')}>Quarantine</Button>
                    <Button variant="ghost" className="px-4 py-2" disabled={busy === c.case_id} onClick={() => decide(c.case_id, 'mark_safe')}>Mark safe</Button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>
    </div>
  );
}

function ConfirmButtonLarge({ label, confirmLabel, onConfirm }) {
  const [asking, setAsking] = useState(false);
  const [busy, setBusy] = useState(false);
  if (!asking) return <Button variant="danger" onClick={() => setAsking(true)}><ShieldAlert className="h-4 w-4" /> {label}</Button>;
  return (
    <div className="flex flex-wrap gap-2">
      <Button variant="danger" loading={busy} onClick={async () => { setBusy(true); try { await onConfirm(); } finally { setBusy(false); setAsking(false); } }}>{confirmLabel}</Button>
      <Button variant="ghost" onClick={() => setAsking(false)}>Cancel</Button>
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────

const TABS = [
  { id: 'accounts', label: 'Accounts', icon: Users },
  { id: 'analyses', label: 'Stored analyses', icon: Database },
  { id: 'audit', label: 'Audit log', icon: ScrollText },
  { id: 'review', label: 'Review and quarantine', icon: ShieldAlert },
];

export default function AdminDashboard() {
  const { user: me } = useAuth();
  const [tab, setTab] = useState('accounts');
  const [users, setUsers] = useState([]);
  const [analyses, setAnalyses] = useState([]);
  const [events, setEvents] = useState([]);
  const [cases, setCases] = useState([]);
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const load = useCallback(() => Promise.all([
    api.getUsers().then(setUsers),
    api.getAnalysisHistory(100).then((d) => setAnalyses(d.results || [])),
    api.getAuditEvents(100).then((d) => setEvents(d.events || [])),
    api.getPendingReviews().then((d) => setCases(d.cases || [])),
    api.getDefenseStatus().then(setStatus),
  ]).catch((e) => setError(e.message)).finally(() => setLoading(false)), []);

  useEffect(() => { load(); }, [load]);

  const notify = (m) => { setError(''); setMessage(m); };
  const fail = (m) => { setMessage(''); setError(m); };
  const reload = () => { setLoading(true); load(); };
  const counts = { accounts: users.length, analyses: analyses.length, audit: events.length, review: cases.length };

  return (
    <div className="mx-auto max-w-[1360px] px-5 py-10 md:px-10 md:py-14">
      <PageHeader title="Admin" accent="console" lede="Manage accounts, read the audit log and every stored analysis, and work the review queue for this admin account.">
        <Button variant="ghost" onClick={reload}><RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} /> Refresh</Button>
      </PageHeader>

      <section aria-label="Totals" className="mb-8 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Count label="Accounts" value={users.length} note={`${users.filter((u) => u.role === 'admin').length} administrator${users.filter((u) => u.role === 'admin').length === 1 ? '' : 's'}`} />
        <Count label="Stored analyses" value={analyses.length} note="newest 100 across all accounts" />
        <Count label="Audit events" value={events.length} note="newest 100" />
        <Count label="Waiting for review" value={cases.length} note="this admin account" />
      </section>

      {(message || error) && (
        <div className="mb-6">
          {message && <div role="status" className="flex items-center justify-between gap-3 rounded-2xl border border-statusSafe/40 bg-statusSafe/[0.07] px-4 py-3 text-sm text-statusSafe">{message}<button onClick={() => setMessage('')} aria-label="Dismiss"><X className="h-4 w-4" /></button></div>}
          <ErrorNote>{error}</ErrorNote>
        </div>
      )}

      <div className="mb-6 flex flex-wrap gap-2" role="group" aria-label="Admin sections">
        {TABS.map((t) => (
          <button
            key={t.id}
            aria-pressed={tab === t.id}
            onClick={() => setTab(t.id)}
            className={`inline-flex items-center gap-2 rounded-2xl border px-4 py-2 text-sm font-medium transition-colors ${tab === t.id ? 'border-uv bg-uv/[0.08] text-uv' : 'border-borderHairline text-textSecondary hover:border-borderStrong hover:text-textPrimary'}`}
          >
            <t.icon className="h-4 w-4" aria-hidden /> {t.label} <span className="text-textMuted">{counts[t.id]}</span>
          </button>
        ))}
      </div>

      {tab === 'accounts' && <AccountsTab users={users} loading={loading} me={me} onChanged={reload} notify={notify} fail={fail} />}
      {tab === 'analyses' && <AnalysesTab rows={analyses} loading={loading} onChanged={reload} notify={notify} fail={fail} />}
      {tab === 'audit' && <AuditTab events={events} loading={loading} users={users} />}
      {tab === 'review' && <ReviewTab cases={cases} status={status} loading={loading} onChanged={reload} notify={notify} fail={fail} />}
      {!loading && error && users.length === 0 && <Empty title="The console could not load">{error}</Empty>}
    </div>
  );
}
