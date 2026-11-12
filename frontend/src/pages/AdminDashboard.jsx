import { useCallback, useEffect, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Users, UserPlus, RefreshCw, ShieldAlert, ScrollText, Database, ChevronDown, X } from 'lucide-react';
import { api } from '../services/api';
import { useAuth } from '../hooks/useAuth';
import {
  PageHeader, Panel, Button, IconButton, ErrorNote, ErrorState, Notice, Field, Tabs, TabPanel, Table, Badge, VerdictBadge,
  ConfirmButton, Loading, Page, StatCard, RadioCards, EASE, DURATION, percent,
} from '../components/ui';

const ROLES = [
  { id: 'user', label: 'User', note: 'Runs checks and scans; sees only their own results.' },
  { id: 'analyst', label: 'Analyst', note: 'Same access as User today.' },
  { id: 'admin', label: 'Administrator', note: 'This console: accounts, audit log, every stored result.' },
];
function when(iso) {
  return iso ? new Date(iso).toLocaleString() : '—';
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
      transition={{ duration: DURATION.base, ease: EASE }}
      className="overflow-hidden"
    >
      <div className="glass-card mb-6">
        <div className="space-y-5 p-6">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h3 className="text-lg font-medium">New account</h3>
              <p className="mt-1 text-sm text-textSecondary">The person signs in with these details. No confirmation email is sent.</p>
            </div>
            <IconButton icon={X} label="Close the new-account form" onClick={onClose} className="-mr-2 -mt-2" />
          </div>
          <div className="grid gap-4 md:grid-cols-2">
            <Field label="Username" value={form.username} onChange={set('username')} required minLength={3} maxLength={64} autoComplete="off" placeholder="3 to 64 characters" />
            <Field label="Password" type="password" value={form.password} onChange={set('password')} required minLength={10} maxLength={72} autoComplete="new-password" placeholder="At least 10 characters" />
          </div>
          <RadioCards name="role" label="Role" options={ROLES.map((r) => ({ ...r, hint: r.note }))} value={form.role} onChange={(role) => setForm({ ...form, role })} />
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
    <Panel title="Accounts" aside={!adding && <Button variant="primary" size="sm" onClick={() => setAdding(true)}><UserPlus className="h-4 w-4" /> New account</Button>}>
      <AnimatePresence>
        {adding && <NewAccount onClose={() => setAdding(false)} onCreated={(msg) => { setAdding(false); notify(msg); onChanged(); }} />}
      </AnimatePresence>
      {loading ? <Loading>Loading accounts…</Loading> : users.length === 0 ? (
        <p className="text-sm text-textSecondary">No accounts yet.</p>
      ) : (
        <Table minWidth={560} className="-mx-3">
            <thead>
              <tr>
                <th>Account</th>
                <th>Role</th>
                <th>Created</th>
                <th className="num"><span className="sr-only">Actions</span></th>
              </tr>
            </thead>
            <tbody>
              {users.map((u) => {
                const self = u.id === me?.id;
                return (
                  <tr key={u.id}>
                    <td>
                      <div className="flex items-center gap-3">
                        <span className="icon-chip h-9 w-9 text-sm font-medium uppercase">{u.username.charAt(0)}</span>
                        <div>
                          <div className="font-medium">{u.username}{self && <Badge tone="neutral" dot={false} className="ml-2">you</Badge>}</div>
                          <div className="font-mono text-[11px] text-textMuted">{u.id}</div>
                        </div>
                      </div>
                    </td>
                    <td>
                      <Badge tone={u.role === 'admin' ? 'accent' : 'neutral'} dot={false}>{ROLES.find((r) => r.id === u.role)?.label || u.role}</Badge>
                    </td>
                    <td className="nowrap text-textMuted">{when(u.created_at)}</td>
                    <td className="num">
                      <ConfirmButton
                        size="sm"
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
        </Table>
      )}
    </Panel>
  );
}

// ── Stored analyses ──────────────────────────────────────────────────────────

function AnalysesTab({ rows, loading, onChanged, notify, fail }) {
  return (
    <Panel title="Every stored analysis" aside={<span className="text-xs text-textMuted">all accounts · newest 100</span>}>
      <p className="-mt-2 mb-5 max-w-[70ch] text-sm text-textSecondary">Batches and model scans from every account. Deleting one removes it for its owner too, and cannot be undone.</p>
      {loading ? <Loading>Loading stored analyses…</Loading> : rows.length === 0 ? (
        <p className="text-sm text-textSecondary">Nothing stored yet.</p>
      ) : (
        <Table minWidth={640} className="-mx-3">
            <thead>
              <tr>
                <th>File</th>
                <th>Source</th>
                <th>Verdict</th>
                <th>When</th>
                <th className="num"><span className="sr-only">Actions</span></th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const name = r.filename || r.model_filename || r.id.slice(0, 8);
                return (
                  <tr key={`${r.source}-${r.id}`}>
                    <td className="max-w-[260px] truncate font-mono">{name}</td>
                    <td className="nowrap text-textSecondary">{(r.source || '').replace('_', ' ')}</td>
                    <td><VerdictBadge verdict={r.verdict} kind={r.source === 'model_scan' ? 'model' : 'data'} /></td>
                    <td className="nowrap text-textMuted">{when(r.created_at)}</td>
                    <td className="num">
                      <ConfirmButton
                        size="sm"
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
        </Table>
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
      {loading ? <Loading>Loading the audit log…</Loading> : events.length === 0 ? (
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
                  className="grid min-h-[48px] w-full grid-cols-[1fr_auto] items-center gap-3 rounded-xl px-2 py-3 text-left text-sm transition-colors hover:bg-white/[0.03] sm:grid-cols-[180px_1fr_160px_auto]"
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
            <ConfirmButton
              icon={ShieldAlert}
              label="Quarantine flagged rows"
              confirmLabel="Yes, quarantine them"
              onConfirm={async () => {
                try {
                  const r = await api.triggerQuarantine();
                  notify(r.samples_affected != null ? `${r.samples_affected} rows quarantined.` : 'Quarantine ran.');
                  onChanged();
                } catch (e) { fail(e.message); }
              }}
            />
          </div>
        </Panel>
        <Panel title="Review queue" aside={<span className="text-xs text-textMuted">{cases.length} waiting</span>}>
          {loading ? <Loading>Loading the review queue…</Loading> : cases.length === 0 ? (
            <p className="text-sm text-textSecondary">Nothing waiting. A case opens when a batch lands between suspicious and confirmed.</p>
          ) : (
            <ul className="divide-y divide-borderHairline">
              {cases.map((c) => (
                <li key={c.case_id} className="flex flex-wrap items-center justify-between gap-3 py-3.5">
                  <div className="min-w-0">
                    <div className="font-mono text-sm">{c.case_id.slice(0, 12)}…</div>
                    <div className="text-xs text-textMuted">suspicion {percent(c.suspicion_score)} · {c.n_samples} rows · {when(c.created_at)}</div>
                  </div>
                  <div className="flex gap-2">
                    <Button variant="danger" size="sm" disabled={busy === c.case_id} onClick={() => decide(c.case_id, 'approve_quarantine')}>Quarantine</Button>
                    <Button variant="ghost" size="sm" disabled={busy === c.case_id} onClick={() => decide(c.case_id, 'mark_safe')}>Mark safe</Button>
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
  const [loaded, setLoaded] = useState(false);       // counts stay a dash until one load has succeeded
  const [loadError, setLoadError] = useState(null);

  const load = useCallback(() => Promise.all([
    api.getUsers().then(setUsers),
    api.getAnalysisHistory(100).then((d) => setAnalyses(d.results || [])),
    api.getAuditEvents(100).then((d) => setEvents(d.events || [])),
    api.getPendingReviews().then((d) => setCases(d.cases || [])),
    api.getDefenseStatus().then(setStatus),
  ]).then(() => { setLoaded(true); setLoadError(null); }).catch(setLoadError).finally(() => setLoading(false)), []);

  useEffect(() => { load(); }, [load]);

  const notify = (m) => { setError(''); setMessage(m); };
  const fail = (m) => { setMessage(''); setError(m); };
  const reload = () => { setLoading(true); load(); };
  const counts = loaded ? { accounts: users.length, analyses: analyses.length, audit: events.length, review: cases.length } : {};
  const admins = users.filter((u) => u.role === 'admin').length;

  return (
    <Page>
      <PageHeader title="Admin" accent="console" lede="Manage accounts, read the audit log and every stored analysis, and work the review queue for this admin account.">
        <Button variant="ghost" onClick={reload}><RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} /> Refresh</Button>
      </PageHeader>

      <section aria-label="Totals" className="mb-8 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Accounts" value={counts.accounts ?? '—'} note={loaded ? `${admins} administrator${admins === 1 ? '' : 's'}` : null} />
        <StatCard label="Stored analyses" value={counts.analyses ?? '—'} note="newest 100 across all accounts" />
        <StatCard label="Audit events" value={counts.audit ?? '—'} note="newest 100" />
        <StatCard label="Waiting for review" value={counts.review ?? '—'} note="this admin account" />
      </section>

      {loaded && loadError && <ErrorNote className="mb-6">Refresh failed: {loadError.message}</ErrorNote>}
      {(message || error) && (
        <div className="mb-6">
          <Notice tone="safe" onDismiss={() => setMessage('')}>{message}</Notice>
          <ErrorNote>{error}</ErrorNote>
        </div>
      )}

      {!loaded && loadError ? (
        <ErrorState error={loadError} onRetry={reload} title="The console could not load" />
      ) : <>
      <Tabs
        idBase="admin"
        label="Admin sections"
        className="mb-6"
        value={tab}
        onChange={setTab}
        tabs={TABS.map((t) => ({ ...t, count: counts[t.id] }))}
      />

      <TabPanel idBase="admin" value={tab}>
        {tab === 'accounts' && <AccountsTab users={users} loading={loading} me={me} onChanged={reload} notify={notify} fail={fail} />}
        {tab === 'analyses' && <AnalysesTab rows={analyses} loading={loading} onChanged={reload} notify={notify} fail={fail} />}
        {tab === 'audit' && <AuditTab events={events} loading={loading} users={users} />}
        {tab === 'review' && <ReviewTab cases={cases} status={status} loading={loading} onChanged={reload} notify={notify} fail={fail} />}
      </TabPanel>
      </>}
    </Page>
  );
}
