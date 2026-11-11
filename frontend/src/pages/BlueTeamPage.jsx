import { useCallback, useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { RefreshCw, ChevronDown, Bot, UserRound, ShieldCheck, Search, Siren } from 'lucide-react';
import { api } from '../services/api';
import { PageHeader, Panel, Button, ErrorNote, ErrorState, Notice, PreviewTag, StatCard, Segmented, Loading, Page, orDash, percent, DURATION, EASE } from '../components/ui';

// Threat level is the backend's reading of the latest batch's suspicion (0.15 / 0.35 / 0.65 cut-offs).
const LEVELS = {
  CRITICAL: { label: 'Critical', color: 'var(--status-critical)', note: 'Latest batch is past the confirmed threshold.' },
  ELEVATED: { label: 'Elevated', color: 'var(--status-warn)', note: 'Latest batch is suspicious.' },
  GUARDED: { label: 'Guarded', color: 'var(--status-info)', note: 'Latest batch shows weak evidence.' },
  NOMINAL: { label: 'Nominal', color: 'var(--status-safe)', note: 'Nothing beyond what clean batches show.' },
};
// A level the interface does not know is shown as itself, never as Nominal.
const levelOf = (code) => LEVELS[code] || { label: code ? String(code).toLowerCase() : 'Not reported', color: 'var(--text-muted)', note: 'The server did not report a level the interface recognises.' };
const REFRESH_MS = 15000;

function when(iso, timeOnly = false) {
  if (!iso) return '—';
  const d = new Date(iso);
  return timeOnly ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : d.toLocaleString();
}
// Playbook steps arrive with a leading emoji; the icon system is drawn, so drop it.
const plain = (s) => String(s).replace(/^[^\p{L}\p{N}]+/u, '');

function ReviewQueue({ cases, onDecide, busy }) {
  if (!cases.length) {
    return <p className="text-sm text-textSecondary">Nothing waiting. A case opens when a batch lands between suspicious and confirmed.</p>;
  }
  return (
    <ul className="divide-y divide-borderHairline">
      {cases.map((c) => {
        const ev = c.evidence_summary || {};
        return (
          <li key={c.case_id} className="flex flex-wrap items-center justify-between gap-4 py-4">
            <div className="min-w-0">
              <div className="font-mono text-sm">{c.case_id.slice(0, 12)}…</div>
              <div className="mt-1 text-xs text-textMuted">
                {c.n_samples} rows · suspicion {percent(c.suspicion_score)} · opened {when(c.created_at)}
              </div>
              <div className="mt-1 font-mono text-xs text-textSecondary">
                label −log₁₀ p {orDash(ev.label_evidence, (n) => Number(n).toFixed(2))} · shift −log₁₀ p {orDash(ev.shift_evidence, (n) => Number(n).toFixed(2))}
                {ev.attack_type && ev.attack_type !== 'unknown' && <> · looks like {ev.attack_type.replace(/_/g, ' ')}</>}
              </div>
            </div>
            <div className="flex gap-2">
              <Button variant="danger" className="px-4 py-2" disabled={busy === c.case_id} onClick={() => onDecide(c.case_id, 'approve_quarantine')}>Quarantine rows</Button>
              <Button variant="ghost" className="px-4 py-2" disabled={busy === c.case_id} onClick={() => onDecide(c.case_id, 'mark_safe')}>Mark safe</Button>
            </div>
          </li>
        );
      })}
    </ul>
  );
}

function Incidents({ items }) {
  const [open, setOpen] = useState(null);
  if (!items.length) return <p className="text-sm text-textSecondary">No actions yet. Automatic quarantines and review decisions are listed here.</p>;
  return (
    <ul className="divide-y divide-borderHairline">
      {items.map((inc, i) => {
        const human = inc.type === 'human_decision';
        const expanded = open === i;
        const Icon = human ? UserRound : Bot;
        return (
          <li key={`${inc.timestamp}-${i}`}>
            <button
              type="button"
              aria-expanded={expanded}
              onClick={() => setOpen(expanded ? null : i)}
              className="grid w-full grid-cols-[20px_1fr_auto_16px] items-center gap-3 py-3 text-left text-sm hover:bg-white/[0.02]"
            >
              <Icon className="h-4 w-4 text-textMuted" aria-label={human ? 'Person' : 'Automatic'} />
              <span className="min-w-0 truncate">
                <span className="capitalize">{(inc.action || '').replace(/_/g, ' ')}</span>
                {inc.samples_affected ? <span className="text-textMuted"> · {inc.samples_affected} rows</span> : null}
                {inc.reviewer ? <span className="text-textMuted"> · by {inc.reviewer}</span> : null}
              </span>
              <span className="text-xs text-textMuted">{when(inc.timestamp, true)}</span>
              <ChevronDown className={`h-4 w-4 text-textMuted transition-transform ${expanded ? 'rotate-180' : ''}`} aria-hidden />
            </button>
            {expanded && (
              <div className="mb-3 rounded-2xl bg-white/[0.04] p-4 text-sm text-textSecondary">
                {inc.reason || 'No reason recorded.'}
                <div className="mt-2 text-xs text-textMuted">{when(inc.timestamp)}</div>
              </div>
            )}
          </li>
        );
      })}
    </ul>
  );
}

function Playbooks() {
  const [list, setList] = useState([]);
  const [selected, setSelected] = useState(null);
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => { api.listPlaybooks().then((d) => setList(d.playbooks || [])).catch((e) => setError(e.message)); }, []);

  const pick = async (id) => {
    if (selected === id) { setSelected(null); setDetail(null); return; }
    setSelected(id);
    setDetail(null);
    try { setDetail(await api.getPlaybook(id)); } catch (e) { setError(e.message); }
  };

  return (
    <div>
      <ErrorNote>{error}</ErrorNote>
      {list.length > 0 && <Segmented label="Playbooks" options={list.map((p) => ({ id: p.id, label: p.attack }))} value={selected} onChange={pick} />}
      {!selected && list.length > 0 && <p className="mt-5 text-sm text-textSecondary">Pick an attack pattern to see its response steps.</p>}
      {selected && !detail && !error && <Loading className="mt-5">Loading the playbook…</Loading>}
      {detail && (
        <motion.div initial={{ opacity: 0.4, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: DURATION.base, ease: EASE }} className="mt-6 space-y-6">
          <div>
            <div className="flex flex-wrap items-baseline gap-3">
              <h3 className="text-xl font-medium">{detail.attack}</h3>
              {detail.severity && <span className="rounded-lg bg-white/[0.06] px-2 py-0.5 text-xs capitalize text-textSecondary">{detail.severity} severity</span>}
            </div>
            {detail.description && <p className="mt-2 max-w-[65ch] text-textSecondary">{detail.description}</p>}
          </div>
          {[
            ['Right away', 'immediate_steps', Siren],
            ['Investigate', 'investigation_steps', Search],
            ['Fix for next time', 'remediation', ShieldCheck],
          ].map(([title, key, Icon]) => (detail[key]?.length ? (
            <div key={key}>
              <div className="rule-label flex items-center gap-2"><Icon className="h-3.5 w-3.5" aria-hidden /> {title}</div>
              <ol className="mt-3 space-y-2">
                {detail[key].map((step, i) => (
                  <li key={i} className="grid grid-cols-[24px_1fr] gap-2 text-sm">
                    <span className="font-mono text-textMuted">{i + 1}</span>
                    <span>{plain(step)}</span>
                  </li>
                ))}
              </ol>
            </div>
          ) : null))}
          {detail.regulatory && <p className="text-xs text-textMuted">Framework reference (informational, not compliance): {detail.regulatory}</p>}
        </motion.div>
      )}
    </div>
  );
}

export default function BlueTeamPage() {
  const [status, setStatus] = useState(null);
  const [incidents, setIncidents] = useState([]);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState('');

  const [loadError, setLoadError] = useState(null);
  const [incidentTotal, setIncidentTotal] = useState(null);
  // /blueteam/status answers NOMINAL, CLEAN and 0 when nothing has been analysed, the same as for a clean batch.
  // The history count tells the two apart; if it cannot load, the status is shown as the server sent it.
  const [analysed, setAnalysed] = useState(null);
  // The status embeds only the newest 5 waiting cases; /defense/hitl/pending has the whole queue.
  const [queue, setQueue] = useState(null);
  const load = useCallback(() => Promise.all([
    api.getBlueTeamStatus(),
    api.getBlueTeamIncidents(),
    api.getHistory(null, 1).then((h) => h.stats?.total_analyses ?? null).catch(() => null),
    api.getPendingReviews().then((d) => d.cases || []).catch(() => null),
  ])
    .then(([s, inc, n, q]) => {
      setStatus(s); setIncidents(inc.incidents || []); setIncidentTotal(inc.total ?? null); setAnalysed(n); setQueue(q);
      setError(''); setLoadError(null);
    })
    .catch((e) => { setError(e.message); setLoadError(e); }), []);

  useEffect(() => {
    load();
    const t = setInterval(load, REFRESH_MS);
    return () => clearInterval(t);
  }, [load]);

  const decide = async (caseId, decision) => {
    setBusy(caseId);
    setMessage('');
    try {
      await api.submitReviewDecision(caseId, decision);
      setMessage(decision === 'approve_quarantine' ? 'Case approved: its rows were quarantined.' : 'Case marked safe.');
      await load();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy('');
    }
  };

  const nothingYet = analysed === 0;
  const level = nothingYet
    ? { label: 'No batch yet', color: 'var(--text-muted)', note: 'The threat level reads your latest batch. Examine one, or run a sample batch from the Overview.' }
    : levelOf(status?.threat_level);
  const cases = queue ?? status?.pending_cases ?? [];
  const queueDepth = status?.hitl_queue_depth ?? cases.length;

  return (
    <Page>
      <PageHeader
        title="Response"
        accent="desk"
        preview
        lede="Automatic actions on flagged batches, the human-review queue and response playbooks. Quarantine and review decisions are real; the threat level is a heuristic reading of your latest batch."
      >
        <Button variant="ghost" onClick={load}><RefreshCw className="h-4 w-4" /> Refresh</Button>
      </PageHeader>

      {status && <ErrorNote className="mb-6">{error}</ErrorNote>}
      <Notice tone="safe" className="mb-6">{message}</Notice>

      {!status && error ? (
        <ErrorState error={loadError} onRetry={load} title="The response desk could not load" />
      ) : !status ? (
        <Loading>Loading the response desk…</Loading>
      ) : status && (
        <>
          <section className="edge-card mb-8">
            <div className="edge-inner relative grid gap-6 p-8 md:grid-cols-[1fr_auto] md:items-end md:p-10">
              <div aria-hidden className="pointer-events-none absolute inset-0" style={{ background: 'radial-gradient(50% 90% at 85% 40%, rgba(71,0,214,0.45), transparent 70%)' }} />
              <div className="relative">
                <div className="rule-label">Threat level <PreviewTag className="ml-2" /></div>
                <div className="font-display mt-3 text-[clamp(2.6rem,5vw,4.4rem)] font-medium leading-none tracking-[-0.05em]" style={{ color: level.color }}>{level.label}</div>
                <p className="mt-3 text-textSecondary">{level.note}</p>
              </div>
              <dl className="relative grid grid-cols-3 gap-6 text-sm md:text-right">
                <div><dt className="rule-label">Latest verdict</dt><dd className="mt-1 font-mono">{nothingYet ? '—' : (status.current_verdict || '—').replace(/_/g, ' ').toLowerCase()}</dd></div>
                <div><dt className="rule-label">Suspicion</dt><dd className="mt-1 font-mono">{nothingYet ? '—' : percent(status.suspicion_score)}</dd></div>
                <div><dt className="rule-label">Updated</dt><dd className="mt-1 font-mono">{when(status.updated_at, true)}</dd></div>
              </dl>
            </div>
          </section>

          <section aria-label="Counts" className="mb-8 grid gap-4 sm:grid-cols-3">
            <StatCard label="Rows quarantined" value={orDash(status.total_quarantined)} note="across your batches" />
            <StatCard label="Defense actions" value={orDash(status.n_defense_actions)} note={status.defense_mode ? `defense mode: ${status.defense_mode}` : null} />
            <StatCard label="Waiting for review" value={queueDepth} />
          </section>

          <div className="grid gap-8 lg:grid-cols-2">
            <Panel title="Review queue" aside={<span className="text-xs text-textMuted">{cases.length < queueDepth ? `first ${cases.length} of ${queueDepth}` : `${queueDepth} waiting`}</span>}>
              <ReviewQueue cases={cases} onDecide={decide} busy={busy} />
              {cases.length < queueDepth && <p className="mt-4 text-xs text-textMuted">The full queue could not load, so these are the newest {cases.length}. Refresh to try again.</p>}
            </Panel>
            <Panel title="Actions taken" aside={<span className="text-xs text-textMuted">{incidentTotal > incidents.length ? `newest ${incidents.length} of ${incidentTotal}` : 'newest first'}</span>}>
              <div className="max-h-[420px] overflow-y-auto pr-1"><Incidents items={incidents} /></div>
            </Panel>
          </div>

          <Panel title="Response playbooks" className="mt-8">
            <Playbooks />
          </Panel>
        </>
      )}
    </Page>
  );
}
