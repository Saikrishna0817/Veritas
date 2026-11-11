import { useEffect, useState, lazy, Suspense } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowRight, FileCode2, Play, Radio, ScanLine, TriangleAlert, Gauge } from 'lucide-react';
import { api } from '../services/api';
import { DATA_VERDICTS, Button, Reveal, ErrorNote, StatCard, VerdictBadge, Loading } from '../components/ui';

// three.js is heavy; the specimen loads after the readout is on screen.
const Specimen3D = lazy(() => import('../components/Specimen3D'));

const BENCH = [
  { label: 'Row AUPRC at 5% poison', note: 'random ranking = 0.05', values: { 'Old Veritas': 0.11, veritas2: 0.69, cleanlab: 0.54, 'Two-sample test': 0.49 } },
  { label: 'Batches caught at 5% false alarms', note: '12 held-out OpenML datasets', values: { 'Old Veritas': 0.33, veritas2: 0.78, cleanlab: 0.55, 'Two-sample test': 0.14 } },
];

const EVENT_COPY = {
  sample_analyzed: (d) => `Batch scored · ${d?.n_samples ?? '?'} rows · suspicion ${pct(d?.suspicion_score)}`,
  attack_confirmed: (d) => `Evidence above threshold${d?.attack_type ? ` · pattern: ${d.attack_type.replace(/_/g, ' ')}` : ''}`,
  defense_triggered: (d) => `${(d?.action || 'action').replace(/_/g, ' ')} · ${d?.samples_affected ?? 0} rows held back`,
  human_review_required: () => 'Case opened for human review',
};

function pct(v) {
  return v == null ? '—' : `${Math.round(v * 100)}%`;
}

function timeAgo(iso) {
  if (!iso) return '';
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(iso).toLocaleDateString();
}

function VerdictReadout({ latest }) {
  if (!latest) {
    return (
      <div>
        <div className="rule-label">No batch examined yet</div>
        <motion.p
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
          className="mt-4 font-display text-[clamp(2.2rem,4.2vw,3.9rem)] font-medium leading-[0.92]"
        >
          Put the batch on the slide.<br /><span className="serif-accent">Watch what glows.</span>
        </motion.p>
        <p className="mt-5 max-w-[46ch] text-textSecondary">
          The specimen here is illustrative: 2,400 synthetic rows with a 5% poisoned tail. Examine your own CSV and it is replaced by your rows, each one lifted by its evidence.
        </p>
      </div>
    );
  }
  const v = DATA_VERDICTS[latest.verdict] || DATA_VERDICTS.SUSPICIOUS;
  const flags = latest.layer_results?.row_flags;
  return (
    <div>
      <div className="rule-label">Latest batch · {latest.dataset_info?.filename || latest.source}</div>
      <motion.div
        initial={{ opacity: 0, letterSpacing: '0.08em' }}
        animate={{ opacity: 1, letterSpacing: '-0.02em' }}
        transition={{ duration: 1.1, ease: [0.16, 1, 0.3, 1] }}
        className="mt-3 font-display text-[clamp(2.6rem,6vw,5.4rem)] font-medium leading-[0.9]"
        style={{ color: v.color }}
      >
        {v.label}
      </motion.div>
      <p className="mt-4 max-w-[46ch] text-textSecondary">{v.blurb}</p>
      <dl className="mt-6 grid grid-cols-3 gap-6 border-t border-borderHairline pt-5 font-mono">
        <div><dt className="rule-label">Evidence</dt><dd className="mt-1 whitespace-nowrap text-lg">{v.p}</dd></div>
        <div><dt className="rule-label">Flagged rows</dt><dd className="mt-1 text-lg text-flag">{flags?.flagged_indices_incoming?.length ?? 0}</dd></div>
        <div><dt className="rule-label">Rows checked</dt><dd className="mt-1 text-lg">{(latest.n_samples ?? 0).toLocaleString()}</dd></div>
      </dl>
    </div>
  );
}

function BenchLedger() {
  const names = Object.keys(BENCH[0].values);
  return (
    <div className="panel overflow-hidden">
      <div className="flex flex-col gap-2 border-b border-borderHairline p-6 md:flex-row md:items-end md:justify-between">
        <div>
          <h2 className="font-display text-2xl font-medium">Measured, on data it never saw</h2>
          <p className="mt-2 max-w-[62ch] text-sm text-textSecondary">
            Benchmark v2: 18 real OpenML datasets, six injected attacks. Tuned on 6, reported on the 12 held out. On real human label noise (CIFAR-10N) it matches cleanlab rather than beating it.
          </p>
        </div>
        <span className="font-mono text-[11px] text-textMuted">docs/model-card.md</span>
      </div>
      <div className="divide-y divide-borderHairline">
        {BENCH.map((row, ri) => (
          <div key={row.label} className="grid gap-4 p-6 md:grid-cols-[220px_1fr]">
            <div>
              <div className="font-semibold">{row.label}</div>
              <div className="mt-1 text-xs text-textMuted">{row.note}</div>
            </div>
            <div className="space-y-2.5">
              {names.map((n, i) => {
                const val = row.values[n];
                const ours = n === 'veritas2';
                return (
                  <div key={n} className="grid grid-cols-[120px_1fr_44px] items-center gap-3 text-sm">
                    <span className={ours ? 'font-semibold text-textPrimary' : 'text-textMuted'}>{n}</span>
                    <div className="h-2.5 overflow-hidden rounded-sm bg-ink-950">
                      <motion.div
                        className="h-full rounded-sm"
                        style={{ background: ours ? 'var(--uv)' : '#4A494B' }}
                        initial={{ width: 0 }}
                        whileInView={{ width: `${val * 100}%` }}
                        viewport={{ once: true }}
                        transition={{ duration: 1.1, delay: ri * 0.15 + i * 0.08, ease: [0.16, 1, 0.3, 1] }}
                      />
                    </div>
                    <span className={`text-right font-mono ${ours ? 'text-uv' : 'text-textSecondary'}`}>{val.toFixed(2)}</span>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}


/** Stat cards in the style of the reference's dashboard: label, round icon chip, big number with a small unit. */
function StatRow({ stats, usage }) {
  const verdicts = stats?.by_verdict || {};
  const flaggedBatches = (verdicts.SUSPICIOUS || 0) + (verdicts.CONFIRMED_POISONED || 0);
  const used = usage?.used || {};
  const limits = usage?.limits || {};
  const pct = (k) => (limits[k] ? Math.round(((used[k] || 0) / limits[k]) * 100) : null);
  const loading = stats === undefined;
  const dash = '—';
  return (
    <section aria-label="Your numbers" className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <StatCard label="Batches examined" icon={ScanLine} value={loading ? dash : (stats.total_analyses ?? 0)} note="uploads, reference datasets and sample batches" />
      <StatCard
        label="Batches flagged"
        icon={TriangleAlert}
        value={loading ? dash : flaggedBatches}
        note="suspicious or confirmed"
        accent={!loading && stats.total_analyses ? `${Math.round((flaggedBatches / stats.total_analyses) * 100)}%` : null}
      />
      <StatCard label="Model files scanned" icon={FileCode2} value={loading ? dash : (stats.model_scans ?? 0)} />
      <StatCard
        label="Daily quota used"
        icon={Gauge}
        value={usage ? (pct('analyses') ?? dash) : dash}
        unit={usage && pct('analyses') !== null ? '%' : null}
        note={usage ? `${used.analyses || 0} of ${limits.analyses || 'unlimited'} analyses today, UTC` : null}
      />
    </section>
  );
}

export default function Dashboard({ wsEvents = [] }) {
  const navigate = useNavigate();
  const [latest, setLatest] = useState(null);
  const [history, setHistory] = useState(null);
  const [usage, setUsage] = useState(null);
  const [demoBusy, setDemoBusy] = useState(false);
  const [error, setError] = useState('');

  const load = () => {
    api.getLatestUpload().then(setLatest).catch(() => setLatest(null));
    api.getHistory(null, 8).then(setHistory).catch(() => setHistory({ results: [], stats: {} }));
    api.getUsage().then(setUsage).catch(() => setUsage(null));
  };
  useEffect(load, []);

  const runDemo = async () => {
    setDemoBusy(true);
    setError('');
    try {
      await api.runDemo();
      load();
    } catch (e) {
      setError(e.message || 'The sample batch could not be run.');
    } finally {
      setDemoBusy(false);
    }
  };

  const flags = latest?.layer_results?.row_flags;
  const stats = history?.stats || {};

  return (
    <div className="mx-auto max-w-[1360px] px-5 md:px-10">
      {/* Specimen hero */}
      <section className="relative grid items-center gap-6 py-10 xl:min-h-screen xl:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] xl:py-0">
        <div className="relative z-10">
          <h1 className="sr-only">Veritas overview</h1>
          <VerdictReadout latest={latest} />
          <div className="mt-10 flex flex-wrap gap-3">
            <Button onClick={() => navigate('/upload')} variant="uv" className="px-6 py-3.5 text-[15px]">
              Examine a batch <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
            </Button>
            <Button onClick={() => navigate('/model-scan')} variant="ghost" className="px-6 py-3.5 text-[15px]">
              <FileCode2 className="h-4 w-4" /> Scan a model file
            </Button>
          </div>
          <button
            onClick={runDemo}
            disabled={demoBusy}
            className="mt-3 inline-flex min-h-[44px] items-center gap-2 text-sm text-textMuted underline-offset-4 hover:text-textPrimary hover:underline disabled:opacity-50"
          >
            <Play className="h-3.5 w-3.5" /> {demoBusy ? 'Running a synthetic sample batch…' : 'Or run a synthetic sample batch'}
          </button>
          {error && <div className="mt-4"><ErrorNote>{error}</ErrorNote></div>}
        </div>

        <div className="relative h-[46vh] min-h-[300px] xl:h-[82vh]">
          <div aria-hidden className="pointer-events-none absolute inset-0" style={{ background: 'radial-gradient(50% 50% at 50% 55%, rgba(123,77,255,0.45), rgba(71,0,214,0.2) 45%, transparent 72%)' }} />
          <Suspense fallback={<div className="h-full w-full animate-pulse rounded-2xl bg-violet/[0.04]" />}>
            <Specimen3D
              className="absolute inset-0"
              scores={flags?.row_scores}
              flagged={flags?.flagged_indices_incoming}
              loop
            />
          </Suspense>
          <div className="pointer-events-none absolute bottom-3 left-0 right-0 flex flex-wrap items-center justify-center gap-5 font-mono text-[11px] text-textMuted">
            <span className="flex items-center gap-2"><span className="h-2 w-2 rounded-full bg-[#B9A3FF]" /> row, height = evidence</span>
            <span className="flex items-center gap-2"><span className="h-2 w-2 rounded-full bg-flag" /> flagged row</span>
            <span>{flags?.row_scores?.length ? `${flags.row_scores.length.toLocaleString()} real rows` : 'illustrative specimen'}</span>
          </div>
        </div>
      </section>

      <Reveal><StatRow stats={history ? history.stats || {} : undefined} usage={usage} /></Reveal>

      <Reveal className="mt-16"><BenchLedger /></Reveal>

      {/* Recent work + live tape */}
      <section className="mt-16 grid gap-6 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <Reveal>
          <div className="panel h-full">
            <div className="flex items-baseline justify-between border-b border-borderHairline p-6">
              <h2 className="font-display text-xl font-medium">Recent runs</h2>
              <Link to="/history" className="-my-3 inline-flex min-h-[44px] items-center text-sm text-violet hover:text-[var(--red-bright)]">All history →</Link>
            </div>
            {history === null ? (
              <Loading className="p-6">Loading recent runs…</Loading>
            ) : history.results?.length ? (
              <ul className="divide-y divide-borderHairline">
                {history.results.map((r) => {
                  return (
                    <li key={`${r.source}-${r.id}`} className="grid grid-cols-[1fr_auto] items-center gap-4 px-6 py-4">
                      <div className="min-w-0">
                        <div className="truncate font-mono text-sm">{r.filename || r.model_filename || r.id}</div>
                        <div className="mt-0.5 text-xs text-textMuted">{r.source?.replace('_', ' ')} · {timeAgo(r.created_at)}</div>
                      </div>
                      <VerdictBadge verdict={r.verdict} kind={r.source === 'model_scan' ? 'model' : 'data'} />
                    </li>
                  );
                })}
              </ul>
            ) : (
              <div className="p-6 text-sm text-textSecondary">Nothing yet. Examine a batch or scan a model file and it lands here.</div>
            )}
            {history && (
              <div className="flex gap-8 border-t border-borderHairline px-6 py-4 font-mono text-xs text-textMuted">
                <span>{stats.total_analyses ?? 0} batches</span>
                <span>{stats.model_scans ?? 0} model scans</span>
              </div>
            )}
          </div>
        </Reveal>

        <Reveal delay={0.1}>
          <div className="panel h-full">
            <div className="flex items-center gap-2 border-b border-borderHairline p-6">
              <Radio className="h-4 w-4 text-uv" aria-hidden />
              <h2 className="font-display text-xl font-medium">Live events</h2>
            </div>
            <ol className="max-h-[360px] space-y-0 overflow-y-auto p-2" aria-live="polite">
              {wsEvents.length === 0 && <li className="p-4 text-sm text-textMuted">Quiet. Events from your analyses stream here as they happen.</li>}
              {wsEvents.map((e, i) => (
                <motion.li
                  key={`${e.receivedAt}-${i}`}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="grid grid-cols-[64px_1fr] gap-3 rounded-lg px-4 py-2.5 text-sm hover:bg-white/[0.02]"
                >
                  <time className="font-mono text-[11px] text-textMuted">{new Date(e.receivedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</time>
                  <span className={e.event === 'attack_confirmed' ? 'text-flag' : 'text-textSecondary'}>
                    {(EVENT_COPY[e.event] || (() => (e.event || 'event').replace(/_/g, ' ')))(e.data)}
                  </span>
                </motion.li>
              ))}
            </ol>
          </div>
        </Reveal>
      </section>
    </div>
  );
}
