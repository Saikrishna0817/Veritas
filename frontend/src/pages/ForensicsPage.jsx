import { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { RefreshCw } from 'lucide-react';
import { api } from '../services/api';
import { DATA_VERDICTS, PageHeader, Panel, Button, Readout, Empty, PreviewTag } from '../components/ui';

const SOURCES = [
  { id: 'auto', label: 'Latest analysis' },
  { id: 'upload', label: 'My latest upload' },
  { id: 'demo', label: 'Latest sample batch' },
];
const EMPTY = {
  auto: 'Nothing analysed yet. Examine a batch or analyse a reference dataset first.',
  upload: 'No uploaded CSV yet. Examine a batch first.',
  demo: 'No sample batch yet. Run one from the Overview.',
};

function Classification({ c }) {
  if (!c) return null;
  const insufficient = c.classification_status === 'insufficient_evidence' || c.attack_type === 'unknown';
  const probs = Object.entries(c.probabilities || {}).sort(([, a], [, b]) => b - a);
  return (
    <Panel title="Which pattern the flagged rows resemble" aside={<PreviewTag />}>
      {insufficient ? (
        <p className="text-textSecondary">{c.description || 'The signals do not point to a specific attack pattern.'}</p>
      ) : (
        <>
          <div className="font-display text-3xl font-medium capitalize tracking-[-0.04em]">{c.attack_type.replace(/_/g, ' ')}</div>
          {c.description && <p className="mt-2 text-textSecondary">{c.description}</p>}
          <dl className="mt-6 grid grid-cols-2 gap-5">
            <Readout label="Heuristic confidence" value={`${Math.round((c.confidence || 0) * 100)}%`} />
            <Readout label="Typical severity" value={c.severity} />
          </dl>
        </>
      )}
      {probs.length > 0 && (
        <div className="mt-6 space-y-3">
          {probs.map(([type, p], i) => (
            <div key={type} className="grid grid-cols-[8.5rem_1fr_3rem] items-center gap-3 text-sm">
              <span className="capitalize text-textSecondary">{type.replace(/_/g, ' ')}</span>
              <div className="h-2.5 overflow-hidden rounded-full bg-white/[0.07]">
                <motion.div className="h-full rounded-full" style={{ background: i === 0 && !insufficient ? 'var(--uv)' : '#4A494B' }} initial={{ width: 0 }} animate={{ width: `${p * 100}%` }} transition={{ duration: 1, delay: i * 0.06, ease: [0.19, 1, 0.22, 1] }} />
              </div>
              <span className="text-right font-mono text-textSecondary">{(p * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      )}
      <p className="mt-6 text-xs text-textMuted">A rule-based reading of the two checks' signals. It is not measured on the benchmark and is not attribution.</p>
    </Panel>
  );
}

export default function ForensicsPage() {
  const [source, setSource] = useState('auto');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [missing, setMissing] = useState(false);

  // Callers set `loading` before calling, so the effect below never sets state synchronously.
  const load = useCallback((src) => api.getLatestForensics(src)
    .then((d) => { setData(d); setMissing(false); })
    .catch(() => { setData(null); setMissing(true); })
    .finally(() => setLoading(false)), []);

  useEffect(() => { load(source); }, [source, load]);
  const pick = (id) => { if (id !== source) { setLoading(true); setSource(id); } };
  const refresh = () => { setLoading(true); load(source); };

  const v = data ? DATA_VERDICTS[data.verdict] : null;
  const info = data?.dataset_info;
  const pattern = data?.injection_pattern;

  return (
    <div className="mx-auto max-w-[1360px] px-5 py-10 md:px-10 md:py-14">
      <PageHeader
        title="Forensics"
        preview
        lede="A heuristic reading of a flagged batch: which attack pattern the flagged rows resemble and how they differ from the rest. Treat it as a lead to check, not a finding."
      >
        <Button variant="ghost" onClick={refresh} loading={loading}><RefreshCw className="h-4 w-4" /> Refresh</Button>
      </PageHeader>

      <div className="mb-8 flex flex-wrap gap-2" role="group" aria-label="Which analysis">
        {SOURCES.map((s) => (
          <button
            key={s.id}
            aria-pressed={source === s.id}
            onClick={() => pick(s.id)}
            className={`rounded-2xl border px-4 py-2 text-sm font-medium transition-colors ${source === s.id ? 'border-uv bg-uv/[0.08] text-uv' : 'border-borderHairline text-textSecondary hover:border-borderStrong hover:text-textPrimary'}`}
          >
            {s.label}
          </button>
        ))}
      </div>

      {loading && !data && <p className="text-sm text-textMuted" role="status">Loading…</p>}
      {!loading && missing && (
        <Empty title="Nothing to read yet" action={source === 'demo'
          ? <Link to="/" className="text-sm text-violet hover:underline">Go to Overview →</Link>
          : <Link to="/upload" className="text-sm text-violet hover:underline">Examine a batch →</Link>}>
          {EMPTY[source]}
        </Empty>
      )}

      {data && (
        <div className="space-y-8">
          <section className="edge-card">
            <div className="edge-inner relative grid gap-6 p-8 md:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] md:p-10">
              <div aria-hidden className="pointer-events-none absolute inset-0" style={{ background: 'radial-gradient(45% 90% at 90% 50%, rgba(71,0,214,0.4), transparent 70%)' }} />
              <div className="relative">
                <div className="rule-label break-all">{info?.filename || data.source}</div>
                <div className="font-display mt-3 text-[clamp(2.4rem,4.6vw,4rem)] font-medium leading-none tracking-[-0.05em]" style={{ color: v?.color }}>{v?.label || data.verdict}</div>
                {v?.blurb && <p className="mt-3 max-w-[46ch] text-textSecondary">{v.blurb}</p>}
              </div>
              <dl className="relative grid grid-cols-2 gap-6 self-end sm:grid-cols-4">
                <Readout label="Flagged rows" value={pattern?.n_poisoned_samples ?? 0} color="var(--flag)" />
                <Readout label="Rows in file" value={info?.n_rows?.toLocaleString() ?? '—'} />
                <Readout label="Features" value={info?.n_features ?? '—'} />
                <Readout label="Source" value={(data.source || '—').replace('_', ' ')} />
              </dl>
            </div>
          </section>

          <div className="grid gap-8 lg:grid-cols-2">
            <Panel title="What stands out" aside={<PreviewTag />}>
              {pattern?.narrative ? (
                <div className="space-y-2 text-textSecondary">
                  {pattern.narrative.split('\n').filter(Boolean).map((line, i) => <p key={i}>{line}</p>)}
                </div>
              ) : <p className="text-textSecondary">No summary for this analysis.</p>}
              {pattern?.sigma_shift != null && (
                <div className="mt-6"><Readout label="Average feature shift of flagged rows" value={pattern.sigma_shift} unit="σ" /></div>
              )}
            </Panel>
            <Classification c={data.attack_classification} />
          </div>
        </div>
      )}
    </div>
  );
}
