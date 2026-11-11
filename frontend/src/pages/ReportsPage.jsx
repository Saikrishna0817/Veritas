import { useState } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FileText, Download, ShieldAlert } from 'lucide-react';
import { api } from '../services/api';
import {
  DATA_VERDICTS, PageHeader, Panel, Button, ErrorNote, Segmented, VerdictBadge, Readout, Empty, Loading, Page, downloadJson, EASE, DURATION,
} from '../components/ui';

const SOURCES = [
  { id: 'auto', label: 'Latest analysis', desc: 'The newest CSV upload or reference-dataset run. It never falls back to a sample batch.' },
  { id: 'upload', label: 'My latest upload', desc: 'The last CSV you examined.' },
  { id: 'real_dataset', label: 'Latest reference dataset', desc: 'The last reference dataset you analysed, with its known injected poison.' },
  { id: 'demo', label: 'Latest sample batch', desc: 'The last synthetic sample batch run from the Overview.' },
];
const MISSING = {
  auto: 'Nothing analysed yet. Examine a batch or analyse a reference dataset first.',
  upload: 'No uploaded CSV yet. Examine a batch first.',
  real_dataset: 'No reference dataset analysed yet. Analyse one on Reference datasets first.',
  demo: 'No sample batch yet. Run one from the Overview.',
};

function when(iso) {
  if (!iso) return '—';
  return new Date(/(Z|[+-]\d\d:\d\d)$/.test(iso) ? iso : `${iso}Z`).toLocaleString();   // the backend sends naive UTC
}

function Report({ report }) {
  const summary = report.executive_summary || {};
  const v = DATA_VERDICTS[summary.verdict];
  const info = report.dataset_info || {};
  return (
    <motion.div initial={{ opacity: 0.4, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: DURATION.slow, ease: EASE }} className="space-y-8">
      <section className="edge-card">
        <div className="edge-inner relative grid gap-8 p-8 md:grid-cols-[minmax(0,7fr)_minmax(0,5fr)] md:p-10">
          <div aria-hidden className="pointer-events-none absolute inset-0" style={{ background: 'radial-gradient(45% 90% at 90% 40%, rgba(71,0,214,0.38), transparent 70%)' }} />
          <div className="relative min-w-0">
            <h2 className="font-display text-[clamp(1.8rem,3.2vw,2.6rem)] font-medium leading-tight">{report.title || 'Evidence summary'}</h2>
            <p className="mt-3 text-sm text-textSecondary">
              From {report.data_source || 'the latest analysis'}{info.filename && <> · <span className="font-mono">{info.filename}</span></>}
              {info.n_rows != null && <> · {info.n_rows.toLocaleString()} rows</>}
            </p>
            <p className="mt-1 font-mono text-xs text-textMuted break-all">{report.report_id} · {when(report.generated_at)}</p>
          </div>
          <dl className="relative grid grid-cols-2 gap-6 self-end">
            <div className="col-span-2">
              <dt className="rule-label">Verdict</dt>
              <dd className="font-display mt-2 text-[clamp(2rem,3.6vw,3rem)] font-medium leading-none tracking-[-0.05em]" style={{ color: v?.color }}>
                {v?.label || (summary.verdict ? summary.verdict.replace(/_/g, ' ').toLowerCase() : '—')}
              </dd>
            </div>
            <Readout label="Attack pattern" value={summary.attack_type ? summary.attack_type.replace(/_/g, ' ') : '—'} note="heuristic, research preview" />
            <Readout label="Confidence" value={summary.confidence != null ? `${(summary.confidence * 100).toFixed(1)}%` : '—'} />
          </dl>
        </div>
      </section>

      {report.limitations && (
        <Panel title={<span className="inline-flex items-center gap-2"><ShieldAlert className="h-4 w-4 text-statusWarn" aria-hidden /> What this summary is not</span>}>
          <p className="max-w-[75ch] text-textSecondary">{report.limitations.notice}</p>
          {report.limitations.framework_references && <p className="mt-3 max-w-[75ch] text-sm text-textMuted">{report.limitations.framework_references}</p>}
          {report.limitations.report_reference && <p className="mt-4 text-xs text-textMuted">Reference <span className="font-mono break-all">{report.limitations.report_reference}</span></p>}
        </Panel>
      )}

      <div className="grid gap-8 lg:grid-cols-2">
        {report.layer_scores && (
          <Panel title="Evidence by check">
            <ul className="space-y-5">
              {Object.entries(report.layer_scores).map(([layer, score]) => {
                const pct = Math.round((score || 0) * 100);
                return (
                  <li key={layer}>
                    <div className="flex items-baseline justify-between gap-3 text-sm">
                      <span className="text-textSecondary">{layer.replace(/_/g, ' ')}</span>
                      <span className="font-mono tabular">{pct}%</span>
                    </div>
                    <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-white/[0.06]">
                      <motion.div className="h-full rounded-full bg-violet" initial={{ width: 0 }} animate={{ width: `${pct}%` }} transition={{ duration: DURATION.grow, ease: EASE }} />
                    </div>
                  </li>
                );
              })}
            </ul>
          </Panel>
        )}
        {report.attack_narrative && (
          <Panel title="Narrative" aside={<VerdictBadge verdict={summary.verdict} />}>
            <div className="space-y-2 text-textSecondary">
              {report.attack_narrative.split('\n').filter(Boolean).map((line, i) => <p key={i}>{line}</p>)}
            </div>
          </Panel>
        )}
      </div>

      {report.defense_actions?.length > 0 && (
        <Panel title="Defense actions taken" aside={<span className="text-xs text-textMuted">{report.defense_actions.length}</span>}>
          <ul className="divide-y divide-borderHairline">
            {report.defense_actions.map((a, i) => (
              <li key={i} className="grid gap-1 py-3.5 sm:grid-cols-[minmax(0,1fr)_auto] sm:gap-4">
                <div className="min-w-0">
                  <div className="font-medium">{String(a.action || '').replace(/_/g, ' ')}</div>
                  <div className="text-sm text-textSecondary">{a.reason}</div>
                </div>
                <div className="text-sm text-textMuted sm:text-right">
                  {a.samples_affected != null && <div>{a.samples_affected.toLocaleString()} rows</div>}
                  <div className="font-mono text-xs">{when(a.timestamp)}</div>
                </div>
              </li>
            ))}
          </ul>
        </Panel>
      )}
    </motion.div>
  );
}

export default function ReportsPage() {
  const [source, setSource] = useState('auto');
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const generate = async () => {
    setLoading(true);
    setError('');
    try {
      setReport(await api.generateReport(source));
    } catch (e) {
      // A 404 means there is nothing to summarise for this source; say what to do instead of the raw detail.
      setError(e.status === 404 ? MISSING[source] : e.message);
    } finally {
      setLoading(false);
    }
  };
  const pick = (id) => { setSource(id); setReport(null); setError(''); };
  const current = SOURCES.find((s) => s.id === source);

  return (
    <Page>
      <PageHeader title="Evidence" accent="reports" lede="A draft analyst summary of one analysis, mapped to NIST AI RMF concepts. It is not a certification, an attribution or legal evidence.">
        {report && <Button variant="ghost" onClick={() => downloadJson(report, `veritas_report_${report.report_id?.slice(0, 8) || 'latest'}.json`)}><Download className="h-4 w-4" /> Download JSON</Button>}
        <Button variant="primary" onClick={generate} loading={loading}>{!loading && <FileText className="h-4 w-4" />} {report ? 'Generate again' : 'Generate report'}</Button>
      </PageHeader>

      <div className="mb-8 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <Segmented label="Which analysis to summarise" options={SOURCES} value={source} onChange={pick} />
        <p className="text-sm text-textMuted md:max-w-[46ch] md:text-right">{current.desc}</p>
      </div>

      <ErrorNote className="mb-8">{error}</ErrorNote>
      {loading && !report && <Loading>Putting the report together…</Loading>}
      {!loading && !report && !error && (
        <Empty title="No report yet" action={<Link to="/upload" className="inline-flex min-h-[44px] items-center text-sm text-violet hover:underline">Examine a batch →</Link>}>
          Pick which analysis to summarise, then generate the report. It reads the stored result; nothing is re-run.
        </Empty>
      )}
      {report && <Report report={report} />}
    </Page>
  );
}
