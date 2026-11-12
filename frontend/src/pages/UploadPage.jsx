import { useEffect, useMemo, useState, lazy, Suspense } from 'react';
import { motion } from 'framer-motion';
import { Download, RotateCcw, AlertTriangle } from 'lucide-react';
import { useAnalysisRun } from '../hooks/useAnalysisRun';
import JobCard from '../components/JobCard';
import {
  PageHeader, Button, Panel, Readout, Dropzone, ErrorNote, PreviewTag, Field, RadioCards, Table, Badge, Page,
  downloadJson, EASE, DURATION, verdictInfo, orDash,
} from '../components/ui';

const Specimen3D = lazy(() => import('../components/Specimen3D'));

const STAGES = ['Reading schema', 'Splitting off the trusted reference', 'Label check', 'Distribution-shift check', 'Calibrating against clean rows'];
const REF_MODES = [
  { id: 'order', label: 'First 70% of this file', hint: 'Only works if the first rows are known good.' },
  { id: 'file', label: 'A separate trusted CSV', hint: 'Same columns as the batch.' },
  { id: 'column', label: 'Rows before a cutoff', hint: 'A number or date column, e.g. ingested_at < 2026-03-01.' },
];

function StageTicker() {
  const [i, setI] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setI((x) => Math.min(x + 1, STAGES.length - 1)), 1800);
    return () => clearInterval(t);
  }, []);
  return (
    <div className="panel scan-sweep p-5" role="status" aria-live="polite">
      <ol className="space-y-2 font-mono text-sm">
        {STAGES.map((s, k) => (
          <li key={s} className={`flex items-center gap-3 transition-colors ${k < i ? 'text-textMuted' : k === i ? 'text-uv' : 'text-textMuted/50'}`}>
            <span className="w-4">{k < i ? '✓' : k === i ? '›' : ''}</span>{s}
          </li>
        ))}
      </ol>
    </div>
  );
}

function EvidenceBar({ name, check, score }) {
  const reported = check?.evidence != null;
  const ev = reported ? check.evidence : 0;
  const ran = check?.ran !== false;
  const width = Math.min(ev / 8, 1) * 100;
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <span className="font-semibold">{name}</span>
        <span className="font-mono text-sm text-textSecondary">
          {!ran ? 'skipped (one label value)' : !reported ? 'not reported' : <>−log₁₀ p = <span className={check?.alarm ? 'text-flag' : ''}>{ev.toFixed(2)}</span></>}
        </span>
      </div>
      <div className="relative mt-2 h-3 rounded-sm bg-ink-950">
        {[1.3, 2, 4].map((t) => (
          <span key={t} className="absolute inset-y-[-3px] w-px bg-borderStrong" style={{ left: `${(t / 8) * 100}%` }} aria-hidden />
        ))}
        <motion.div
          className="h-full origin-left rounded-sm"
          style={{ width: `${width}%`, background: check?.alarm ? 'var(--flag)' : 'var(--violet)' }}
          initial={{ scaleX: 0 }}
          animate={{ scaleX: 1 }}
          transition={{ duration: DURATION.grow, ease: EASE }}
        />
      </div>
      <div className="relative mt-1 h-4 font-mono text-[10px] text-textMuted">
        {[['0.05', 1.3], ['0.01', 2], ['1e-4', 4]].map(([l, t]) => (
          <span key={l} className="absolute -translate-x-1/2" style={{ left: `${(t / 8) * 100}%` }}>{l}</span>
        ))}
      </div>
      {score != null && <div className="sr-only">Scaled score {Math.round(score * 100)}%</div>}
    </div>
  );
}

function RankedRows({ scores, flagged, offset }) {
  const [showAll, setShowAll] = useState(false);
  const rows = useMemo(() => {
    const flagSet = new Set(flagged);
    return scores.map((s, i) => ({ i, s, f: flagSet.has(i) })).sort((a, b) => b.s - a.s);
  }, [scores, flagged]);
  const visible = rows.slice(0, showAll ? 200 : 25);
  const max = rows[0]?.s || 1;
  return (
    <Panel
      title="Rows to review first"
      aside={<span className="font-mono text-xs text-textMuted">{flagged.length} flagged · {scores.length.toLocaleString()} ranked</span>}
    >
      <p className="-mt-2 mb-4 max-w-[70ch] text-sm text-textSecondary">
        Ranked by row evidence: −log₁₀ of the share of clean reference rows that score at least as high. 2.0 means the row is in the clean top 1%.
        Row numbers are positions in your file.
      </p>
      <Table minWidth={420} className="-mx-3">
          <thead>
            <tr>
              <th>Rank</th>
              <th>File row</th>
              <th>Evidence</th>
              <th className="w-[40%]"><span className="sr-only">Evidence bar</span></th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {visible.map((r, k) => (
              <tr key={r.i}>
                <td className="py-2 text-textMuted">{k + 1}</td>
                <td className="py-2">{(r.i + offset + 1).toLocaleString()}</td>
                <td className={`py-2 ${r.f ? 'text-flag' : 'text-textSecondary'}`}>{r.s.toFixed(2)}{r.f && <Badge tone="flag" dot={false} className="ml-2 font-sans">flagged</Badge>}</td>
                <td className="py-2">
                  <div className="h-1.5 rounded-sm bg-ink-950"><div className="h-full rounded-sm" style={{ width: `${(r.s / max) * 100}%`, background: r.f ? 'var(--flag)' : '#4A494B' }} /></div>
                </td>
              </tr>
            ))}
          </tbody>
      </Table>
      {rows.length > 25 && (
        <button type="button" onClick={() => setShowAll((x) => !x)} className="mt-2 inline-flex min-h-[44px] items-center text-sm text-violet hover:text-[var(--red-bright)]">
          {showAll ? 'Show top 25' : `Show top ${Math.min(rows.length, 200)}`}
        </button>
      )}
    </Panel>
  );
}

function Result({ result, onReset }) {
  const v = verdictInfo(result.verdict);
  const lr = result.layer_results || {};
  const flags = lr.row_flags || {};
  const info = result.dataset_info || {};
  const cls = result.attack_classification;
  const warnings = [...new Set([...(info.warnings || []), ...(result.warnings || [])])];

  return (
    <div className="space-y-8">
      <section className="edge-card"><div className="edge-inner relative grid xl:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <div aria-hidden className="pointer-events-none absolute inset-0" style={{ background: 'radial-gradient(55% 80% at 80% 60%, rgba(71,0,214,0.5), transparent 70%)' }} />
        <div className="relative z-10 p-8 md:p-12">
          <div className="rule-label">{info.filename}</div>
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 1.2, ease: EASE }}
            className="mt-3 font-display text-[clamp(2.4rem,5vw,4.4rem)] font-medium leading-[0.9]"
            style={{ color: v.color }}
          >
            {v.label}
          </motion.div>
          <p className="mt-4 max-w-[46ch] text-textSecondary">{v.blurb}</p>
          <div className="mt-8 grid grid-cols-2 gap-6">
            <Readout label="Batch evidence" value={v.p} />
            <Readout label="Flagged rows" value={(flags.flagged_indices_incoming || []).length} color="var(--flag)" note={`of ${orDash(result.n_samples, (n) => n.toLocaleString())} checked`} />
            <Readout label="Mode" value={result.detection_mode === 'supervised' || info.has_labels ? 'Labelled' : 'Unlabelled'} note={info.label_column ? `label: ${info.label_column}` : 'shift check only'} />
            <Readout label="Time" value={result.elapsed_ms ? (result.elapsed_ms / 1000).toFixed(1) : '—'} unit="s" />
          </div>
          <div className="mt-8 flex flex-wrap gap-3">
            <Button variant="ghost" onClick={onReset}><RotateCcw className="h-4 w-4" /> Examine another batch</Button>
            <Button
              variant="ghost"
              onClick={() => downloadJson(result, `veritas_${(info.filename || 'batch').replace(/\.csv$/i, '')}_${Date.now()}.json`)}
            >
              <Download className="h-4 w-4" /> Download JSON
            </Button>
          </div>
        </div>
        <div className="relative min-h-[360px] border-t border-borderHairline xl:border-l xl:border-t-0">
          <Suspense fallback={null}>
            <Specimen3D className="absolute inset-0" scores={flags.row_scores} flagged={flags.flagged_indices_incoming} />
          </Suspense>
          <span className="pointer-events-none absolute bottom-3 right-4 font-mono text-[11px] text-textMuted">your rows · height = evidence · move the pointer to tilt</span>
        </div>
      </div></section>

      {warnings.length > 0 && (
        <div className="rounded-2xl border border-statusWarn/40 bg-statusWarn/[0.06] p-4 text-sm text-[#FFD9AE]">
          {warnings.map((w) => <div key={w} className="flex gap-2"><AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />{w}</div>)}
        </div>
      )}

      <div className="grid gap-8 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        {flags.row_scores?.length ? (
          <RankedRows scores={flags.row_scores} flagged={flags.flagged_indices_incoming || []} offset={info.reference_split || 0} />
        ) : <div />}
        <div className="space-y-8">
          <Panel title="Two checks, one verdict">
            <div className="space-y-6">
              <EvidenceBar name="Label check" check={lr.label_check} score={result.layer_scores?.label_check} />
              <EvidenceBar name="Distribution-shift check" check={lr.shift_check} score={result.layer_scores?.shift_check} />
            </div>
            <p className="mt-4 text-xs text-textMuted">
              Ticks mark the verdict cut-offs. The shift check joins the row ranking only when its own evidence reaches 3.0
              {lr.shift_check?.in_row_ranking ? ' (it did here).' : ' (it did not here).'}
            </p>
          </Panel>
          <Panel title="Schema">
            <dl className="grid grid-cols-2 gap-4 text-sm">
              {[
                ['Rows in file', info.n_rows?.toLocaleString()],
                ['Features', info.n_features],
                ['Label column', info.label_column || 'none'],
                ['Reference rows', info.reference_split?.toLocaleString()],
              ].map(([k, val]) => (
                <div key={k}><dt className="rule-label">{k}</dt><dd className="mt-1 break-all font-mono">{val ?? '—'}</dd></div>
              ))}
            </dl>
            {info.feature_names?.length > 0 && (
              <div className="mt-4 flex flex-wrap gap-1.5">
                {info.feature_names.slice(0, 40).map((f) => (
                  <span key={f} className="rounded-lg border border-borderHairline bg-ink-950 px-2 py-0.5 font-mono text-[11px] text-textSecondary">{f}</span>
                ))}
                {info.feature_names.length > 40 && <span className="px-2 py-0.5 font-mono text-[11px] text-textMuted">+{info.feature_names.length - 40}</span>}
              </div>
            )}
          </Panel>
          {cls?.attack_type && (
            <Panel title="Pattern guess" aside={<PreviewTag />}>
              <div className="font-display text-2xl font-medium capitalize">{cls.attack_type.replace(/_/g, ' ')}</div>
              {cls.description && <p className="mt-2 text-sm text-textSecondary">{cls.description}</p>}
              <p className="mt-3 text-xs text-textMuted">A heuristic reading of which attack the flagged rows resemble. It is not measured on the benchmark.</p>
            </Panel>
          )}
        </div>
      </div>

      {result.injection_pattern?.narrative && (
        <Panel title="Draft analyst notes" aside={<PreviewTag />}>
          <pre className="whitespace-pre-wrap font-mono text-sm leading-relaxed text-textSecondary">{result.injection_pattern.narrative}</pre>
        </Panel>
      )}
    </div>
  );
}

export default function UploadPage() {
  const [file, setFile] = useState(null);
  const [refMode, setRefMode] = useState('order');
  const [referenceFile, setReferenceFile] = useState(null);
  const [splitColumn, setSplitColumn] = useState('');
  const [splitCutoff, setSplitCutoff] = useState('');
  const { run: start, loading, job, result, error, setError, reset: clearResult } = useAnalysisRun('csv');

  const pick = (setter) => (f) => {
    if (!f) return;
    if (!f.name.toLowerCase().endsWith('.csv')) return setError('Only .csv files are accepted.');
    if (f.size > 200 * 1024 * 1024) return setError('That file is over 200 MB. Split it or sample it first.');
    setError('');
    setter(f);
  };

  const run = async () => {
    if (!file) return;
    if (refMode === 'file' && !referenceFile) return setError('Choose the trusted reference CSV, or switch to another reference option.');
    if (refMode === 'column' && (!splitColumn.trim() || !splitCutoff.trim())) return setError('Enter both the column name and the cutoff value.');
    start(file, {
      referenceFile: refMode === 'file' ? referenceFile : null,
      splitColumn: refMode === 'column' ? splitColumn.trim() : '',
      splitCutoff: refMode === 'column' ? splitCutoff.trim() : '',
    });
  };

  const reset = () => { clearResult(); setFile(null); setReferenceFile(null); };

  return (
    <Page>
      <PageHeader
        title={result ? 'Examination' : 'Examine a'}
        accent={result ? null : 'batch'}
        lede={result ? null : 'Give Veritas a labelled CSV and say which rows you trust. It ranks the rest by how likely their label is wrong or their features were tampered with, then gives one calibrated verdict.'}
      />
      {result ? (
        <Result result={result} onReset={reset} />
      ) : (
        <div className="grid gap-8 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
          <div className="space-y-5">
            <Dropzone
              accept=".csv"
              file={file}
              onFile={pick(setFile)}
              label="Drop the batch CSV"
              hint="Up to 200 MB and 200,000 rows. Files over 10 MB run in the background. A label column is detected automatically; answer-key columns such as is_poisoned are ignored."
            />
            <ErrorNote>{error}</ErrorNote>
            {job ? <JobCard {...job} /> : loading && <StageTicker />}
          </div>

          <fieldset className="panel space-y-5 p-6" disabled={loading}>
            <legend className="sr-only">Trusted reference</legend>
            <div>
              <h2 className="font-display text-lg font-medium">Which rows do you trust?</h2>
              <p className="mt-1 text-sm text-textSecondary">Everything else is checked against these.</p>
            </div>
            <RadioCards name="ref" label="Reference source" hideLabel stacked options={REF_MODES} value={refMode} onChange={setRefMode} />
            {refMode === 'file' && (
              <Dropzone accept=".csv" file={referenceFile} onFile={pick(setReferenceFile)} label="Drop the trusted CSV" />
            )}
            {refMode === 'column' && (
              <div className="grid grid-cols-2 gap-3">
                {[['Column', splitColumn, setSplitColumn, 'ingested_at'], ['Cutoff', splitCutoff, setSplitCutoff, '2026-03-01']].map(([l, val, set, ph]) => (
                  <Field key={l} label={l} mono value={val} onChange={(e) => set(e.target.value)} placeholder={ph} />
                ))}
              </div>
            )}
            <Button variant="uv" onClick={run} loading={loading} disabled={!file} className="w-full py-4 text-[15px]">
              {loading ? 'Examining…' : 'Run poisoning detection'}
            </Button>
          </fieldset>
        </div>
      )}
    </Page>
  );
}
