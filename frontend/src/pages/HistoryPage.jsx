import { useEffect, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { RefreshCw, X, Download } from 'lucide-react';
import { api } from '../services/api';
import { DATA_VERDICTS, MODEL_VERDICTS, PageHeader, Button, ErrorNote, downloadJson } from '../components/ui';

const SOURCES = [
  ['', 'All'],
  ['upload', 'Uploads'],
  ['real_dataset', 'Reference datasets'],
  ['demo', 'Sample batches'],
  ['model_scan', 'Model scans'],
];

function verdictOf(v) {
  return DATA_VERDICTS[v] || MODEL_VERDICTS[v] || { label: v || '—', color: 'var(--text-muted)' };
}

function Detail({ row, onClose }) {
  const [full, setFull] = useState(null);
  const [error, setError] = useState('');
  useEffect(() => {
    const get = row.source === 'model_scan' ? api.getModelScan(row.id) : api.getHistoricalResult(row.id);
    get.then(setFull).catch((e) => setError(e.message));
  }, [row]);
  const v = verdictOf(row.verdict);
  const flags = full?.layer_results?.row_flags;
  const scan = full?.scan;
  return (
    <motion.aside
      initial={{ opacity: 0, x: 24 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 24 }}
      transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
      className="panel sticky top-6 h-fit p-6"
      aria-label="Run detail"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="rule-label">{row.source?.replace('_', ' ')}</div>
          <div className="mt-1 break-all font-mono text-sm">{row.filename || row.model_filename}</div>
        </div>
        <button onClick={onClose} aria-label="Close detail" className="rounded-2xl p-1.5 text-textMuted hover:bg-white/[0.04] hover:text-textPrimary"><X className="h-4 w-4" /></button>
      </div>
      <div className="mt-5 font-display text-4xl font-medium leading-none" style={{ color: v.color }}>{v.label}</div>
      {v.blurb && <p className="mt-3 text-sm text-textSecondary">{v.blurb}</p>}
      <ErrorNote>{error}</ErrorNote>
      {!full && !error && <p className="mt-6 text-sm text-textMuted" role="status">Loading the full result…</p>}
      {full && (
        <dl className="mt-6 grid grid-cols-2 gap-4 border-t border-borderHairline pt-5 font-mono text-sm">
          {flags && <div><dt className="rule-label">Flagged rows</dt><dd className="mt-1 text-flag">{flags.flagged_indices_incoming?.length ?? 0}</dd></div>}
          {full.n_samples != null && <div><dt className="rule-label">Rows checked</dt><dd className="mt-1">{full.n_samples.toLocaleString()}</dd></div>}
          {full.layer_results?.label_check && <div><dt className="rule-label">Label evidence</dt><dd className="mt-1">{full.layer_results.label_check.evidence?.toFixed(2)}</dd></div>}
          {full.layer_results?.shift_check && <div><dt className="rule-label">Shift evidence</dt><dd className="mt-1">{full.layer_results.shift_check.evidence?.toFixed(2)}</dd></div>}
          {scan && <div><dt className="rule-label">Format</dt><dd className="mt-1">{scan.format}</dd></div>}
          {scan && <div><dt className="rule-label">Findings</dt><dd className="mt-1">{scan.findings?.length ?? 0}</dd></div>}
        </dl>
      )}
      {full && (
        <Button variant="ghost" className="mt-6 w-full" onClick={() => downloadJson(full, `veritas_${row.source}_${row.id}.json`)}>
          <Download className="h-4 w-4" /> Download JSON
        </Button>
      )}
    </motion.aside>
  );
}

export default function HistoryPage() {
  const [source, setSource] = useState('');
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [selected, setSelected] = useState(null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let live = true;
    api.getHistory(source || null, 100)
      .then((d) => { if (live) { setData(d); setError(''); } })
      .catch((e) => live && setError(e.message));
    return () => { live = false; };
  }, [source, nonce]);

  const results = data?.results || [];
  const stats = data?.stats || {};
  const byVerdict = stats.by_verdict || {};

  return (
    <div className="mx-auto max-w-[1360px] px-5 py-10 md:px-10 md:py-14">
      <PageHeader title="Run" accent="history" lede="Every batch you examined and every file you scanned, newest first. Select a run to see its evidence.">
        <Button variant="ghost" onClick={() => setNonce((n) => n + 1)}><RefreshCw className="h-4 w-4" /> Refresh</Button>
      </PageHeader>

      <div className="mb-8 grid grid-cols-2 gap-px overflow-hidden rounded-2xl border border-borderHairline bg-borderHairline md:grid-cols-5">
        {[
          ['Batches', stats.total_analyses ?? 0, 'var(--text-primary)'],
          ['Model scans', stats.model_scans ?? 0, 'var(--text-primary)'],
          ['Confirmed', byVerdict.CONFIRMED_POISONED ?? 0, DATA_VERDICTS.CONFIRMED_POISONED.color],
          ['Suspicious', byVerdict.SUSPICIOUS ?? 0, DATA_VERDICTS.SUSPICIOUS.color],
          ['Clean', byVerdict.CLEAN ?? 0, DATA_VERDICTS.CLEAN.color],
        ].map(([l, n, c]) => (
          <div key={l} className="bg-bgPanel px-5 py-4">
            <div className="rule-label">{l}</div>
            <div className="mt-1 font-mono text-3xl font-bold" style={{ color: c }}>{n}</div>
          </div>
        ))}
      </div>

      <div className="mb-6 flex flex-wrap gap-2" role="group" aria-label="Filter by source">
        {SOURCES.map(([id, label]) => (
          <button
            key={id || 'all'}
            aria-pressed={source === id}
            onClick={() => { setSource(id); setSelected(null); }}
            className={`rounded-2xl border px-3.5 py-2 text-sm transition-colors ${source === id ? 'border-uv bg-uv/[0.08] text-uv' : 'border-borderHairline text-textSecondary hover:border-borderStrong hover:text-textPrimary'}`}
          >
            {label}
          </button>
        ))}
      </div>

      <ErrorNote>{error}</ErrorNote>

      <div className={`grid gap-6 ${selected ? 'lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]' : ''}`}>
        <div className="panel overflow-x-auto">
          <table className="w-full min-w-[640px] text-sm">
            <thead>
              <tr className="rule-label border-b border-borderHairline text-left">
                <th className="px-5 py-3 font-normal">File</th>
                <th className="px-3 py-3 font-normal">Source</th>
                <th className="px-3 py-3 font-normal">Verdict</th>
                <th className="px-3 py-3 text-right font-normal">Rows</th>
                <th className="px-5 py-3 text-right font-normal">When</th>
              </tr>
            </thead>
            <tbody>
              {data === null && !error && <tr><td colSpan={5} className="px-5 py-6 text-textMuted">Loading…</td></tr>}
              {data && results.length === 0 && <tr><td colSpan={5} className="px-5 py-6 text-textSecondary">No runs for this filter yet.</td></tr>}
              {results.map((r) => {
                const v = verdictOf(r.verdict);
                const active = selected?.id === r.id && selected?.source === r.source;
                return (
                  <tr
                    key={`${r.source}-${r.id}`}
                    tabIndex={0}
                    onClick={() => setSelected(r)}
                    onKeyDown={(e) => e.key === 'Enter' && setSelected(r)}
                    className={`cursor-pointer border-b border-borderHairline transition-colors last:border-0 ${active ? 'bg-violet/[0.1]' : 'hover:bg-white/[0.02]'}`}
                  >
                    <td className="max-w-[280px] truncate px-5 py-3 font-mono">{r.filename || r.model_filename || r.id}</td>
                    <td className="px-3 py-3 text-textSecondary">{r.source?.replace('_', ' ')}</td>
                    <td className="px-3 py-3"><span className="font-mono text-[11px] font-bold uppercase" style={{ color: v.color }}>{v.short || v.label}</span></td>
                    <td className="px-3 py-3 text-right font-mono text-textSecondary">{r.n_samples?.toLocaleString() ?? '—'}</td>
                    <td className="px-5 py-3 text-right font-mono text-xs text-textMuted">{r.created_at ? new Date(r.created_at).toLocaleString() : '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <AnimatePresence mode="wait">
          {selected && <Detail key={`${selected.source}-${selected.id}`} row={selected} onClose={() => setSelected(null)} />}
        </AnimatePresence>
      </div>
    </div>
  );
}
