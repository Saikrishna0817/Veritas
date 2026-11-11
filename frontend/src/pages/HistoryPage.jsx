import { useEffect, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { RefreshCw, X, Download } from 'lucide-react';
import { api } from '../services/api';
import {
  DATA_VERDICTS, MODEL_VERDICTS, PageHeader, Button, IconButton, ErrorNote, ErrorState, Segmented, Table, VerdictBadge, Loading, Page, downloadJson,
} from '../components/ui';

const SOURCES = [
  { id: '', label: 'All' },
  { id: 'upload', label: 'Uploads' },
  { id: 'real_dataset', label: 'Reference datasets' },
  { id: 'demo', label: 'Sample batches' },
  { id: 'model_scan', label: 'Model scans' },
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
        <IconButton icon={X} label="Close detail" onClick={onClose} className="-mr-2 -mt-2" />
      </div>
      <div className="mt-5 font-display text-4xl font-medium leading-none" style={{ color: v.color }}>{v.label}</div>
      {v.blurb && <p className="mt-3 text-sm text-textSecondary">{v.blurb}</p>}
      <ErrorNote>{error}</ErrorNote>
      {!full && !error && <Loading className="mt-6">Loading the full result…</Loading>}
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
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let live = true;
    api.getHistory(source || null, 100)
      .then((d) => { if (live) { setData(d); setError(null); } })
      .catch((e) => live && setError(e));
    return () => { live = false; };
  }, [source, nonce]);

  const results = data?.results || [];
  const stats = data?.stats;
  const byVerdict = stats?.by_verdict || {};
  const count = (n) => (stats ? n ?? 0 : '—');   // a dash until the numbers arrive, never a made-up 0

  return (
    <Page>
      <PageHeader title="Run" accent="history" lede="Every batch you examined and every file you scanned, newest first. Select a run to see its evidence.">
        <Button variant="ghost" onClick={() => setNonce((n) => n + 1)}><RefreshCw className="h-4 w-4" /> Refresh</Button>
      </PageHeader>

      <div className="mb-8 grid grid-cols-2 gap-px overflow-hidden rounded-2xl border border-borderHairline bg-borderHairline md:grid-cols-5">
        {[
          ['Batches', count(stats?.total_analyses), 'var(--text-primary)'],
          ['Model scans', count(stats?.model_scans), 'var(--text-primary)'],
          ['Confirmed', count(byVerdict.CONFIRMED_POISONED), DATA_VERDICTS.CONFIRMED_POISONED.color],
          ['Suspicious', count(byVerdict.SUSPICIOUS), DATA_VERDICTS.SUSPICIOUS.color],
          ['Clean', count(byVerdict.CLEAN), DATA_VERDICTS.CLEAN.color],
        ].map(([l, n, c]) => (
          <div key={l} className="bg-bgPanel px-5 py-4">
            <div className="rule-label">{l}</div>
            <div className="mt-1 font-mono text-3xl font-bold tabular" style={{ color: c }}>{n}</div>
          </div>
        ))}
      </div>

      <Segmented label="Filter by source" className="mb-6" options={SOURCES} value={source} onChange={(id) => { setSource(id); setSelected(null); }} />

      {/* Nothing loaded yet: the failure is the page. A failed refresh keeps the rows and says so above them. */}
      {error && !data && <ErrorState error={error} onRetry={() => { setError(null); setNonce((n) => n + 1); }} />}
      {error && data && <ErrorNote className="mb-6">{error.message}</ErrorNote>}

      {(data || !error) && <div className={`grid gap-6 ${selected ? 'lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]' : ''}`}>
        <div className="panel min-w-0 px-3 py-2">
          <Table minWidth={640}>
            <thead>
              <tr>
                <th>File</th>
                <th>Source</th>
                <th>Verdict</th>
                <th className="num">Rows</th>
                <th className="num">When</th>
              </tr>
            </thead>
            <tbody>
              {data === null && !error && <tr><td colSpan={5} className="py-6"><Loading>Loading runs…</Loading></td></tr>}
              {data && results.length === 0 && <tr><td colSpan={5} className="py-6 text-textSecondary">No runs for this filter yet.</td></tr>}
              {results.map((r) => {
                const active = selected?.id === r.id && selected?.source === r.source;
                return (
                  <tr
                    key={`${r.source}-${r.id}`}
                    tabIndex={0}
                    onClick={() => setSelected(r)}
                    onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), setSelected(r))}
                    aria-selected={active}
                  >
                    <td className="max-w-[280px] truncate font-mono">{r.filename || r.model_filename || r.id}</td>
                    <td className="nowrap text-textSecondary">{r.source?.replace('_', ' ')}</td>
                    <td><VerdictBadge verdict={r.verdict} kind={r.source === 'model_scan' ? 'model' : 'data'} /></td>
                    <td className="num font-mono text-textSecondary">{r.n_samples?.toLocaleString() ?? '—'}</td>
                    <td className="num font-mono text-xs text-textMuted">{r.created_at ? new Date(r.created_at).toLocaleString() : '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </Table>
        </div>
        <AnimatePresence mode="wait">
          {selected && <Detail key={`${selected.source}-${selected.id}`} row={selected} onClose={() => setSelected(null)} />}
        </AnimatePresence>
      </div>}
    </Page>
  );
}
