import { useEffect, useState, lazy, Suspense } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Download, Play, X } from 'lucide-react';
import { api } from '../services/api';
import { DATA_VERDICTS, PageHeader, Button, ErrorNote, Readout } from '../components/ui';

const Specimen3D = lazy(() => import('../components/Specimen3D'));

function Result({ result, onClose }) {
  const v = DATA_VERDICTS[result.verdict] || DATA_VERDICTS.SUSPICIOUS;
  const lr = result.layer_results || {};
  const flags = lr.row_flags || {};
  return (
    <motion.div
      initial={{ height: 0, opacity: 0 }}
      animate={{ height: 'auto', opacity: 1 }}
      exit={{ height: 0, opacity: 0 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      className="overflow-hidden"
    >
      <div className="grid border-t border-borderHairline bg-ink-950/50 md:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <div className="p-6">
          <div className="flex items-start justify-between gap-3">
            <div className="font-display text-4xl font-medium leading-none" style={{ color: v.color }}>{v.label}</div>
            <button onClick={onClose} aria-label="Close result" className="rounded-2xl p-1.5 text-textMuted hover:bg-white/[0.04] hover:text-textPrimary"><X className="h-4 w-4" /></button>
          </div>
          {result.poison_note && <p className="mt-3 text-sm text-statusWarn">Injected for ground truth: {result.poison_note}</p>}
          <div className="mt-6 grid grid-cols-2 gap-5">
            <Readout label="Flagged rows" value={flags.flagged_indices_incoming?.length ?? 0} color="var(--flag)" note={`of ${(result.n_samples ?? 0).toLocaleString()} checked`} />
            <Readout label="Evidence" value={v.p} />
            <Readout label="Label check" value={lr.label_check?.evidence?.toFixed(2) ?? '—'} note="−log₁₀ p" />
            <Readout label="Shift check" value={lr.shift_check?.evidence?.toFixed(2) ?? '—'} note="−log₁₀ p" />
          </div>
        </div>
        <div className="relative min-h-[300px] border-t border-borderHairline md:border-l md:border-t-0">
          <Suspense fallback={null}>
            <Specimen3D className="absolute inset-0" scores={flags.row_scores} flagged={flags.flagged_indices_incoming} />
          </Suspense>
        </div>
      </div>
    </motion.div>
  );
}

export default function RealDatasetsPage() {
  const [catalog, setCatalog] = useState(null);
  const [busy, setBusy] = useState(null);
  const [results, setResults] = useState({});
  const [error, setError] = useState('');

  useEffect(() => {
    api.getRealDatasets().then((d) => setCatalog(d.datasets || [])).catch((e) => { setCatalog([]); setError(e.message); });
  }, []);

  const analyze = async (id) => {
    setBusy(id);
    setError('');
    try {
      const data = await api.analyzeRealDataset(id);
      setResults((r) => ({ ...r, [id]: data }));
    } catch (e) {
      setError(`Could not analyse ${id}: ${e.message}`);
    } finally {
      setBusy(null);
    }
  };

  const download = async (id) => {
    try { await api.downloadRealDataset(id); } catch (e) { setError(`Download failed: ${e.message}`); }
  };

  return (
    <div className="mx-auto max-w-[1360px] px-5 py-10 md:px-10 md:py-14">
      <PageHeader
        title="Reference" accent="datasets"
        lede="Public datasets with a known amount of poison injected, so you can see what Veritas finds when the answer is known. Analyse one, or download the CSV and run it through Examine a batch."
      />
      <ErrorNote>{error}</ErrorNote>
      <div className="panel mt-6 overflow-hidden">
        {catalog === null && <div className="p-6 text-sm text-textMuted" role="status">Loading the catalogue…</div>}
        {catalog?.length === 0 && !error && <div className="p-6 text-sm text-textSecondary">No reference datasets are configured on this server.</div>}
        <ul className="divide-y divide-borderHairline">
          {catalog?.map((d, i) => (
            <motion.li
              key={d.id}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
            >
              <div className="grid items-center gap-4 p-5 md:grid-cols-[minmax(0,1fr)_auto]">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                    <h2 className="font-display text-lg font-medium">{d.name}</h2>
                    <span className="font-mono text-[11px] uppercase tracking-wider text-textMuted">{d.domain}</span>
                  </div>
                  <p className="mt-1 text-sm text-textSecondary">{d.description}</p>
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {d.attack_types?.map((t) => (
                      <span key={t} className="rounded-lg border border-borderHairline px-2 py-0.5 font-mono text-[11px] text-textSecondary">{t.replace(/_/g, ' ')}</span>
                    ))}
                  </div>
                </div>
                <div className="flex gap-2">
                  <Button variant={results[d.id] ? 'ghost' : 'uv'} onClick={() => analyze(d.id)} loading={busy === d.id} disabled={busy && busy !== d.id}>
                    {busy !== d.id && <Play className="h-4 w-4" />} {results[d.id] ? 'Run again' : 'Analyse'}
                  </Button>
                  <Button variant="ghost" onClick={() => download(d.id)} aria-label={`Download ${d.name} as CSV`}><Download className="h-4 w-4" /> CSV</Button>
                </div>
              </div>
              <AnimatePresence>
                {results[d.id] && (
                  <Result result={results[d.id]} onClose={() => setResults((r) => { const n = { ...r }; delete n[d.id]; return n; })} />
                )}
              </AnimatePresence>
            </motion.li>
          ))}
        </ul>
      </div>
    </div>
  );
}
