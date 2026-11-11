import { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { Download, RotateCcw, FileCode2 } from 'lucide-react';
import { api } from '../services/api';
import { useAnalysisRun } from '../hooks/useAnalysisRun';
import JobCard from '../components/JobCard';
import { MODEL_VERDICTS, PageHeader, Button, Panel, Dropzone, ErrorNote, Badge, VerdictBadge, Page, downloadJson, formatBytes } from '../components/ui';

const EXTENSIONS = ['.pkl', '.pickle', '.joblib', '.pt', '.pth', '.bin', '.ckpt', '.safetensors', '.onnx', '.npy', '.npz', '.gguf', '.h5', '.keras', '.zip'];
const MAX_MB = 200;
const SEVERITY = { critical: 'var(--status-critical)', high: 'var(--status-critical)', medium: 'var(--status-warn)', low: 'var(--status-info)', info: 'var(--text-muted)' };

/** A tape of the file's first bytes scrolling past the read head while the sandbox scans it. */
function ByteTape({ file, active }) {
  const [hex, setHex] = useState([]);
  useEffect(() => {
    if (!file) return;
    file.slice(0, 768).arrayBuffer().then((b) => setHex([...new Uint8Array(b)].map((x) => x.toString(16).padStart(2, '0'))));
  }, [file]);
  const lines = useMemo(() => {
    const out = [];
    for (let i = 0; i < hex.length; i += 16) out.push({ off: i, bytes: hex.slice(i, i + 16) });
    return out;
  }, [hex]);
  if (!file || !hex.length) return null;
  return (
    <div className={`panel relative h-[260px] overflow-hidden ${active ? 'scan-sweep' : ''}`} aria-hidden>
      <div className="absolute inset-x-0 top-1/2 z-10 h-px bg-uv/70 shadow-uv" />
      <motion.div
        className="p-5 font-mono text-[12px] leading-6 text-textMuted"
        animate={active ? { y: [0, -lines.length * 24 + 200] } : { y: 0 }}
        transition={active ? { duration: Math.max(lines.length * 0.25, 4), repeat: Infinity, ease: 'linear' } : { duration: 0.6 }}
      >
        {lines.map((l) => (
          <div key={l.off} className="flex gap-4 whitespace-nowrap">
            <span className="text-violet/70">{l.off.toString(16).padStart(6, '0')}</span>
            <span>{l.bytes.join(' ')}</span>
          </div>
        ))}
      </motion.div>
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(180deg,var(--bg-panel),transparent_25%,transparent_75%,var(--bg-panel))]" />
    </div>
  );
}

function Result({ result, onReset }) {
  const scan = result.scan || {};
  const v = MODEL_VERDICTS[result.verdict] || MODEL_VERDICTS.REVIEW;
  const findings = scan.findings || [];
  return (
    <div className="space-y-8">
      <section className="edge-card"><div className="edge-inner relative p-8 md:p-12">
        <div aria-hidden className="pointer-events-none absolute -right-20 -top-20 h-72 w-72 rounded-full blur-3xl" style={{ background: v.color, opacity: 0.12 }} />
        <div className="rule-label break-all">{result.model_filename}</div>
        <motion.div
          initial={{ opacity: 0, clipPath: 'inset(0 100% 0 0)' }}
          animate={{ opacity: 1, clipPath: 'inset(0 0% 0 0)' }}
          transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
          className="mt-3 font-display text-[clamp(2.2rem,5vw,4.4rem)] font-medium leading-[0.92]"
          style={{ color: v.color }}
        >
          {v.label}
        </motion.div>
        <p className="mt-4 max-w-[56ch] text-lg text-textSecondary">{v.blurb}</p>
        <dl className="mt-8 grid grid-cols-2 gap-6 border-t border-borderHairline pt-6 font-mono text-sm md:grid-cols-4">
          <div><dt className="rule-label">Format</dt><dd className="mt-1">{scan.format || '—'}</dd></div>
          <div><dt className="rule-label">Size</dt><dd className="mt-1">{formatBytes(scan.size_bytes)}</dd></div>
          <div><dt className="rule-label">Findings</dt><dd className="mt-1">{findings.length}</dd></div>
          <div className="min-w-0"><dt className="rule-label">SHA-256</dt><dd className="mt-1 truncate" title={scan.sha256}>{scan.sha256 || '—'}</dd></div>
        </dl>
        <div className="mt-8 flex flex-wrap gap-3">
          <Button variant="ghost" onClick={onReset}><RotateCcw className="h-4 w-4" /> Scan another file</Button>
          <Button variant="ghost" onClick={() => downloadJson(result, `veritas_scan_${result.scan_id || Date.now()}.json`)}><Download className="h-4 w-4" /> Download JSON</Button>
        </div>
      </div></section>

      <div className="grid gap-8 lg:grid-cols-2">
        <Panel title="Findings" aside={<span className="font-mono text-xs text-textMuted">{findings.length}</span>}>
          {findings.length === 0 ? (
            <p className="text-sm text-textSecondary">Nothing flagged.</p>
          ) : (
            <ul className="space-y-3">
              {findings.map((f, i) => (
                <motion.li key={i} initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.05 * i }} className="grid grid-cols-[84px_1fr] items-start gap-3 text-sm">
                  <Badge color={SEVERITY[f.severity] || 'var(--text-muted)'} className="justify-self-start">{f.severity}</Badge>
                  <span className="break-words text-textSecondary">{f.detail}</span>
                </motion.li>
              ))}
            </ul>
          )}
        </Panel>
        <Panel title="Imports the file would perform" aside={<span className="font-mono text-xs text-textMuted">{(scan.imports || []).length}</span>}>
          {(scan.imports || []).length === 0 ? (
            <p className="text-sm text-textSecondary">None. The file declares no Python imports.</p>
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {scan.imports.map((imp) => {
                const name = Array.isArray(imp) ? imp.join('.') : String(imp);
                return <span key={name} className="rounded-lg border border-borderHairline bg-ink-950 px-2 py-1 font-mono text-xs text-textSecondary">{name}</span>;
              })}
            </div>
          )}
          {result.model_metadata && <p className="mt-4 font-mono text-sm text-textSecondary">scikit-learn model: {result.model_metadata.model_type}</p>}
        </Panel>
      </div>
      <p className="max-w-[80ch] text-sm text-textMuted">{result.interpretation}</p>
    </div>
  );
}

export default function ModelScanPage() {
  const [file, setFile] = useState(null);
  const { run: start, loading, job, result, error, setError, reset: clearResult } = useAnalysisRun('model');
  const [history, setHistory] = useState([]);

  const loadHistory = () => api.getModelScanHistory(8).then((d) => setHistory(d.results || d.scans || (Array.isArray(d) ? d : []))).catch(() => {});
  useEffect(() => { loadHistory(); }, []);

  const pick = (f) => {
    if (!f) return;
    if (!EXTENSIONS.some((ext) => f.name.toLowerCase().endsWith(ext))) return setError(`Unsupported file type. Supported: ${EXTENSIONS.join(' ')}`);
    if (f.size > MAX_MB * 1024 * 1024) return setError(`That file is over ${MAX_MB} MB.`);
    setError('');
    setFile(f);
  };

  const run = () => start(file).then(loadHistory);

  return (
    <Page>
      <PageHeader
        title={result ? 'Scan result' : 'Scan a model'}
        accent={result ? null : 'file'}
        lede={result ? null : 'Find out whether loading a model file could run code. The file is read as bytes in an isolated process; it is never loaded or executed.'}
      />
      {result ? (
        <Result result={result} onReset={() => { clearResult(); setFile(null); }} />
      ) : (
        <div className="grid gap-8 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
          <div className="space-y-5">
            <Dropzone accept={EXTENSIONS.join(',')} file={file} onFile={pick} icon={FileCode2} label="Drop a model file" hint={`pickle, joblib, PyTorch, safetensors, ONNX, NumPy, GGUF, Keras or zip · up to ${MAX_MB} MB; over 10 MB runs in the background`} />
            {job ? <JobCard {...job} /> : <ByteTape file={file} active={loading} />}
            <ErrorNote>{error}</ErrorNote>
            <Button variant="uv" onClick={run} loading={loading} disabled={!file} className="w-full py-4 text-[15px]">
              {loading ? 'Reading the opcode stream…' : 'Scan the file'}
            </Button>
          </div>
          <div className="space-y-8">
            <Panel title="What the verdicts mean">
              <ul className="space-y-4">
                {Object.entries(MODEL_VERDICTS).map(([k, val]) => (
                  <li key={k} className="grid grid-cols-[10px_1fr] gap-3">
                    <span className="mt-1.5 h-2.5 w-2.5 rounded-full" style={{ background: val.color }} />
                    <span><span className="font-semibold" style={{ color: val.color }}>{val.label}.</span> <span className="text-sm text-textSecondary">{val.blurb}</span></span>
                  </li>
                ))}
              </ul>
            </Panel>
            <Panel title="Recent scans">
              {history.length === 0 ? (
                <p className="text-sm text-textSecondary">No scans yet.</p>
              ) : (
                <ul className="divide-y divide-borderHairline">
                  {history.map((h) => {
                    return (
                      <li key={h.id} className="flex items-center justify-between gap-3 py-2.5 text-sm">
                        <span className="truncate font-mono">{h.model_filename}</span>
                        <VerdictBadge verdict={h.verdict} kind="model" />
                      </li>
                    );
                  })}
                </ul>
              )}
            </Panel>
          </div>
        </div>
      )}
    </Page>
  );
}
