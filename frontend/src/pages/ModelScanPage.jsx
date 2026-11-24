import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { api } from '../services/api';
import { followJob, startDirect } from '../services/uploads';
import { MODEL_VERDICTS as VERDICTS } from '../components/results';
import { Skeleton } from '../components/Skeleton';
import { LoadError } from '../components/LoadError';
import { useLoad } from '../hooks/useLoad';
import { useMore } from '../hooks/useMore';
import { LoadMoreButton } from '../components/LoadMoreButton';
import { PageIntro, PageTitle } from '../ui/Reveal';
import { GLYPHS } from '../../shared/argilla/keys/glyphs';
import { MAX_KEYS, setHot, useStage, useStageState } from '../scene/stage';

const EXTENSIONS = ['.pkl', '.pickle', '.joblib', '.pt', '.pth', '.bin', '.ckpt', '.safetensors', '.onnx', '.npy', '.npz', '.gguf', '.h5', '.keras', '.zip'];
const MAX_MB = 200;

// What the scan says about each import, from its findings: a critical dangerous_import or a medium unknown_import
// names the module and attribute; every other import matched the known data containers.
const STATUS = {
  dangerous: { words: 'can run code or reach the system', v: 'DANGEROUS', tint: '--ev-3', lift: 2.6 },
  unknown: { words: 'not on the known-safe list', v: 'REVIEW', tint: '--ev-2', lift: 1.5 },
  safe: { words: 'a known data container', v: '', tint: '--glaze-cream', lift: 0.4 },
};
const ORDER = { dangerous: 0, unknown: 1, safe: 2 };

function importsWithStatus(scan) {
  const flagged = new Map();
  for (const f of scan.findings || []) {
    if (f.module == null || f.name == null) continue;
    const key = `${f.module}.${f.name}`;
    if (f.kind === 'dangerous_import') flagged.set(key, 'dangerous');
    else if (f.kind === 'unknown_import' && flagged.get(key) !== 'dangerous') flagged.set(key, 'unknown');
  }
  return (scan.imports || [])
    .map(([m, n]) => ({ name: `${m}.${n}`, status: flagged.get(`${m}.${n}`) || 'safe' }))
    .sort((a, b) => ORDER[a.status] - ORDER[b.status]);
}

const legend = (s) => {
  const i = GLYPHS.indexOf(String(s).charAt(0).toUpperCase());
  return i < 0 ? GLYPHS.indexOf('*') : i;
};

// The stage beside the imports: one key per import (at most 24), glazed and lifted by what the scan says about it.
function ImportKeys({ imports }) {
  const ref = useRef(null);
  const items = useMemo(() => (imports.length
    ? imports.slice(0, MAX_KEYS).map((im) => ({ glyph: legend(im.name), tint: STATUS[im.status].tint, lift: STATUS[im.status].lift }))
    : null), [imports]);
  useStage('keys', ref, { items });
  return <div className="in-keys-stage" ref={ref} aria-hidden="true" />;
}

function ScanResult({ result, actions }) {
  const scan = result.scan || {};
  const { hot } = useStageState();
  // An unknown code is shown as it is, never as another verdict.
  const [title, note] = VERDICTS[result.verdict] || [result.verdict ? `Verdict: ${String(result.verdict).replace(/_/g, ' ').toLowerCase()}` : 'No verdict in this result', 'This verdict is not one the interface recognises. Download the scan to read it in full.'];
  const findings = scan.findings || [];
  const imports = useMemo(() => importsWithStatus(result.scan || {}), [result.scan]);
  const saveJson = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' }));
    const a = document.createElement('a');
    a.href = url; a.download = `veritas-model-scan-${result.scan_id || Date.now()}.json`; a.click();
    URL.revokeObjectURL(url);
  };
  return (
    // evidence before verdict (docs/PRODUCT.md, principle 1): what the file would import and what was found, then the conclusion
    <div className="in-result">
      {imports.length > 0 && (
        <section className="in-section">
          <div className="in-review-head">
            <div>
              <h2 className="in-h2">Imports the file would perform</h2>
              <p className="in-sub">What loading the file would import, each checked against known data containers.</p>
            </div>
            <ImportKeys imports={imports} />
          </div>
          <ul className="in-imports">
            {imports.map((im, i) => (
              <li key={im.name} data-hot={hot === i || undefined}
                {...(i < MAX_KEYS ? { onMouseEnter: () => setHot(i), onMouseLeave: () => setHot(-1) } : {})}>
                <span className="in-path">{im.name}</span>
                <span className="in-chip" data-v={STATUS[im.status].v}>{STATUS[im.status].words}</span>
              </li>
            ))}
          </ul>
        </section>
      )}
      <section className="in-section">
        <h2 className="in-h2">Findings</h2>
        {findings.length === 0 ? <p className="in-empty">Nothing flagged.</p> : (
          <div className="in-table-wrap">
            <table className="in-table in-table-stack">
              <thead><tr><th>Severity</th><th>Detail</th></tr></thead>
              <tbody>{findings.map((f, i) => <tr key={i}><td className="in-tag" data-label="Severity">{f.severity}</td><td data-label="Detail">{f.detail}</td></tr>)}</tbody>
            </table>
          </div>
        )}
      </section>
      <section className="in-section">
        <div className="in-verdict">
          <div className="in-gauge" data-v={result.verdict} aria-hidden="true"><i /></div>
          <div>
            <h2 className="in-verdict-title">{title}</h2>
            <p className="in-verdict-note">{note}</p>
            <ul className="in-facts">
              <li><b className="in-path">{result.model_filename}</b></li>
              {scan.format && <li>format <b>{scan.format}</b></li>}
              {scan.size_bytes != null && <li className="in-num">{(scan.size_bytes / 1024).toFixed(1)} KB</li>}
              {scan.sha256 && <li className="in-num">sha256 {scan.sha256.slice(0, 16)}…</li>}
            </ul>
          </div>
        </div>
      </section>
      <section className="in-section">
        {result.interpretation && <p className="in-prose">{result.interpretation}</p>}
        <div className="in-actions">
          <button className="in-btn in-btn-quiet" onClick={saveJson}>Download scan (JSON)</button>
          {actions}
        </div>
      </section>
    </div>
  );
}

const when = (iso) => new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });

/** One saved scan, opened from the recent scans. */
function ScanDetail({ id }) {
  const { data: result, error, retry } = useLoad(useCallback(() => api.getModelScan(id), [id]));
  return (
    <main className="in-main"><div className="in-page">
      <Link to="/model-scan" className="in-link-text in-back">← Scanner</Link>
      {error && <LoadError error={error} onRetry={retry} />}
      {!result && !error && <Skeleton label="Loading this scan…" title lines={2} rows={3} />}
      {result && (
        <>
          <PageTitle className="in-h1-file">{result.model_filename || 'Model scan'}</PageTitle>
          <PageIntro>{`Model file scan${result.scan?.format ? ` · ${result.scan.format}` : ''}`}</PageIntro>
          <ScanResult result={result} actions={<Link to="/model-scan" className="in-btn" style={{ textDecoration: 'none' }}>Scan another file</Link>} />
        </>
      )}
    </div></main>
  );
}

const fetchScans = (before) => api.getModelScanHistory(10, before).then((d) => ({ items: d.scans || [], next: d.next_before }));

export default function ModelScanPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [over, setOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const { data: firstScans, error: recentError, retry: reloadRecent } = useLoad(useCallback(() => (id ? Promise.resolve(null) : fetchScans()), [id]));
  const moreScans = useMore(fetchScans, firstScans?.next);
  const recent = firstScans && [...firstScans.items, ...moreScans.items];
  const stop = useRef(() => {});
  useEffect(() => () => stop.current(), []);
  if (id) return <ScanDetail id={id} />;

  const choose = (f) => {
    if (!f) return;
    if (!EXTENSIONS.some((ext) => f.name.toLowerCase().endsWith(ext))) return setError(`Supported files: ${EXTENSIONS.join(' ')}`);
    if (f.size > MAX_MB * 1048576) return setError(`The file is larger than ${MAX_MB} MB.`);
    setError(null); setFile(f); setResult(null);
  };
  const scan = async (e) => {
    e.preventDefault();
    if (!file) return;
    setLoading(true); setError(null);
    try {
      const jobId = await startDirect(api, 'model', file);   // a large file straight to storage, when the server has it
      if (jobId) {                                           // the scan then runs as a job; loading ends with it
        stop.current = followJob(api, jobId, {
          onStatus: () => {},
          onDone: (raw) => { setResult(raw); setLoading(false); reloadRecent(); },
          onFail: (message) => { setError(message || 'The scan failed.'); setLoading(false); },
        });
        return;
      }
      const form = new FormData();
      form.append('model_file', file);
      setResult(await api.scanModel(form));
      reloadRecent();                              // the new scan is now first in the list
    } catch (err) { setError(err.message || 'The scan failed.'); }
    setLoading(false);
  };

  return (
    <main className="in-main"><div className="in-page">
      <PageTitle>Model file scanner</PageTitle>
      <PageIntro>Check whether a model file can run code when it is loaded. The file is read statically in an isolated process; it is never loaded or run. This does not tell you whether the model was trained on poisoned data.</PageIntro>
      {!result && (
        <form className="in-form" onSubmit={scan}>
          <label className="in-drop" data-over={over}
            onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
            onDrop={(e) => { e.preventDefault(); setOver(false); choose(e.dataTransfer.files[0]); }}>
            <input type="file" accept={EXTENSIONS.join(',')} onChange={(e) => choose(e.target.files[0])} />
            <strong>{file ? 'Selected, choose again to replace' : 'Choose a model file or drop it here'}</strong>
            {file ? <span className="in-file">{file.name} · {(file.size / 1024).toFixed(1)} KB</span>
                  : <span className="in-hint">pickle, joblib, PyTorch, safetensors, ONNX, NumPy, GGUF, Keras; up to {MAX_MB} MB</span>}
          </label>
          <div className="in-actions">
            <button className="in-btn" type="submit" disabled={!file || loading}>{loading ? 'Scanning…' : 'Scan'}</button>
            {error && <span className="in-error" role="alert">{error}</span>}
          </div>
        </form>
      )}
      {result && <ScanResult result={result} actions={<button className="in-btn" onClick={() => { setResult(null); setFile(null); }}>Scan another file</button>} />}
      <section className="in-section" style={{ marginTop: 32 }}>
        <h2 className="in-h2">Recent scans</h2>
        {recentError && <LoadError error={recentError} onRetry={reloadRecent} />}
        {!recent && !recentError && <Skeleton label="Loading your recent scans…" lines={0} rows={3} />}
        {recent && !recent.length && <p className="in-empty">No scans yet. Scanned files are listed here.</p>}
        {recent?.length > 0 && (
          <div className="in-table-wrap">
            <table className="in-table in-table-stack">
              <thead><tr><th>File</th><th>Format</th><th>Verdict</th><th>When</th></tr></thead>
              {/* the whole row opens the scan for a mouse; the file name is the real link for keyboards and screen readers */}
              <tbody>{recent.map((r) => (
                <tr key={r.id} className="in-row-link" onClick={() => navigate(`/model-scan/${r.id}`)}>
                  <td className="in-path" data-label="File"><Link to={`/model-scan/${r.id}`} className="in-link-text" onClick={(e) => e.stopPropagation()}>{r.model_filename}</Link></td>
                  <td className="in-tag" data-label="Format">{r.model_type}</td>
                  <td data-label="Verdict"><span className="in-chip" data-v={r.verdict}>{(VERDICTS[r.verdict] || [r.verdict])[0]}</span></td>
                  <td className="in-tag" data-label="When">{when(r.created_at)}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        )}
        <LoadMoreButton more={moreScans} what="scans" />
      </section>
    </div></main>
  );
}
