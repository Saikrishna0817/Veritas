import { useEffect, useState } from 'react';
import { api } from '../services/api';
import { MODEL_VERDICTS as VERDICTS } from '../components/results';

const EXTENSIONS = ['.pkl', '.pickle', '.joblib', '.pt', '.pth', '.bin', '.ckpt', '.safetensors', '.onnx', '.npy', '.npz', '.gguf', '.h5', '.keras', '.zip'];
const MAX_MB = 200;

function ScanResult({ result, onReset }) {
  const scan = result.scan || {};
  // An unknown code is shown as it is, never as another verdict.
  const [title, note] = VERDICTS[result.verdict] || [result.verdict ? `Verdict: ${String(result.verdict).replace(/_/g, ' ').toLowerCase()}` : 'No verdict in this result', 'This verdict is not one the interface recognises. Download the scan to read it in full.'];
  const findings = scan.findings || [];
  const saveJson = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' }));
    const a = document.createElement('a');
    a.href = url; a.download = `veritas-model-scan-${result.scan_id || Date.now()}.json`; a.click();
    URL.revokeObjectURL(url);
  };
  return (
    <div className="in-result">
      <section className="in-section">
        <div className="in-verdict">
          <div className="in-gauge" data-v={result.verdict} aria-hidden="true" />
          <div>
            <h2 className="in-verdict-title">{title}</h2>
            <p className="in-verdict-note">{note}</p>
            <ul className="in-facts">
              <li><b className="in-num">{result.model_filename}</b></li>
              <li>format <b>{scan.format}</b></li>
              <li className="in-num">{scan.size_bytes != null ? `${(scan.size_bytes / 1024).toFixed(1)} KB` : ''}</li>
              {scan.sha256 && <li className="in-num">sha256 {scan.sha256.slice(0, 16)}…</li>}
            </ul>
          </div>
        </div>
      </section>
      <section className="in-section">
        <h2 className="in-h2">Findings</h2>
        {findings.length === 0 ? <p className="in-empty">Nothing flagged.</p> : (
          <div className="in-table-wrap">
            <table className="in-table">
              <thead><tr><th>Severity</th><th>Detail</th></tr></thead>
              <tbody>{findings.map((f, i) => <tr key={i}><td className="in-tag">{f.severity}</td><td>{f.detail}</td></tr>)}</tbody>
            </table>
          </div>
        )}
      </section>
      {(scan.imports || []).length > 0 && (
        <section className="in-section">
          <h2 className="in-h2">Imports the file would perform</h2>
          <p className="in-sub">What loading the file would import. Each one is checked against known data containers.</p>
          <ul className="in-notes in-num">{scan.imports.map(([m, n]) => <li key={`${m}.${n}`}>{m}.{n}</li>)}</ul>
        </section>
      )}
      <section className="in-section">
        <p className="in-prose">{result.interpretation}</p>
        <div className="in-actions">
          <button className="in-btn in-btn-quiet" onClick={saveJson}>Download scan (JSON)</button>
          <button className="in-btn" onClick={onReset}>Scan another file</button>
        </div>
      </section>
    </div>
  );
}

export default function ModelScanPage() {
  const [file, setFile] = useState(null);
  const [over, setOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [recent, setRecent] = useState([]);
  useEffect(() => { api.getModelScanHistory(10).then((d) => setRecent(d.scans || [])).catch(() => setRecent([])); }, [result]);

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
      const form = new FormData();
      form.append('model_file', file);
      setResult(await api.scanModel(form));
    } catch (err) { setError(err.message || 'The scan failed.'); }
    finally { setLoading(false); }
  };

  return (
    <main className="in-main"><div className="in-page">
      <h1 className="in-h1">Model file scanner</h1>
      <p className="in-lede">Check whether a model file can run code when it is loaded. The file is read statically in an isolated process; it is never loaded or run. This does not tell you whether the model was trained on poisoned data.</p>
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
      {result && <ScanResult result={result} onReset={() => { setResult(null); setFile(null); }} />}
      {recent.length > 0 && (
        <section className="in-section" style={{ marginTop: 32 }}>
          <h2 className="in-h2">Recent scans</h2>
          <div className="in-table-wrap">
            <table className="in-table">
              <thead><tr><th>File</th><th>Format</th><th>Verdict</th><th>When</th></tr></thead>
              <tbody>{recent.map((r) => (
                <tr key={r.id}>
                  <td className="in-path">{r.model_filename}</td><td className="in-tag">{r.model_type}</td>
                  <td><span className="in-chip" data-v={r.verdict}>{(VERDICTS[r.verdict] || [r.verdict])[0]}</span></td>
                  <td className="in-tag">{new Date(r.created_at).toLocaleString()}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        </section>
      )}
    </div></main>
  );
}
