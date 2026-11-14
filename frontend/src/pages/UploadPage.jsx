import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../services/api';
import { KINDS, Result } from '../components/results';

const STEPS = [['uploading', 'Uploading'], ['queued', 'Queued'], ['running', 'Analysing'], ['completed', 'Done']];
const JOB_KEY = 'veritas.lastJob';

function remember(value) {
  try { value ? localStorage.setItem(JOB_KEY, JSON.stringify(value)) : localStorage.removeItem(JOB_KEY); } catch { /* storage may be blocked */ }
}
function recall() {
  try { return JSON.parse(localStorage.getItem(JOB_KEY) || 'null'); } catch { return null; }
}

// ── Input ─────────────────────────────────────────────────────────────────────────────────────────
function DropZone({ kind, file, onFile, label = 'Choose a file or drop it here' }) {
  const [over, setOver] = useState(false);
  return (
    <label className="in-drop" data-over={over}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); onFile(e.dataTransfer.files[0]); }}>
      <input type="file" accept={KINDS[kind].accept} onChange={(e) => onFile(e.target.files[0])} />
      <strong>{file ? 'Selected, choose again to replace' : label}</strong>
      {file ? <span className="in-file">{file.name} · {file.size < 1048576 ? `${Math.max(1, Math.round(file.size / 1024))} KB` : `${(file.size / 1048576).toFixed(1)} MB`}</span>
            : <span className="in-hint">{KINDS[kind].accept} up to {KINDS[kind].maxMb} MB</span>}
    </label>
  );
}

function ReferenceOptions({ kind, mode, setMode, refFile, setRefFile, split, setSplit }) {
  const table = kind === 'table';
  const opts = table
    ? [['order', 'First 70% of the file', 'Use when rows are in time order and the early rows are trusted.'],
       ['file', 'A separate reference file', 'A CSV you trust, with the same columns. The whole main file is then checked.'],
       ['column', 'Rows before a cutoff', 'Rows with a column value below the cutoff (a date or number) are the reference.']]
    : [['order', 'A random 70% of each class', 'Poison anywhere in the zip also lands in the reference.'],
       ['file', 'A separate reference zip', 'Images or clips you trust, in the same folder layout. Recommended.']];
  return (
    <fieldset className="in-fieldset">
      <legend className="in-legend">Trusted reference</legend>
      {opts.map(([value, title, help]) => (
        <div key={value}>
          <label className="in-radio">
            <input type="radio" name="ref" value={value} checked={mode === value} onChange={() => setMode(value)} />
            <span>{title}<small>{help}</small></span>
          </label>
          {mode === value && value === 'file' && (
            <div className="in-inline" style={{ display: 'block' }}>
              <DropZone kind={kind} file={refFile} onFile={setRefFile} label="Choose the reference file" />
            </div>
          )}
          {mode === value && value === 'column' && (
            <div className="in-inline">
              <input className="in-input" placeholder="column, e.g. date" aria-label="Split column" value={split.column}
                onChange={(e) => setSplit({ ...split, column: e.target.value })} />
              <input className="in-input" placeholder="cutoff, e.g. 2024-01-01" aria-label="Cutoff value" value={split.cutoff}
                onChange={(e) => setSplit({ ...split, cutoff: e.target.value })} />
            </div>
          )}
        </div>
      ))}
    </fieldset>
  );
}

function SampleData({ disabled, onRun }) {
  const [list, setList] = useState([]);
  const [pick, setPick] = useState('');
  useEffect(() => { api.getRealDatasets().then((d) => setList(d.datasets || [])).catch(() => setList([])); }, []);
  if (!list.length) return null;
  return (
    <div className="in-actions in-hint">
      <span>No file at hand? Try a sample with known poison:</span>
      <select className="in-input" aria-label="Sample dataset" value={pick} onChange={(e) => setPick(e.target.value)} disabled={disabled}>
        <option value="">Choose a sample</option>
        {list.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
      </select>
      <button type="button" className="in-btn in-btn-quiet" disabled={disabled || !pick} onClick={() => onRun(pick)}>Run sample</button>
    </div>
  );
}

function JobProgress({ stage, startedAt }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => { const t = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(t); }, []);
  const at = STEPS.findIndex(([k]) => k === stage);
  return (
    <div aria-live="polite">
      <ol className="in-steps">
        {STEPS.map(([key, label], i) => (
          <li key={key} className="in-step" data-state={i < at || stage === 'completed' ? 'done' : i === at ? 'now' : 'todo'}>
            <span className="in-dot" />{label}
          </li>
        ))}
      </ol>
      <div className="in-elapsed in-num">{Math.round((now - startedAt) / 1000)} s elapsed. You can leave this page; the result is saved to History.</div>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────────────────────────
export default function UploadPage() {
  const [saved] = useState(recall);              // a job still running when the page was left
  const [kind, setKind] = useState(saved?.kind || 'table');
  const [file, setFile] = useState(null);
  const [refMode, setRefMode] = useState('order');
  const [refFile, setRefFile] = useState(null);
  const [split, setSplit] = useState({ column: '', cutoff: '' });
  const [job, setJob] = useState(saved?.id ? { ...saved, stage: 'queued' } : null);   // { id, kind, startedAt, stage }
  const [result, setResult] = useState(null); // { kind, raw }
  const [error, setError] = useState(null);
  const poll = useRef(null);

  const follow = useCallback((j) => {
    clearTimeout(poll.current);
    const tick = async () => {
      try {
        const s = await api.getJob(j.id);
        if (s.status === 'completed') { setJob(null); remember(null); setResult({ kind: j.kind, raw: s.result }); return; }
        if (s.status === 'failed') { setJob(null); remember(null); setError(s.error || 'The analysis failed.'); return; }
        setJob({ ...j, stage: s.status });
        poll.current = setTimeout(tick, 1500);
      } catch (e) { setJob(null); remember(null); setError(e.message); }
    };
    tick();
  }, []);

  useEffect(() => {
    if (saved?.id) follow(saved);
    return () => clearTimeout(poll.current);
  }, [follow, saved]);

  const chooseKind = (k) => { setKind(k); setFile(null); setRefFile(null); setRefMode('order'); setError(null); };
  const chooseFile = (f) => {
    if (!f) return;
    const spec = KINDS[kind];
    if (!f.name.toLowerCase().endsWith(spec.accept)) return setError(`Expected a ${spec.accept} file for ${spec.label.toLowerCase()}.`);
    if (f.size > spec.maxMb * 1048576) return setError(`The file is larger than ${spec.maxMb} MB.`);
    setError(null); setFile(f);
  };

  const runSample = async (id) => {          // samples run synchronously on the server: no job id to follow
    setError(null); setResult(null); setKind('table');
    setJob({ id: null, kind: 'table', startedAt: Date.now(), stage: 'running' });
    try { setResult({ kind: 'table', raw: await api.analyzeRealDataset(id) }); }
    catch (e) { setError(e.message); }
    finally { setJob(null); }
  };

  const ready = file && !job && (refMode !== 'file' || refFile) && (refMode !== 'column' || (split.column && split.cutoff));
  const submit = async (e) => {
    e.preventDefault();
    if (!ready) return;
    setError(null); setResult(null);
    const startedAt = Date.now();
    setJob({ id: null, kind, startedAt, stage: 'uploading' });
    try {
      const { job_id } = await api.submitAnalysis(kind, file, {
        referenceFile: refMode === 'file' ? refFile : null,
        splitColumn: refMode === 'column' ? split.column : null, splitCutoff: refMode === 'column' ? split.cutoff : null,
      });
      const j = { id: job_id, kind, startedAt };
      remember(j);
      follow(j);
    } catch (err) { setJob(null); setError(err.message); }
  };

  return (
    <main className="in-main">
      <div className="in-page">
        <h1 className="in-h1">Check a dataset</h1>
        <p className="in-lede">Upload the data you are about to train on and a reference you trust. Veritas measures how far the new data departs from the reference and lists what to review.</p>

        <form className="in-form" onSubmit={submit}>
          <div className="in-kinds" role="radiogroup" aria-label="Data type">
            {Object.entries(KINDS).map(([k, spec]) => (
              <label key={k} className="in-kind">
                <input type="radio" name="kind" value={k} checked={kind === k} onChange={() => chooseKind(k)} disabled={!!job} />
                <span>{spec.label}</span>
              </label>
            ))}
          </div>
          <p className="in-hint" style={{ margin: '-8px 0 0' }}>{KINDS[kind].format}</p>
          <DropZone kind={kind} file={file} onFile={chooseFile} />
          {kind === 'table' && <SampleData disabled={!!job} onRun={runSample} />}
          <ReferenceOptions kind={kind} mode={refMode} setMode={setRefMode} refFile={refFile} setRefFile={setRefFile} split={split} setSplit={setSplit} />
          <div className="in-actions">
            <button className="in-btn" type="submit" disabled={!ready}>{job ? 'Analysing…' : 'Analyse'}</button>
            {error && <span className="in-error" role="alert">{error}</span>}
          </div>
        </form>

        {job && <section className="in-section" style={{ marginTop: 28 }}><JobProgress stage={job.stage} startedAt={job.startedAt} /></section>}
        {result && <Result kind={result.kind} raw={result.raw} />}
      </div>
    </main>
  );
}
