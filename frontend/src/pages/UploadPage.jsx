import { Fragment, useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../services/api';
import '../styles/instrument.css';

// ── What each data type accepts and what its result cannot tell you ───────────────────────────────
const KINDS = {
  table: {
    label: 'Table', accept: '.csv', maxMb: 200, noun: 'rows',
    format: 'A CSV with a label column. Text columns with up to 50 distinct values are checked as categories.',
    limits: ['Below about 3% poison the batch verdict is weak; the row list below is still worth reading.',
             'Rows moved towards a victim row with correct labels (feature collision) are found as rows, rarely as cells.'],
  },
  images: {
    label: 'Images', accept: '.zip', maxMb: 500, noun: 'images',
    format: 'A zip with one folder per class (cat/001.png). Up to 6,000 images are used, resized to 32×32.',
    limits: ['Images are checked at 32×32: a trigger smaller than about 1/32 of the image can vanish.',
             'Triggers added to correctly labelled images (clean-label) are not detected.'],
  },
  audio: {
    label: 'Audio', accept: '.zip', maxMb: 500, noun: 'clips',
    format: 'A zip of PCM WAV files with one folder per class (dog/001.wav). Clips are mixed to mono and padded or trimmed to one length.',
    limits: ['Triggers added to correctly labelled clips (clean-label) are not detected.',
             'Small sets (a few hundred clips) rarely give a batch verdict at 1% poison.'],
  },
  video: {
    label: 'Video', accept: '.zip', maxMb: 500, noun: 'clips',
    format: 'A zip of video files (mp4, avi, mov, mkv, webm) with one folder per class. 8 frames per clip are checked.',
    limits: ['Only triggers present on every frame were benchmarked; a moving or brief trigger can blur out.',
             'Triggers added to correctly labelled clips (clean-label) are not detected.'],
  },
};

// Verdict cut-offs are -log10 p values calibrated on clean batches (veritas2.verdict_for).
const VERDICTS = {
  CONFIRMED_POISONED: { title: 'Strong evidence of poisoning', note: 'The evidence passes the 1-in-10,000 false-alarm level. Review the listed items before training.' },
  SUSPICIOUS: { title: 'Evidence of poisoning', note: 'The evidence passes the 1-in-100 false-alarm level. Review the listed items before training.' },
  LOW_RISK: { title: 'Weak evidence of poisoning', note: 'The evidence passes the 1-in-20 false-alarm level only. One clean dataset in twenty looks like this.' },
  CLEAN: { title: 'No evidence of poisoning found', note: 'Below the 1-in-20 false-alarm level. This is not proof the data is clean: see the limits below.' },
};

const CHECK_NAMES = {
  label_check: ['Labels', 'labels that do not fit the rest of the row'], shift_check: ['Distribution shift', 'incoming rows a model can tell apart from the reference'],
  resid_check: ['Column consistency', 'rows whose columns no longer fit together'],
  label: ['Labels', 'labels that do not fit the content'], shift: ['Content shift', 'incoming items that look unlike the reference'],
  pixel: ['Pixel statistics', 'patches, noise and blending'], dup: ['Near-duplicates', 'copies of the same item'],
  cell: ['Shared patch', 'one local pattern repeated across items'], resid: ['Shared faint pattern', 'a faint pattern repeated across items'],
  chroma: ['Colour noise', 'noise in flat regions'],
};

const SCALE_MAX = 8;
const TICKS = [[1.3, '1/20'], [2, '1/100'], [4, '1/10⁴']];   // false-alarm levels of the verdict cut-offs
const evLevel = (e) => (e >= 4 ? 3 : e >= 2 ? 2 : e >= 1.3 ? 1 : 0);
const STEPS = [['uploading', 'Uploading'], ['queued', 'Queued'], ['running', 'Analysing'], ['completed', 'Done']];
const JOB_KEY = 'veritas.lastJob';
const fmt = (n) => (typeof n === 'number' ? n.toLocaleString('en') : n);

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

// ── Result ────────────────────────────────────────────────────────────────────────────────────────
function normalise(kind, r) {
  if (kind === 'table') {
    const lr = r.layer_results || {};
    const checks = ['label_check', 'shift_check', 'resid_check'].filter((k) => lr[k])
      .map((k) => ({ key: k, evidence: lr[k].evidence || 0, inRanking: k === 'label_check' ? lr[k].ran !== false : !!lr[k].in_row_ranking }));
    const info = r.dataset_info || {};
    return { verdict: r.verdict, checks, nIncoming: r.n_samples, nReference: info.reference_split, profile: r.attack_profile,
             cells: r.suspicious_cells || [], items: null, notes: [...(info.warnings || []), ...(r.warnings || [])] };
  }
  const info = r.dataset_info || {};
  const checks = Object.entries(r.checks || {}).map(([key, c]) => ({ key, evidence: c.evidence || 0, inRanking: c.in_row_ranking }));
  return { verdict: r.verdict, checks, nIncoming: info.n_incoming, nReference: info.n_reference, profile: null, cells: null,
           items: r.flagged_images || r.flagged_clips || [], nFlagged: r.n_flagged, leading: r.leading_check, notes: info.warnings || [] };
}

function Verdict({ kind, res }) {
  const v = VERDICTS[res.verdict] || VERDICTS.CLEAN;
  return (
    <div className="in-verdict">
      <div className="in-gauge" data-v={res.verdict} aria-hidden="true" />
      <div>
        <h2 className="in-verdict-title">{v.title}</h2>
        <p className="in-verdict-note">{v.note}</p>
        <ul className="in-facts">
          <li><b className="in-num">{fmt(res.nIncoming)}</b> incoming {KINDS[kind].noun} checked</li>
          <li>against <b className="in-num">{fmt(res.nReference)}</b> reference {KINDS[kind].noun}</li>
          {res.leading && <li>strongest check: <b>{(CHECK_NAMES[res.leading] || [res.leading])[0]}</b></li>}
        </ul>
        <div className="in-limits">{KINDS[kind].limits.map((l) => <span key={l}>{l}</span>)}</div>
      </div>
    </div>
  );
}

function EvidenceMeters({ checks }) {
  const pct = (x) => `${(Math.min(x, SCALE_MAX) / SCALE_MAX) * 100}%`;
  return (
    <div className="in-meters" role="list">
      <div className="in-scale-head" aria-hidden="true">
        <div />
        <div style={{ position: 'relative' }}>
          {TICKS.map(([x, l]) => <span key={x} className="in-tick-label" data-x={x} style={{ left: pct(x), top: 2 }}>{l}</span>)}
        </div>
        <div />
      </div>
      {checks.map((c) => {
        const [name, what] = CHECK_NAMES[c.key] || [c.key, ''];
        return (
          <div key={c.key} className="in-meter" role="listitem">
            <div className="in-meter-name">{name}<small>{what}</small></div>
            <div className="in-track" role="img" aria-label={`${name}: evidence ${c.evidence.toFixed(2)}`}>
              {c.evidence > 0 && <div className="in-fill" data-v={evLevel(c.evidence)} style={{ width: pct(c.evidence) }} />}
              {TICKS.map(([x]) => <span key={x} className="in-tick" style={{ left: pct(x) }} />)}
            </div>
            <div className="in-meter-val in-num">{c.evidence >= SCALE_MAX ? `>${SCALE_MAX}` : c.evidence.toFixed(1)}
              <small>{c.inRanking ? 'ranked' : 'not ranked'}</small></div>
          </div>
        );
      })}
    </div>
  );
}

const ATTACK_NAMES = {
  label_flip_random: 'Random label flipping', label_flip_targeted: 'Targeted label flipping', backdoor_ood: 'Backdoor trigger, out-of-range values',
  backdoor_inrange: 'Backdoor trigger, in-range values', feature_collision: 'Feature collision (clean-label)', covariate_shift: 'Covariate shift or corrupted source',
};
const FEATURE_NAMES = {
  label_evidence: 'Label check evidence', shift_evidence: 'Shift check evidence', resid_evidence: 'Column consistency evidence',
  label_blamed: 'Share of rows whose label is the top cell', trigger_columns: 'Share whose top cell is in the 3 commonest columns',
  label_concentration: 'Over-representation of their commonest label', out_of_range: 'Share with a value outside the reference range',
  tightness: 'Spread of the rows relative to the reference', unusual_columns: 'Share of their columns that look unusual',
};

function AttackProfile({ profile }) {
  const probs = Object.entries(profile.probabilities).sort((a, b) => b[1] - a[1]);
  return (
    <div className="in-profile">
      <div>
        <p className="in-profile-name">{ATTACK_NAMES[profile.likely] || profile.label}</p>
        <p className="in-sub">The closest of the six attacks Veritas was benchmarked on. On held-out data it names the exact attack 74% of the time and the right kind 86%. An attack of another kind still gets one of these names.</p>
      </div>
      <div>
        <div className="in-probs">
          {probs.map(([k, p]) => (
            <div key={k} className="in-prob" data-top={k === profile.likely}>
              <span>{ATTACK_NAMES[k] || k}</span>
              <div className="in-prob-bar" style={{ width: `${Math.max(p * 100, 1)}%` }} />
              <span className="in-num" style={{ textAlign: 'right' }}>{Math.round(p * 100)}%</span>
            </div>
          ))}
        </div>
        <details className="in-details">
          <summary>What the suspicious rows have in common</summary>
          <div className="in-kv">
            {Object.entries(profile.features).map(([k, v]) => <Fragment key={k}><span>{FEATURE_NAMES[k] || k}</span><span className="in-num">{v}</span></Fragment>)}
          </div>
        </details>
      </div>
    </div>
  );
}

function toCsv(rows) {
  const esc = (x) => `"${String(x ?? '').replace(/"/g, '""')}"`;
  return rows.map((r) => r.map(esc).join(',')).join('\n');
}
function download(name, text) {
  const url = URL.createObjectURL(new Blob([text], { type: 'text/csv' }));
  const a = document.createElement('a');
  a.href = url; a.download = name; a.click();
  URL.revokeObjectURL(url);
}

function FlaggedTable({ kind, res }) {
  const [showAll, setShowAll] = useState(false);
  const table = kind === 'table';
  const rows = table ? res.cells : res.items;
  if (!rows.length) {
    return <p className="in-empty">Nothing is listed: items are listed only when the batch reaches the 1-in-100 false-alarm level, so a weak result does not send you chasing noise.</p>;
  }
  const shown = showAll ? rows : rows.slice(0, 100);
  const exportCsv = () => download(table ? 'veritas-suspicious-rows.csv' : `veritas-flagged-${kind}.csv`, toCsv(table
    ? [['line', 'row_score', 'column', 'value', 'cell_score'], ...rows.flatMap((r) => r.cells.map((c) => [r.csv_line, r.row_score, c.column, c.value, c.score]))]
    : [['file', 'class', 'score', 'check'], ...rows.map((r) => [r.file, r.class, r.score, r.check])]));
  return (
    <>
      <div className="in-table-wrap">
        <table className="in-table">
          <thead>
            {table
              ? <tr><th className="in-right">Line</th><th className="in-right">Row score</th><th>Most suspicious cells (column, value, score)</th></tr>
              : <tr><th>File</th><th>Class</th><th className="in-right">Score</th><th>Flagged by</th></tr>}
          </thead>
          <tbody>
            {shown.map((r, i) => table ? (
              <tr key={i}>
                <td className="in-num">{r.csv_line}</td>
                <td className="in-num">{r.row_score.toFixed(2)}</td>
                <td>{r.cells.map((c) => (
                  <span key={c.column} className="in-cell"><b>{c.column}</b><em>{String(c.value)}</em><small>{c.score.toFixed(1)}</small></span>
                ))}</td>
              </tr>
            ) : (
              <tr key={i}>
                <td className="in-path">{r.file}</td>
                <td className="in-tag">{r.class ?? 'none'}</td>
                <td className="in-num">{r.score.toFixed(2)}</td>
                <td className="in-tag">{(CHECK_NAMES[r.check] || [r.check])[0]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="in-table-foot">
        <span className="in-num">{fmt(rows.length)} listed{!table && res.nFlagged > rows.length ? ` of ${fmt(res.nFlagged)} flagged` : ''}, most suspicious first.{table ? ' Line 1 is the header row.' : ''}</span>
        <span className="in-actions">
          {rows.length > 100 && <button className="in-btn in-btn-quiet" onClick={() => setShowAll(!showAll)}>{showAll ? 'Show first 100' : `Show all ${fmt(rows.length)}`}</button>}
          <button className="in-btn in-btn-quiet" onClick={exportCsv}>Download CSV</button>
        </span>
      </div>
    </>
  );
}

function Result({ kind, raw }) {
  const res = normalise(kind, raw);
  return (
    <div className="in-result">
      <section className="in-section"><Verdict kind={kind} res={res} /></section>
      <section className="in-section">
        <h2 className="in-h2">Evidence from each check</h2>
        <p className="in-sub">Each bar is −log₁₀ of a p-value measured against your reference: further right means harder to explain by chance. Ticks mark the false-alarm levels of the verdicts. The verdict uses the strongest check.</p>
        <EvidenceMeters checks={res.checks} />
      </section>
      {res.profile && (
        <section className="in-section">
          <h2 className="in-h2">What the suspicious rows look like</h2>
          <AttackProfile profile={res.profile} />
        </section>
      )}
      <section className="in-section">
        <h2 className="in-h2">{kind === 'table' ? 'Rows to review' : `${KINDS[kind].label} to review`}</h2>
        <FlaggedTable kind={kind} res={res} />
      </section>
      {res.notes.length > 0 && (
        <section className="in-section">
          <h3 className="in-h3">Notes from reading your data</h3>
          <ul className="in-notes">{res.notes.map((n) => <li key={n}>{n}</li>)}</ul>
        </section>
      )}
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
    <main className="instrument">
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
