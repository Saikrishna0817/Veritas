import { Fragment, useState } from 'react';

// Results of any analysis (table, images, audio, video), shared by the upload page, History and the dashboard.
/* eslint-disable react-refresh/only-export-components */

// ── What each data type accepts and what its result cannot tell you ───────────────────────────────
export const KINDS = {
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
             'At 1% poison, label flips are ranked but rarely give a batch verdict (about 1 in 10 batches).'],
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
export const VERDICTS = {
  CONFIRMED_POISONED: { title: 'Strong evidence of poisoning', note: 'The evidence passes the 1-in-10,000 false-alarm level. Review the listed items before training.' },
  SUSPICIOUS: { title: 'Evidence of poisoning', note: 'The evidence passes the 1-in-100 false-alarm level. Review the listed items before training.' },
  LOW_RISK: { title: 'Weak evidence of poisoning', note: 'The evidence passes the 1-in-20 false-alarm level only. One clean dataset in twenty looks like this.' },
  CLEAN: { title: 'No evidence of poisoning found', note: 'Below the 1-in-20 false-alarm level. This is not proof the data is clean: see the limits below.' },
};

export const CHECK_NAMES = {
  label_check: ['Labels', 'labels that do not fit the rest of the row'], shift_check: ['Distribution shift', 'incoming rows a model can tell apart from the reference'],
  resid_check: ['Column consistency', 'rows whose columns no longer fit together'],
  label: ['Labels', 'labels that do not fit the content'], shift: ['Content shift', 'incoming items that look unlike the reference'],
  pixel: ['Pixel statistics', 'patches, noise and blending'], dup: ['Near-duplicates', 'copies of the same item'],
  cell: ['Shared patch', 'one local pattern repeated across items'], resid: ['Shared faint pattern', 'a faint pattern repeated across items'],
  chroma: ['Colour noise', 'noise in flat regions'],
  sig: ['Within-class pattern', 'a pattern shared by part of one class, labels left correct'],
};

const SCALE_MAX = 8;
const TICKS = [[1.3, '1/20'], [2, '1/100'], [4, '1/10⁴']];   // false-alarm levels of the verdict cut-offs
const evLevel = (e) => (e >= 4 ? 3 : e >= 2 ? 2 : e >= 1.3 ? 1 : 0);
export const fmt = (n) => (typeof n === 'number' ? n.toLocaleString('en') : n);

// Saved results carry their source; this maps it to the data type.
export const KIND_OF_SOURCE = { upload: 'table', real_dataset: 'table', image_upload: 'images', audio_upload: 'audio', video_upload: 'video' };

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

export const ATTACK_NAMES = {
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

// No trusted reference: ranked lists per check, deliberately without a verdict.
function Screening({ kind, raw }) {
  const noun = KINDS[kind].noun;
  const notes = raw.dataset_info?.warnings || [];
  return (
    <div className="in-result">
      <section className="in-section">
        <div className="in-verdict">
          <div className="in-gauge" aria-hidden="true" />
          <div>
            <h2 className="in-verdict-title">Screened without a reference</h2>
            <p className="in-verdict-note">Every one of the {fmt(raw.n_samples)} {noun} was checked against the others. This ranks what to look at first; without data you trust it cannot say whether the set is poisoned.</p>
            <div className="in-limits">
              <span>Label flips and trigger patterns are found this way; changes that look like ordinary data (feature collisions, a shifted source) are not.</span>
              <span>The top of each list is worth a look even when nothing is wrong: compare a few items with ones you know.</span>
            </div>
          </div>
        </div>
      </section>
      {Object.entries(raw.lists || {}).map(([key, items]) => (
        <section key={key} className="in-section">
          <h2 className="in-h2">{raw.checks?.[key]?.description || key}</h2>
          <p className="in-sub">The {fmt(items.length)} highest-ranked {noun} for this check, most unusual first.</p>
          <FlaggedTable kind={kind} res={kind === 'table' ? { cells: items } : { items, nFlagged: items.length }} />
        </section>
      ))}
      {notes.length > 0 && (
        <section className="in-section">
          <h3 className="in-h3">Notes from reading your data</h3>
          <ul className="in-notes">{notes.map((n) => <li key={n}>{n}</li>)}</ul>
        </section>
      )}
    </div>
  );
}

export function Result({ kind, raw }) {
  if (raw.mode === 'self_check') return <Screening kind={kind} raw={raw} />;
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

