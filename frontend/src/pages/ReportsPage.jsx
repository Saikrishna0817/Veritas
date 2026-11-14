import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../services/api';
import { KINDS, KIND_OF_SOURCE, Result } from '../components/results';

// Which response playbook fits a result: tables carry an attack family; for images, audio and video the strongest
// check points at one (a shared patch or pixel change is trigger-like, a shift is a changed source).
const FAMILY_OF_CHECK = { label: 'label_flip', cell: 'backdoor', pixel: 'backdoor', resid: 'backdoor', chroma: 'backdoor', shift: 'covariate_shift' };

function familyOf(res) {
  if (res.verdict === 'CLEAN') return null;
  const t = res.attack_classification?.attack_type;
  if (t && t !== 'unknown' && !FAMILY_OF_CHECK[t]) return t;          // table results name the family directly
  return FAMILY_OF_CHECK[res.leading_check || t] || null;
}

function Playbook({ family }) {
  const [p, setP] = useState(null);
  useEffect(() => { api.getPlaybook(family).then(setP).catch(() => setP(null)); }, [family]);
  if (!p) return null;
  return (
    <section className="in-section">
      <h2 className="in-h2">Response: {p.attack}</h2>
      <p className="in-sub">{p.description}</p>
      <div className="in-cols">
        {[['Now', p.immediate_steps], ['Investigate', p.investigation_steps], ['Prevent', p.remediation]].map(([title, steps]) => (
          <div key={title}>
            <h3 className="in-h3">{title}</h3>
            <ol className="in-steps-list">{steps.map((s) => <li key={s}>{s}</li>)}</ol>
          </div>
        ))}
      </div>
      {p.regulatory && <p className="in-hint" style={{ marginTop: 16 }}>{p.regulatory}. A reference for your own process, not a compliance claim.</p>}
    </section>
  );
}

export default function ReportsPage() {
  const [rows, setRows] = useState(null);
  const [pick, setPick] = useState('');
  const [res, setRes] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    api.getAnalysisHistory(100).then((d) => {
      const list = (d.results || []).filter((r) => KIND_OF_SOURCE[r.source]);
      setRows(list);
      if (list.length) setPick(list[0].id);
    }).catch((e) => setError(e.message));
  }, []);
  useEffect(() => {
    if (pick) api.getHistoricalResult(pick).then(setRes).catch((e) => setError(e.message));
  }, [pick]);
  const kind = res && KIND_OF_SOURCE[res.source];
  const family = res && familyOf(res);
  return (
    <main className="in-main"><div className="in-page">
      <h1 className="in-h1">Reports</h1>
      <p className="in-lede">A printable record of one analysis: the verdict, the evidence, the items to review and, when something was found, the response steps. Use your browser's print dialog to save it as a PDF.</p>
      {error && <p className="in-error" role="alert">{error}</p>}
      {rows && !rows.length && <p className="in-empty">No analyses yet. <Link to="/upload" className="in-link-text">Check a dataset</Link> first.</p>}
      {rows?.length > 0 && (
        <div className="in-actions" style={{ marginBottom: 8 }}>
          <select className="in-input" aria-label="Analysis" value={pick} onChange={(e) => setPick(e.target.value)} style={{ minWidth: 320 }}>
            {rows.map((r) => <option key={r.id} value={r.id}>{r.filename} · {KINDS[KIND_OF_SOURCE[r.source]].label} · {new Date(r.created_at).toLocaleDateString()}</option>)}
          </select>
          <button className="in-btn in-btn-quiet" onClick={() => window.print()} disabled={!res}>Print or save as PDF</button>
        </div>
      )}
      {res && kind && (
        <>
          <p className="in-hint">Report for <span className="in-num">{res.dataset_info?.filename}</span>, generated {new Date().toLocaleString()}.</p>
          <Result kind={kind} raw={res} />
          {family && <Playbook family={family} />}
        </>
      )}
    </div></main>
  );
}
