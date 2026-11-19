import { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../services/api';
import { KINDS, KIND_OF_SOURCE, Result } from '../components/results';
import { Skeleton } from '../components/Skeleton';
import { LoadError } from '../components/LoadError';
import { useLoad } from '../hooks/useLoad';
import { PageIntro, PageTitle } from '../ui/Reveal';

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

// One analysis, loaded when it is picked. Keyed by id so a slow answer for an earlier pick never shows.
function Report({ id }) {
  const { data: res, error, retry } = useLoad(useCallback(() => api.getHistoricalResult(id), [id]));
  const kind = res && KIND_OF_SOURCE[res.source];
  const family = res && familyOf(res);
  if (error) return <LoadError error={error} onRetry={retry} />;
  if (!res) return <Skeleton label="Loading this analysis…" lines={2} rows={4} />;
  if (!kind) return <p className="in-empty">This record was made by an earlier version of Veritas and cannot be shown in the current layout.</p>;
  return (
    <>
      <p className="in-hint">Report for <span className="in-num">{res.dataset_info?.filename}</span>, generated {new Date().toLocaleString()}.</p>
      <Result kind={kind} raw={res} />
      {family && <Playbook family={family} />}
    </>
  );
}

export default function ReportsPage() {
  const [pick, setPick] = useState('');
  const { data: rows, error, retry } = useLoad(useCallback(() => api.getAnalysisHistory(100)
    .then((d) => (d.results || []).filter((r) => KIND_OF_SOURCE[r.source])), []));
  const chosen = pick || rows?.[0]?.id || '';
  return (
    <main className="in-main"><div className="in-page">
      <PageTitle>Reports</PageTitle>
      <PageIntro>A printable record of one analysis: the verdict, the evidence, the items to review and, when something was found, the response steps. Use your browser's print dialog to save it as a PDF.</PageIntro>
      {error && <LoadError error={error} onRetry={retry} />}
      {!rows && !error && <Skeleton label="Loading your analyses…" lines={1} rows={3} />}
      {rows && !rows.length && <p className="in-empty">No analyses yet. <Link to="/upload" className="in-link-text">Check a dataset</Link> first.</p>}
      {rows?.length > 0 && (
        <div className="in-actions" style={{ marginBottom: 8 }}>
          <select className="in-input" aria-label="Analysis" value={chosen} onChange={(e) => setPick(e.target.value)} style={{ minWidth: 0, width: 'min(420px, 100%)' }}>
            {rows.map((r) => <option key={r.id} value={r.id}>{r.filename} · {KINDS[KIND_OF_SOURCE[r.source]].label} · {new Date(r.created_at).toLocaleDateString()}</option>)}
          </select>
          <button className="in-btn in-btn-quiet" onClick={() => window.print()} disabled={!chosen}>Print or save as PDF</button>
        </div>
      )}
      {chosen && <Report key={chosen} id={chosen} />}
    </div></main>
  );
}
