import { useEffect, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { api } from '../services/api';
import { KINDS, KIND_OF_SOURCE, Result, fmt } from '../components/results';

const VERDICT_WORDS = { CONFIRMED_POISONED: 'Strong evidence', SUSPICIOUS: 'Evidence', LOW_RISK: 'Weak evidence', CLEAN: 'No evidence', SCREENED: 'Screened, no reference' };
const when = (iso) => new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });

function Detail({ id }) {
  const [res, setRes] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => { api.getHistoricalResult(id).then(setRes).catch((e) => setError(e.message)); }, [id]);
  const kind = res && KIND_OF_SOURCE[res.source];
  return (
    <main className="in-main"><div className="in-page">
      <Link to="/history" className="in-link-text in-back">← All analyses</Link>
      {error && <p className="in-error" role="alert">{error}</p>}
      {!res && !error && <p className="in-hint">Loading…</p>}
      {res && (
        <>
          <h1 className="in-h1">{res.dataset_info?.filename || 'Analysis'}</h1>
          <p className="in-lede">{KINDS[kind]?.label || 'Data'} · {res.dataset_info?.created_at ? when(res.dataset_info.created_at) : ''}</p>
          {kind ? <Result kind={kind} raw={res} /> : <p className="in-empty">This record was made by an earlier version of Veritas and cannot be shown in the current layout.</p>}
        </>
      )}
    </div></main>
  );
}

export default function HistoryPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [rows, setRows] = useState(null);
  const [filter, setFilter] = useState('all');
  const [error, setError] = useState(null);
  useEffect(() => {
    if (!id) api.getAnalysisHistory(200).then((d) => setRows((d.results || []).filter((r) => KIND_OF_SOURCE[r.source]))).catch((e) => setError(e.message));
  }, [id]);
  if (id) return <Detail id={id} />;
  const shown = (rows || []).filter((r) => filter === 'all' || KIND_OF_SOURCE[r.source] === filter);
  return (
    <main className="in-main"><div className="in-page">
      <h1 className="in-h1">History</h1>
      <p className="in-lede">Every dataset you have checked. Open one to see its full evidence and the items to review.</p>
      <div className="in-filter" role="group" aria-label="Data type">
        {[['all', 'All'], ...Object.entries(KINDS).map(([k, v]) => [k, v.label])].map(([k, label]) => (
          <button key={k} className="in-pill" aria-pressed={filter === k} onClick={() => setFilter(k)}>{label}</button>
        ))}
      </div>
      {error && <p className="in-error" role="alert">{error}</p>}
      {rows && !shown.length && <p className="in-empty">Nothing here yet. <Link to="/upload" className="in-link-text">Check a dataset</Link> and it will be listed.</p>}
      {shown.length > 0 && (
        <div className="in-table-wrap">
          <table className="in-table">
            <thead><tr><th>File</th><th>Type</th><th>Verdict</th><th className="in-right">Incoming</th><th>When</th></tr></thead>
            <tbody>
              {shown.map((r) => (
                <tr key={r.id} className="in-row-link" tabIndex={0} onClick={() => navigate(`/history/${r.id}`)}
                  onKeyDown={(e) => e.key === 'Enter' && navigate(`/history/${r.id}`)}>
                  <td className="in-path">{r.filename}</td>
                  <td className="in-tag">{KINDS[KIND_OF_SOURCE[r.source]].label}</td>
                  <td><span className="in-chip" data-v={r.verdict}>{VERDICT_WORDS[r.verdict] || r.verdict}</span></td>
                  <td className="in-num">{fmt(r.n_samples)}</td>
                  <td className="in-tag">{when(r.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div></main>
  );
}
