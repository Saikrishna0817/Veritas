import { useCallback, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { api } from '../services/api';
import { KINDS, KIND_OF_SOURCE, Result, fmt } from '../components/results';
import { Skeleton } from '../components/Skeleton';
import { LoadError } from '../components/LoadError';
import { useLoad } from '../hooks/useLoad';
import { PageIntro, PageTitle } from '../ui/Reveal';

const VERDICT_WORDS = { CONFIRMED_POISONED: 'Strong evidence', SUSPICIOUS: 'Evidence', LOW_RISK: 'Weak evidence', CLEAN: 'No evidence', SCREENED: 'Screened, no reference' };
const when = (iso) => new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });

function Detail({ id }) {
  const { data: res, error, retry } = useLoad(useCallback(() => api.getHistoricalResult(id), [id]));
  const kind = res && KIND_OF_SOURCE[res.source];
  return (
    <main className="in-main"><div className="in-page">
      <Link to="/history" className="in-link-text in-back">← All analyses</Link>
      {error && <LoadError error={error} onRetry={retry} />}
      {!res && !error && <Skeleton label="Loading this analysis…" title lines={2} rows={4} />}
      {res && (
        <>
          <PageTitle className="in-h1-file">{res.dataset_info?.filename || 'Analysis'}</PageTitle>
          <PageIntro>{`${KINDS[kind]?.label || 'Data'}${res.dataset_info?.created_at ? ` · ${when(res.dataset_info.created_at)}` : ''}`}</PageIntro>
          {kind ? <Result kind={kind} raw={res} /> : <p className="in-empty">This record was made by an earlier version of Veritas and cannot be shown in the current layout.</p>}
        </>
      )}
    </div></main>
  );
}

export default function HistoryPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [filter, setFilter] = useState('all');
  const { data: rows, error, retry } = useLoad(useCallback(() => (id ? Promise.resolve(null)
    : api.getAnalysisHistory(200).then((d) => (d.results || []).filter((r) => KIND_OF_SOURCE[r.source]))), [id]));
  if (id) return <Detail id={id} />;
  const shown = (rows || []).filter((r) => filter === 'all' || KIND_OF_SOURCE[r.source] === filter);
  return (
    <main className="in-main"><div className="in-page">
      <PageTitle>History</PageTitle>
      <PageIntro>Every dataset you have checked. Open one to see its full evidence and the items to review.</PageIntro>
      <div className="in-filter" role="group" aria-label="Data type">
        {[['all', 'All'], ...Object.entries(KINDS).map(([k, v]) => [k, v.label])].map(([k, label]) => (
          <button key={k} className="in-pill" aria-pressed={filter === k} onClick={() => setFilter(k)}>{label}</button>
        ))}
      </div>
      {error && <LoadError error={error} onRetry={retry} />}
      {!rows && !error && <Skeleton label="Loading your analyses…" lines={0} rows={6} />}
      {rows && !shown.length && <p className="in-empty">Nothing here yet. <Link to="/upload" className="in-link-text">Check a dataset</Link> and it will be listed.</p>}
      {shown.length > 0 && (
        <div className="in-table-wrap">
          <table className="in-table in-table-stack">
            <thead><tr><th>File</th><th>Type</th><th>Verdict</th><th className="in-right">Incoming</th><th>When</th></tr></thead>
            <tbody>
              {/* the whole row opens the result for a mouse; the file name is the real link for keyboards and screen readers */}
              {shown.map((r) => (
                <tr key={r.id} className="in-row-link" onClick={() => navigate(`/history/${r.id}`)}>
                  <td className="in-path" data-label="File"><Link to={`/history/${r.id}`} className="in-link-text" onClick={(e) => e.stopPropagation()}>{r.filename}</Link></td>
                  <td className="in-tag" data-label="Type">{KINDS[KIND_OF_SOURCE[r.source]].label}</td>
                  <td data-label="Verdict"><span className="in-chip" data-v={r.verdict}>{VERDICT_WORDS[r.verdict] || r.verdict}</span></td>
                  <td className="in-num" data-label="Incoming">{fmt(r.n_samples)}</td>
                  <td className="in-tag" data-label="When">{when(r.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div></main>
  );
}
