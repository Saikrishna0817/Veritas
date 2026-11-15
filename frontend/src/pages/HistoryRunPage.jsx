import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import { api } from '../services/api';
import AnalysisResult from '../components/analysis/AnalysisResult';
import { DATA_KINDS, kindOf } from '../components/analysis/kinds';
import { PageHeader, Loading, ErrorState, Page } from '../components/ui';

/** One saved analysis, reopened from History with its full evidence and the items to review. */
export default function HistoryRunPage() {
  const { id } = useParams();
  const [state, setState] = useState({ id: null, result: null, error: null });
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let live = true;
    api.getHistoricalResult(id)
      .then((r) => live && setState({ id, result: r, error: null }))
      .catch((e) => live && setState({ id, result: null, error: e }));
    return () => { live = false; };
  }, [id, nonce]);

  const current = state.id === id ? state : { result: null, error: null };
  const { result, error } = current;
  const when = result?.dataset_info?.created_at || result?.created_at;

  return (
    <Page>
      <Link to="/history" className="mb-6 inline-flex min-h-[44px] items-center gap-2 text-sm text-violet hover:text-[var(--red-bright)]">
        <ArrowLeft className="h-4 w-4" aria-hidden /> All runs
      </Link>
      <PageHeader
        title="Saved"
        accent="run"
        lede={result ? `${DATA_KINDS[kindOf(result)].label}${when ? ` · examined ${new Date(when).toLocaleString()}` : ''}` : null}
      />
      {error && <ErrorState error={error} onRetry={() => setNonce((n) => n + 1)} />}
      {!result && !error && <Loading>Loading the saved result…</Loading>}
      {result && <AnalysisResult result={result} />}
    </Page>
  );
}
