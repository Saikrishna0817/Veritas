import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../services/api';
import { KINDS, KIND_OF_SOURCE, fmt } from '../components/results';

const VERDICT_WORDS = { CONFIRMED_POISONED: 'Strong evidence of poisoning', SUSPICIOUS: 'Evidence of poisoning', LOW_RISK: 'Weak evidence', CLEAN: 'No evidence found', SCREENED: 'Screened, no reference' };

export default function Dashboard() {
  const [latest, setLatest] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    api.getAnalysisHistory(200).then((d) => {
      const byKind = {};
      for (const r of d.results || []) {
        const k = KIND_OF_SOURCE[r.source];
        if (k && !byKind[k]) byKind[k] = r;        // rows arrive newest first
      }
      setLatest(byKind);
    }).catch((e) => setError(e.message));
  }, []);
  return (
    <main className="in-main"><div className="in-page">
      <h1 className="in-h1">Overview</h1>
      <p className="in-lede">Check a dataset before you train on it. Veritas compares it with a reference you trust and lists the rows or files that look poisoned.</p>
      <Link to="/upload" className="in-btn" style={{ display: 'inline-block', textDecoration: 'none' }}>Check data</Link>

      <section className="in-section" style={{ marginTop: 40 }}>
        <h2 className="in-h2">Latest result for each data type</h2>
        <p className="in-sub">Open one for its evidence and the items to review. <Link to="/history" className="in-link-text">All analyses</Link></p>
        {error && <p className="in-error" role="alert">{error}</p>}
        {latest && (
          <div className="in-latest">
            {Object.entries(KINDS).map(([k, spec]) => {
              const r = latest[k];
              return (
                <div key={k} className="in-latest-row">
                  <span className="in-tag">{spec.label}</span>
                  {r ? <span><span className="in-path">{r.filename}</span> <span className="in-hint">· {fmt(r.n_samples)} incoming {spec.noun} · {new Date(r.created_at).toLocaleDateString()}</span></span>
                     : <span className="in-hint">Not checked yet</span>}
                  {r ? <Link to={`/history/${r.id}`} className="in-chip in-link-text" data-v={r.verdict}>{VERDICT_WORDS[r.verdict] || r.verdict}</Link>
                     : <Link to="/upload" className="in-link-text">Check {spec.label.toLowerCase()}</Link>}
                </div>
              );
            })}
          </div>
        )}
      </section>

      <section className="in-section">
        <h2 className="in-h2">What a result means</h2>
        <div className="in-prose">
          <p>A verdict is a false-alarm level measured against your reference, not proof. “Evidence of poisoning” means one clean dataset in a hundred would look at least this unusual.</p>
          <p>Every check was measured on held-out public datasets before it shipped; <Link to="/benchmarks" className="in-link-text">see how well each data type does</Link>, including what it misses.</p>
        </div>
      </section>
    </div></main>
  );
}
