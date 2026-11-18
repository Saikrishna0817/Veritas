import { useCallback, useMemo, useRef } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../services/api';
import { KINDS, KIND_OF_SOURCE, fmt } from '../components/results';
import { Skeleton } from '../components/Skeleton';
import { LoadError } from '../components/LoadError';
import { useLoad } from '../hooks/useLoad';
import { PageIntro, PageTitle } from '../ui/Reveal';
import { GLYPHS } from '../../shared/argilla/keys/glyphs';
import { setHot, useStage, useStageState } from '../scene/stage';

const VERDICT_WORDS = { CONFIRMED_POISONED: 'Strong evidence of poisoning', SUSPICIOUS: 'Evidence of poisoning', LOW_RISK: 'Weak evidence', CLEAN: 'No evidence found', SCREENED: 'Screened, no reference' };

// The glaze of a type's keys is its latest verdict's colour, as on its chip (argilla.css .in-chip)
const GLAZE = { CONFIRMED_POISONED: '--ev-3', SUSPICIOUS: '--ev-2', LOW_RISK: '--ev-1' };
const SHORT = { table: 'TAB', images: 'IMG', audio: 'AUD', video: 'VID' };

// Beside the title: one cluster of keys per data type checked, glazed by its latest verdict; nothing for a type
// never checked, and nothing at all until the results have loaded.
function LatestKeys({ latest }) {
  const ref = useRef(null);
  const items = useMemo(() => {
    if (!latest) return null;
    const kinds = Object.keys(KINDS);
    const out = kinds.flatMap((k, group) => (latest[k]
      ? [...SHORT[k]].map((ch) => ({ glyph: GLYPHS.indexOf(ch), tint: GLAZE[latest[k].verdict] || '--ev-0', group, groups: kinds.length }))
      : []));
    return out.length ? out : null;
  }, [latest]);
  useStage('keys', ref, { items });
  return <div className="in-head-stage" ref={ref} aria-hidden="true" />;
}

export default function Dashboard() {
  const { hot } = useStageState();
  const { data: latest, error, retry } = useLoad(useCallback(() => api.getAnalysisHistory(200).then((d) => {
    const byKind = {};
    for (const r of d.results || []) {
      const k = KIND_OF_SOURCE[r.source];
      if (k && !byKind[k]) byKind[k] = r;        // rows arrive newest first
    }
    return byKind;
  }), []));
  return (
    <main className="in-main"><div className="in-page">
      <div className="in-page-head in-page-head-wide">
        <div>
          <p className="in-h3">Overview</p>
          <PageTitle>{'Check the data\nbefore you train.'}</PageTitle>
          <PageIntro>Veritas compares a dataset with a reference you trust and lists the rows or files that look poisoned, with the evidence for each.</PageIntro>
        </div>
        <LatestKeys latest={latest} />
      </div>
      <Link to="/upload" className="in-btn" style={{ display: 'inline-block', textDecoration: 'none' }}>Check data</Link>

      <section className="in-section" style={{ marginTop: 48 }}>
        <h2 className="in-h2">Latest result for each data type</h2>
        <p className="in-sub">Open one for its evidence and the items to review. <Link to="/history" className="in-link-text">All analyses</Link></p>
        {error && <LoadError error={error} onRetry={retry} />}
        {!latest && !error && <Skeleton label="Loading your latest results…" lines={0} rows={4} />}
        {latest && (
          <div className="in-latest">
            {Object.entries(KINDS).map(([k, spec], gi) => {
              const r = latest[k];
              // a checked type's row and its keys light up together
              const sync = r ? { onMouseEnter: () => setHot(gi), onMouseLeave: () => setHot(-1), 'data-hot': hot === gi || undefined } : {};
              return (
                <div key={k} className="in-latest-row" {...sync}>
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
