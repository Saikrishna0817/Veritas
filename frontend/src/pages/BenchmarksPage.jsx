import { Fragment } from 'react';

// Held-out numbers from the benchmark write-ups (docs/*-benchmark-findings.md, docs/model-card.md). Update them
// together with the docs whenever a detector changes.
const ROWS = [
  { kind: 'Tables', data: '12 OpenML datasets (credit-g, adult, spambase, …)', auprc: [0.49, 0.80], caught: [0.42, 0.85], clean: '3.7%',
    misses: 'Covariate shift at 1% (0.18); feature collisions are found as rows, rarely as cells.', doc: 'benchmark-v2-findings.md' },
  { kind: 'Images', data: 'CIFAR-10, GTSRB', auprc: [0.86, 0.91], caught: [0.81, 0.96], clean: '2.9%',
    misses: 'Random label flips at 1% (20% of batches); images are checked at 32×32.', doc: 'image-benchmark-findings.md' },
  { kind: 'Audio', data: 'Speech Commands (test speakers), ESC-50 (folds 4–5)', auprc: [0.59, 0.76], caught: [0.51, 0.86], clean: '3.3%',
    misses: 'Clean-label triggers; small sets such as ESC-50 at 1% (0.29).', doc: 'audio-benchmark-findings.md', note: 'Speech Commands; ESC-50 is 0.29 / 0.53.' },
  { kind: 'Video', data: 'UCF-101 (official split 1, test groups)', auprc: [0.53, 0.75], caught: [0.46, 0.86], clean: '4.2%',
    misses: 'Clean-label triggers; only every-frame triggers were tested.', doc: 'video-benchmark-findings.md' },
];
const pct = (x) => `${Math.round(x * 100)}%`;

const LICENCES = [
  ['OpenML datasets', 'per dataset, mostly public domain or CC BY'], ['CIFAR-10', 'research use'], ['GTSRB', 'research use'],
  ['Speech Commands v0.02', 'CC BY 4.0'], ['ESC-50', 'CC BY-NC 3.0 (non-commercial)'], ['UCF-101', 'research use; terms not stated by the authors'],
];

export default function BenchmarksPage() {
  return (
    <main className="in-main"><div className="in-page">
      <h1 className="in-h1">How well it works</h1>
      <p className="in-lede">Every detector was tuned on development datasets and then measured once on held-out ones it never saw. Attacks were injected at known rates, so every flagged row or file can be checked against the truth.</p>

      <section className="in-section">
        <h2 className="in-h2">Held-out results</h2>
        <p className="in-sub">Row AUPRC: how well the flagged list puts poisoned items first (1.0 is perfect; random equals the poison rate). Caught: share of poisoned batches with evidence at the 1-in-20 level. Clean alarms: share of clean batches that reach the same level (designed to stay under 5%).</p>
        <div className="in-table-wrap">
          <table className="in-table">
            <thead><tr><th>Data</th><th>Held-out datasets</th><th className="in-right">AUPRC at 1% / 5%</th><th className="in-right">Caught at 1% / 5%</th><th className="in-right">Clean alarms</th><th>Main gaps</th></tr></thead>
            <tbody>
              {ROWS.map((r) => (
                <tr key={r.kind}>
                  <td><b>{r.kind}</b></td>
                  <td className="in-tag">{r.data}</td>
                  <td className="in-num">{r.auprc[0].toFixed(2)} / {r.auprc[1].toFixed(2)}{r.note && <div className="in-tag">{r.note}</div>}</td>
                  <td className="in-num">{pct(r.caught[0])} / {pct(r.caught[1])}</td>
                  <td className="in-num">{r.clean}</td>
                  <td className="in-tag">{r.misses}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="in-hint" style={{ marginTop: 12 }}>Averages over seven attack types per data type. Full tables per attack and rate are in the repository under docs/: {ROWS.map((r) => r.doc).join(', ')}.</p>
      </section>

      <section className="in-section">
        <h2 className="in-h2">What no check catches yet</h2>
        <div className="in-prose">
          <p>Clean-label triggers: a trigger added to items whose labels stay correct. Neither the label check nor the pooled pattern checks single them out, for images, audio or video.</p>
          <p>Very small poison rates in small datasets: at 1% of a few hundred items there are only a handful of poisoned ones, and the batch verdict rarely fires. The ranked list still helps.</p>
          <p>Attacks unlike the benchmarked ones. The table attack profile always names the closest of six known attacks.</p>
        </div>
      </section>

      <section className="in-section">
        <h2 className="in-h2">Benchmark data and licences</h2>
        <p className="in-sub">Used only to measure Veritas; none of it ships with the product.</p>
        <dl className="in-kv">{LICENCES.map(([k, v]) => <Fragment key={k}><dt>{k}</dt><dd style={{ margin: 0 }}>{v}</dd></Fragment>)}</dl>
      </section>
    </div></main>
  );
}
