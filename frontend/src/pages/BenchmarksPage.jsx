import { Fragment } from 'react';
import { PageIntro, PageTitle } from '../ui/Reveal';

// Held-out numbers from the benchmark write-ups (docs/*-benchmark-findings.md, docs/model-card.md). Update them
// together with the docs whenever a detector changes.
const ROWS = [
  { kind: 'Tables', data: '12 OpenML datasets (credit-g, adult, spambase, …)', auprc: [0.49, 0.80], caught: [0.42, 0.85], clean: '3.7%',
    misses: 'Covariate shift at 1% (0.18); feature collisions are found as rows, rarely as cells.', doc: 'benchmark-v2-findings.md', note: 'On 8 further datasets never used for tuning: 0.71 / 0.92, 64% / 95% flagged, 1.9% clean.' },
  { kind: 'Images', data: 'CIFAR-10, GTSRB', auprc: [0.87, 0.92], caught: [0.83, 0.97], clean: '3.3%',
    misses: 'Random label flips at 1% (20% of batches); images are checked at 32×32.', doc: 'image-benchmark-findings.md' },
  { kind: 'Audio', data: 'Speech Commands (test speakers), ESC-50 (folds 4–5)', auprc: [0.59, 0.76], caught: [0.51, 0.86], clean: '3.3%',
    misses: 'Clean-label triggers in small sound-event sets (ESC-50); ESC-50 at 1% (0.29).', doc: 'audio-benchmark-findings.md', note: 'Speech Commands, before the within-class check; ESC-50 is 0.29 / 0.53. With it, clean-label triggers are flagged in 19 of 20 Speech Commands batches.' },
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
      <PageTitle>How well it works</PageTitle>
      <PageIntro>Every detector was tuned on development datasets and then measured once on held-out ones it never saw. Attacks were injected at known rates, so every flagged row or file can be checked against the truth.</PageIntro>

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
          <p>Clean-label triggers in video and in small audio sets: a trigger added to items whose labels stay correct. Images and speech-like audio are covered by the within-class check; video (where it is off) and sets with only a dozen items per class are not.</p>
          <p>Real label errors: on human-confirmed errors in CIFAR-10 test images, about 1 in 25 listed images is a confirmed error. The list says where to look, not what is wrong.</p>
          <p>Label flips on data unlike everyday photos: on SVHN digits and Tiny-ImageNet (200 classes) at 10% poison random flips are flagged in 60-80% of batches and targeted flips in 0-40%, against 100% on CIFAR-10. Tiny-ImageNet also gives false alarms on 6% of clean sets.</p>
          <p>A smooth, low-frequency trigger added to correctly labelled images. Attacks tuned to evade the checks were all caught when the attacker also changed labels (14 of 14 working backdoors flagged), but this one was never flagged at 5% poison. A check built for it caught strong versions but flagged about 1 in 4 clean image sets on new data, so it is off.</p>
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
