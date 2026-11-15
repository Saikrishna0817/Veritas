import { motion } from 'framer-motion';
import { PageHeader, Panel, Table, Page, EASE, DURATION } from '../components/ui';

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

/** Two bars, 1% and 5% poison, for one measure. */
function PairBars({ values, label }) {
  return (
    <div className="space-y-1.5" role="img" aria-label={`${label}: ${values[0].toFixed(2)} at 1% poison, ${values[1].toFixed(2)} at 5%`}>
      {values.map((v, i) => (
        <div key={i} className="grid grid-cols-[2.2em_minmax(0,1fr)_2.8em] items-center gap-2 font-mono text-[11px]">
          <span className="text-textMuted">{i ? '5%' : '1%'}</span>
          <span className="h-1.5 rounded-sm bg-ink-950">
            <motion.span
              className="block h-full origin-left rounded-sm"
              style={{ width: `${v * 100}%`, background: i ? 'var(--uv)' : '#6E6A78' }}
              initial={{ scaleX: 0 }}
              whileInView={{ scaleX: 1 }}
              viewport={{ once: true }}
              transition={{ duration: DURATION.grow, delay: i * 0.08, ease: EASE }}
            />
          </span>
          <span className="text-right text-textSecondary">{v.toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}

export default function BenchmarksPage() {
  return (
    <Page>
      <PageHeader
        title="How well it"
        accent="works"
        lede="Every detector was tuned on development datasets and then measured once on held-out ones it never saw. Attacks were injected at known rates, so every flagged row or file can be checked against the truth."
      />

      <div className="space-y-8">
        <Panel title="Held-out results" aside={<span className="font-mono text-[11px] text-textMuted">docs/</span>}>
          <p className="-mt-2 mb-5 max-w-[78ch] text-sm text-textSecondary">
            Row AUPRC: how well the flagged list puts poisoned items first (1.0 is perfect; random equals the poison rate).
            Caught: share of poisoned batches with evidence at the 1-in-20 level. Clean alarms: share of clean batches that reach the same level, designed to stay under 5%.
          </p>
          <Table minWidth={860} className="-mx-3">
            <thead>
              <tr>
                <th>Data</th><th>Held-out datasets</th><th className="w-[18%]">Row AUPRC, 1% / 5%</th><th className="w-[18%]">Caught, 1% / 5%</th><th className="num">Clean alarms</th><th>Main gaps</th>
              </tr>
            </thead>
            <tbody>
              {ROWS.map((r) => (
                <tr key={r.kind}>
                  <td className="py-3 font-semibold">{r.kind}</td>
                  <td className="py-3 text-sm text-textSecondary">{r.data}</td>
                  <td className="py-3">
                    <PairBars values={r.auprc} label="Row AUPRC" />
                    {r.note && <div className="mt-1.5 text-[11px] text-textMuted">{r.note}</div>}
                  </td>
                  <td className="py-3"><PairBars values={r.caught} label="Batches caught" /><span className="sr-only">{pct(r.caught[0])} and {pct(r.caught[1])}</span></td>
                  <td className="num py-3 font-mono">{r.clean}</td>
                  <td className="py-3 text-sm text-textSecondary">{r.misses}</td>
                </tr>
              ))}
            </tbody>
          </Table>
          <p className="mt-4 text-xs text-textMuted">
            Averages over seven attack types per data type. Full tables per attack and rate are in the repository under docs/: {ROWS.map((r) => r.doc).join(', ')}.
          </p>
        </Panel>

        <div className="grid gap-8 lg:grid-cols-2">
          <Panel title="What no check catches yet">
            <div className="space-y-3 text-sm text-textSecondary">
              <p>Clean-label triggers: a trigger added to items whose labels stay correct. For audio and video, neither the label check nor the pooled pattern checks single them out.</p>
              <p>Very small poison rates in small datasets: at 1% of a few hundred items there are only a handful of poisoned ones, and the batch verdict rarely fires. The ranked list still helps.</p>
              <p>Attacks unlike the benchmarked ones. The table attack profile always names the closest of six known attacks.</p>
            </div>
          </Panel>
          <Panel title="Benchmark data and licences">
            <p className="-mt-2 mb-4 text-sm text-textSecondary">Used only to measure Veritas; none of it ships with the product.</p>
            <dl className="grid grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)] gap-x-4 gap-y-2 text-sm">
              {LICENCES.map(([k, v]) => (
                <div key={k} className="contents">
                  <dt className="font-mono text-textPrimary">{k}</dt>
                  <dd className="text-textSecondary">{v}</dd>
                </div>
              ))}
            </dl>
          </Panel>
        </div>
      </div>
    </Page>
  );
}
