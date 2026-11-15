import { useEffect, useState } from 'react';
import { Table2, Image as ImageIcon, AudioLines, Clapperboard } from 'lucide-react';
import { useAnalysisRun } from '../hooks/useAnalysisRun';
import JobCard from '../components/JobCard';
import AnalysisResult from '../components/analysis/AnalysisResult';
import { DATA_KINDS } from '../components/analysis/kinds';
import { PageHeader, Button, Dropzone, ErrorNote, Field, RadioCards, Segmented, Page } from '../components/ui';

const KIND_ORDER = ['csv', 'images', 'audio', 'video'];
const KIND_ICONS = { csv: Table2, images: ImageIcon, audio: AudioLines, video: Clapperboard };
const KIND_OPTIONS = KIND_ORDER.map((id) => ({ id, label: DATA_KINDS[id].label, icon: KIND_ICONS[id] }));

const STAGES = ['Reading schema', 'Splitting off the trusted reference', 'Label check', 'Distribution-shift check', 'Calibrating against clean rows'];
// No reference comes first and is the default, as on the server: trusting part of the file is something to opt into.
const NO_REFERENCE = { id: 'none', label: 'I have no trusted reference', hint: 'Every item is checked against the others: ranked lists, no verdict. Attacks that look like ordinary data are not found this way.' };
const REF_MODES = {
  csv: [
    NO_REFERENCE,
    { id: 'order', label: 'First 70% of this file', hint: 'Only works if the first rows are known good.' },
    { id: 'file', label: 'A separate trusted CSV', hint: 'Same columns as the batch.' },
    { id: 'column', label: 'Rows before a cutoff', hint: 'A number or date column, e.g. ingested_at < 2026-03-01.' },
  ],
  media: [
    NO_REFERENCE,
    { id: 'order', label: 'A random 70% of each class', hint: 'Poison anywhere in the zip also lands in the reference.' },
    { id: 'file', label: 'A separate trusted zip', hint: 'Same folder layout as the batch. Recommended.' },
  ],
};

function StageTicker() {
  const [i, setI] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setI((x) => Math.min(x + 1, STAGES.length - 1)), 1800);
    return () => clearInterval(t);
  }, []);
  return (
    <div className="panel scan-sweep p-5" role="status" aria-live="polite">
      <ol className="space-y-2 font-mono text-sm">
        {STAGES.map((s, k) => (
          <li key={s} className={`flex items-center gap-3 transition-colors ${k < i ? 'text-textMuted' : k === i ? 'text-uv' : 'text-textMuted/50'}`}>
            <span className="w-4">{k < i ? '✓' : k === i ? '›' : ''}</span>{s}
          </li>
        ))}
      </ol>
    </div>
  );
}

export default function UploadPage() {
  const { run: start, loading, job, result, error, setError, reset: clearResult, resumedKind } = useAnalysisRun(KIND_ORDER);
  const [kind, setKind] = useState(resumedKind || 'csv');
  const [file, setFile] = useState(null);
  const [refMode, setRefMode] = useState('none');
  const [referenceFile, setReferenceFile] = useState(null);
  const [splitColumn, setSplitColumn] = useState('');
  const [splitCutoff, setSplitCutoff] = useState('');
  const spec = DATA_KINDS[kind];
  const table = kind === 'csv';

  const chooseKind = (k) => {
    if (k === kind) return;
    setKind(k); setFile(null); setReferenceFile(null); setRefMode('none'); setError('');
  };

  const pick = (setter) => (f) => {
    if (!f) return;
    if (!f.name.toLowerCase().endsWith(spec.accept)) return setError(`Expected a ${spec.accept} file for ${spec.label.toLowerCase()}.`);
    if (f.size > spec.maxMb * 1024 * 1024) return setError(`That file is over ${spec.maxMb} MB.${table ? ' Split it or sample it first.' : ''}`);
    setError('');
    setter(f);
  };

  const run = async () => {
    if (!file) return;
    if (refMode === 'file' && !referenceFile) return setError(`Choose the trusted reference ${table ? 'CSV' : 'zip'}, or switch to another reference option.`);
    if (refMode === 'column' && (!splitColumn.trim() || !splitCutoff.trim())) return setError('Enter both the column name and the cutoff value.');
    start(file, {
      referenceFile: refMode === 'file' ? referenceFile : null,
      splitColumn: refMode === 'column' ? splitColumn.trim() : '',
      splitCutoff: refMode === 'column' ? splitCutoff.trim() : '',
      noReference: refMode === 'none',
    }, kind);
  };

  const reset = () => { clearResult(); setFile(null); setReferenceFile(null); };

  return (
    <Page>
      <PageHeader
        title={result ? 'Examination' : 'Examine a'}
        accent={result ? null : 'batch'}
        lede={result ? null : 'Give Veritas a labelled CSV, or a zip of images, audio or video in class folders, and say which part you trust. It ranks the rest by how likely each item was tampered with, then gives one calibrated verdict.'}
      />
      {result ? (
        <AnalysisResult result={result} onReset={reset} />
      ) : (
        <>
          <fieldset disabled={loading} className="mb-6">
            <legend className="sr-only">Data type</legend>
            <Segmented label="Data type" options={KIND_OPTIONS} value={kind} onChange={chooseKind} />
          </fieldset>
          <div className="grid gap-8 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
            <div className="min-w-0 space-y-5">
              <Dropzone
                key={kind}
                accept={spec.accept}
                file={file}
                onFile={pick(setFile)}
                label={table ? 'Drop the batch CSV' : `Drop the ${spec.label.toLowerCase()} zip`}
                hint={table
                  ? `Up to ${spec.maxMb} MB and 200,000 rows. Files over 10 MB run in the background. ${spec.format}`
                  : `Up to ${spec.maxMb} MB. ${spec.format} Zips always run in the background.`}
              />
              <ErrorNote>{error}</ErrorNote>
              {job ? <JobCard {...job} /> : loading && <StageTicker />}
            </div>

            <fieldset className="panel space-y-5 p-6" disabled={loading}>
              <legend className="sr-only">Trusted reference</legend>
              <div>
                <h2 className="font-display text-lg font-medium">Which {table ? 'rows' : spec.noun} do you trust?</h2>
                <p className="mt-1 text-sm text-textSecondary">Everything else is checked against these.</p>
              </div>
              <RadioCards name="ref" label="Reference source" hideLabel stacked options={table ? REF_MODES.csv : REF_MODES.media} value={refMode} onChange={setRefMode} />
              {refMode === 'file' && (
                <Dropzone accept={spec.accept} file={referenceFile} onFile={pick(setReferenceFile)} label={table ? 'Drop the trusted CSV' : 'Drop the trusted zip'} />
              )}
              {refMode === 'column' && (
                <div className="grid grid-cols-2 gap-3">
                  {[['Column', splitColumn, setSplitColumn, 'ingested_at'], ['Cutoff', splitCutoff, setSplitCutoff, '2026-03-01']].map(([l, val, set, ph]) => (
                    <Field key={l} label={l} mono value={val} onChange={(e) => set(e.target.value)} placeholder={ph} />
                  ))}
                </div>
              )}
              <Button variant="uv" onClick={run} loading={loading} disabled={!file} className="w-full py-4 text-[15px]">
                {loading ? 'Examining…' : refMode === 'none' ? 'Screen without a reference' : 'Run poisoning detection'}
              </Button>
            </fieldset>
          </div>
        </>
      )}
    </Page>
  );
}
