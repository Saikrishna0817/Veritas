import Wordmark from './Wordmark';

// Benchmark figures come from docs/model-card.md only.
export default function Footer() {
  return (
    <footer className="mt-20 border-t border-borderHairline px-5 py-10 md:px-10">
      <div className="mx-auto flex max-w-[1360px] flex-col gap-6 md:flex-row md:items-start md:justify-between">
        <div className="max-w-md space-y-3">
          <Wordmark size={16} />
          <p className="text-sm text-textMuted">
            A triage aid for training data and model files: which rows to review first, which files not to load.
            Not a certification, not proof of an attack, not a compliance claim.
          </p>
        </div>
        <dl className="grid grid-cols-2 gap-x-10 gap-y-4 text-sm text-textSecondary sm:grid-cols-3">
          <div><dt className="rule-label">Detector</dt><dd className="mt-1"><span className="font-mono text-[13px]">veritas2</span> · label + shift</dd></div>
          <div><dt className="rule-label">Row AUPRC at 5% poison</dt><dd className="mt-1"><span className="font-mono tabular text-uv">0.69</span> <span className="text-textMuted">vs <span className="font-mono tabular">0.54</span> cleanlab</span></dd></div>
          <div><dt className="rule-label">Model files</dt><dd className="mt-1">Read as bytes, never loaded</dd></div>
        </dl>
      </div>
    </footer>
  );
}
