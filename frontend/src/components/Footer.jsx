import Wordmark from './Wordmark';

export default function Footer() {
  return (
    <footer className="mt-20 border-t border-borderHairline px-6 py-10 md:px-12">
      <div className="mx-auto flex max-w-[1360px] flex-col gap-6 md:flex-row md:items-start md:justify-between">
        <div className="max-w-md space-y-3">
          <Wordmark size={16} />
          <p className="text-sm text-textMuted">
            A triage aid for training data and model files: which rows to review first, which files not to load.
            Not a certification, not proof of an attack, not a compliance claim.
          </p>
        </div>
        <dl className="grid grid-cols-2 gap-x-10 gap-y-3 font-mono text-[11px] text-textMuted sm:grid-cols-3">
          <div><dt className="uppercase tracking-wider">Detector</dt><dd className="mt-1 text-textSecondary">veritas2 · label + shift</dd></div>
          <div><dt className="uppercase tracking-wider">Row AUPRC @5%</dt><dd className="mt-1 text-uv">0.69 <span className="text-textMuted">vs 0.54 cleanlab</span></dd></div>
          <div><dt className="uppercase tracking-wider">Model files</dt><dd className="mt-1 text-textSecondary">read as bytes, never loaded</dd></div>
        </dl>
      </div>
    </footer>
  );
}
