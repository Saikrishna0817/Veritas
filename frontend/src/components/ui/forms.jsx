import { createElement, useId, useRef, useState } from 'react';
import { UploadCloud, FileCheck2 } from 'lucide-react';
import { formatBytes } from './format';

/**
 * A labelled form control. `as` picks input, select or textarea; `mono` is for data values (column names,
 * dates, keys). `hint` and `error` are announced with the control.
 */
export function Field({ label, hint, error, as = 'input', mono, hideLabel, className = '', children, id: givenId, ...props }) {
  const autoId = useId();
  const id = givenId || autoId;
  const describedBy = [hint && `${id}-hint`, error && `${id}-error`].filter(Boolean).join(' ') || undefined;
  return (
    <div className={className}>
      <label htmlFor={id} className={hideLabel ? 'sr-only' : 'rule-label'}>{label}</label>
      {createElement(as, {
        id,
        ...props,
        'aria-invalid': error ? true : undefined,
        'aria-describedby': describedBy,
        className: `field-input ${hideLabel ? '' : 'mt-1.5'} ${mono ? 'font-mono text-sm' : ''}`,
      }, children)}
      {hint && !error && <p id={`${id}-hint`} className="mt-1.5 text-xs text-textMuted">{hint}</p>}
      {error && <p id={`${id}-error`} className="mt-1.5 text-xs text-[#FFB4B5]">{error}</p>}
    </div>
  );
}

/** File picker that also takes a dropped file. Keyboard: Enter or Space opens the picker. */
export function Dropzone({ accept, onFile, file, hint, label = 'Drop a file or browse', icon = UploadCloud }) {
  const ref = useRef();
  const [over, setOver] = useState(false);
  return (
    <div
      role="button"
      tabIndex={0}
      onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), ref.current?.click())}
      onClick={() => ref.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); onFile(e.dataTransfer.files[0]); }}
      className={`relative flex min-h-[240px] cursor-pointer flex-col items-center justify-center gap-3 rounded-[28px] border border-dashed p-10 text-center transition-colors duration-300
        ${over ? 'border-uv bg-uv/[0.06] scan-sweep' : file ? 'border-statusSafe/60 bg-statusSafe/[0.04]' : 'border-borderStrong bg-ink-950/50 hover:border-violet hover:bg-violet/[0.05]'}`}
    >
      <input ref={ref} type="file" accept={accept} aria-label="Choose a file" className="sr-only" tabIndex={-1} onChange={(e) => onFile(e.target.files[0])} />
      {/* slide corner marks */}
      {['top-2 left-2 border-l border-t', 'top-2 right-2 border-r border-t', 'bottom-2 left-2 border-l border-b', 'bottom-2 right-2 border-r border-b'].map((c) => (
        <span key={c} className={`absolute h-3 w-3 rounded-[2px] border-uv/70 ${c}`} aria-hidden />
      ))}
      {file ? <FileCheck2 className="h-10 w-10 text-statusSafe" aria-hidden /> : createElement(icon, { className: `h-10 w-10 ${over ? 'text-uv' : 'text-textMuted'}`, 'aria-hidden': true })}
      {file ? (
        <>
          <div className="font-mono text-base text-statusSafe break-all">{file.name}</div>
          <div className="text-xs text-textMuted">{formatBytes(file.size)} · click or drop to replace</div>
        </>
      ) : (
        <>
          <div className="font-display text-lg font-bold text-textPrimary">{label}</div>
          {hint && <div className="text-xs text-textMuted max-w-[46ch]">{hint}</div>}
        </>
      )}
    </div>
  );
}
