import { useEffect, useId, useRef } from 'react';
import { createPortal } from 'react-dom';
import { X } from 'lucide-react';
import { IconButton } from './actions';

/**
 * A modal for decisions that need the page held still: irreversible actions that also ask for input.
 * Built on <dialog>, so focus stays inside, Escape closes it and the page behind is inert.
 */
export function Dialog({ open, onClose, title, description, children }) {
  const ref = useRef(null);
  const titleId = useId();
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) {
      d.showModal();
      // showModal focuses the first button (Close); the first field is the useful place to start.
      d.querySelector('input:not([type=hidden]), select, textarea')?.focus();
    }
    if (!open && d.open) d.close();
  }, [open]);
  // Portalled to <body> so the parent's spacing utilities (space-y-*) cannot shift it off centre.
  return createPortal(
    <dialog ref={ref} className="veritas-dialog" aria-labelledby={titleId} onClose={onClose} onCancel={(e) => { e.preventDefault(); onClose?.(); }}
      onClick={(e) => { if (e.target === ref.current) onClose?.(); }}>
      {open && (
        <div className="edge-card quiet">
          <div className="edge-inner p-7 sm:p-8">
            <div className="flex items-start justify-between gap-4">
              <h2 id={titleId} className="font-display text-[26px] font-medium leading-tight">{title}</h2>
              <IconButton icon={X} label="Close" onClick={onClose} className="-mr-2 -mt-1" />
            </div>
            {description && <p className="mt-2 text-sm text-textSecondary">{description}</p>}
            <div className="mt-6">{children}</div>
          </div>
        </div>
      )}
    </dialog>,
    document.body,
  );
}
