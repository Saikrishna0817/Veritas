import { createElement, useId, useRef } from 'react';
import { motion } from 'framer-motion';
import { SPRING } from './motion';

// One row that scrolls sideways when it runs out of room: on a phone a wrapped track turned into a block of buttons.
const TRACK = 'inline-flex max-w-full gap-1 overflow-x-auto rounded-[20px] bg-white/[0.04] p-1 ring-1 ring-inset ring-white/[0.06] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden';

/**
 * A row of toggle buttons for filters and pickers (aria-pressed). Use Tabs instead when the choice
 * swaps a whole panel. options: [{ id, label, icon?, count? }].
 */
export function Segmented({ options, value, onChange, label, className = '' }) {
  const layoutId = useId();
  return (
    <div role="group" aria-label={label} className={`${TRACK} ${className}`}>
      {options.map((o) => (
        <SegmentButton key={String(o.id)} option={o} active={value === o.id} layoutId={layoutId} onClick={() => onChange(o.id)} aria-pressed={value === o.id} />
      ))}
    </div>
  );
}

function SegmentButton({ option, active, layoutId, ...props }) {
  return (
    <button
      type="button"
      {...props}
      className={`relative inline-flex min-h-[40px] shrink-0 items-center gap-2 whitespace-nowrap rounded-2xl px-4 py-2 text-sm font-medium transition-colors duration-200 ${active ? 'text-ink-900' : 'text-textSecondary hover:bg-white/[0.06] hover:text-textPrimary'}`}
    >
      {active && <motion.span layoutId={layoutId} className="absolute inset-0 rounded-2xl bg-white" transition={SPRING} aria-hidden />}
      <span className="relative inline-flex items-center gap-2">
        {option.icon && createElement(option.icon, { className: 'h-4 w-4', 'aria-hidden': true })}
        {option.label}
        {option.count != null && <span className={`tabular text-xs ${active ? 'text-ink-900/60' : 'text-textMuted'}`}><span className="sr-only"> (</span>{option.count}<span className="sr-only">)</span></span>}
      </span>
    </button>
  );
}

/**
 * Tabs that swap a panel: arrow keys, Home and End move between them. Render each panel with
 * <TabPanel idBase={…} value={…}> using the same idBase.
 */
export function Tabs({ tabs, value, onChange, label, idBase, className = '' }) {
  const layoutId = useId();
  const refs = useRef({});
  const onKeyDown = (e) => {
    const i = tabs.findIndex((t) => t.id === value);
    const next = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 }[e.key];
    if (next === undefined) return;
    e.preventDefault();
    const t = tabs[(next + tabs.length) % tabs.length];
    onChange(t.id);
    refs.current[t.id]?.focus();
  };
  return (
    <div role="tablist" aria-label={label} onKeyDown={onKeyDown} className={`${TRACK} ${className}`}>
      {tabs.map((t) => (
        <SegmentButton
          key={t.id}
          ref={(el) => { refs.current[t.id] = el; }}
          option={t}
          active={value === t.id}
          layoutId={layoutId}
          role="tab"
          id={`${idBase}-tab-${t.id}`}
          aria-selected={value === t.id}
          aria-controls={`${idBase}-panel-${t.id}`}
          tabIndex={value === t.id ? 0 : -1}
          onClick={() => onChange(t.id)}
        />
      ))}
    </div>
  );
}

export function TabPanel({ idBase, value, children }) {
  return <div role="tabpanel" id={`${idBase}-panel-${value}`} aria-labelledby={`${idBase}-tab-${value}`} tabIndex={0} className="focus-visible:outline-offset-8">{children}</div>;
}

/** A radio group drawn as cards, for choices that need a sentence each. options: [{ id, label, hint }]. */
export function RadioCards({ name, label, options, value, onChange, stacked, hideLabel, className = '' }) {
  return (
    <fieldset className={className}>
      <legend className={hideLabel ? 'sr-only' : 'rule-label'}>{label}</legend>
      <div className={`grid gap-2 ${hideLabel ? '' : 'mt-2'} ${stacked ? '' : '[grid-template-columns:repeat(auto-fit,minmax(min(100%,200px),1fr))]'}`}>
        {options.map((o) => (
          <label key={o.id} className={`flex cursor-pointer gap-3 rounded-2xl border p-3.5 transition-colors has-[:focus-visible]:outline has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-uv ${value === o.id ? 'border-violet bg-violet/[0.08]' : 'border-borderHairline hover:border-borderStrong hover:bg-white/[0.02]'}`}>
            <input type="radio" name={name} value={o.id} checked={value === o.id} onChange={() => onChange(o.id)} className="mt-1 accent-[#B9A3FF] focus-visible:outline-none" />
            <span>
              <span className="block text-sm font-medium">{o.label}</span>
              {o.hint && <span className="block text-xs text-textMuted">{o.hint}</span>}
            </span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}
