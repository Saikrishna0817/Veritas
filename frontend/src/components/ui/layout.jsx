import { createElement } from 'react';
import { motion } from 'framer-motion';
import { PreviewTag } from './feedback';
import { EASE, DURATION } from './motion';

/** The standard page container under the app shell. */
export function Page({ children, className = '' }) {
  return <div className={`mx-auto max-w-[1360px] px-5 py-10 md:px-10 md:py-14 ${className}`}>{children}</div>;
}

/** Page title with one grey serif word (`accent`), an optional lede, and the page's actions on the right. */
export function PageHeader({ title, accent, lede, children, preview }) {
  return (
    <header className="flex flex-col gap-5 md:flex-row md:items-end md:justify-between pb-8 mb-10 border-b border-borderHairline">
      <div className="max-w-3xl">
        <h1 className="font-display font-medium text-[clamp(2.6rem,5.6vw,5rem)] leading-[1] text-textPrimary">
          {title}{accent && <> <span className="serif-accent">{accent}</span></>}
          {preview && <PreviewTag className="ml-4 align-middle" />}
        </h1>
        {lede && <p className="mt-4 text-textSecondary text-[17px] leading-relaxed max-w-[62ch]">{lede}</p>}
      </div>
      {children && <div className="flex flex-wrap gap-3">{children}</div>}
    </header>
  );
}

export function Panel({ title, aside, children, className = '' }) {
  return (
    <section className={`panel p-6 ${className}`}>
      {(title || aside) && (
        <div className="flex items-baseline justify-between gap-4 mb-5">
          {title && <h2 className="font-display font-bold text-[17px] text-textPrimary">{title}</h2>}
          {aside}
        </div>
      )}
      {children}
    </section>
  );
}

/**
 * The reference's step card: #1A1A1A, 28px corners, a big display number. With an `icon` the number sits
 * bottom-right beside a round icon chip; without one it sits under the label. Only pass real values;
 * use "—" while loading.
 */
export function StatCard({ label, value, unit, note, icon, accent }) {
  const number = (
    <span className={`font-display font-medium leading-none tracking-[-0.05em] tabular ${icon ? 'text-right text-[clamp(2.4rem,3.6vw,3.4rem)]' : 'text-[clamp(2.2rem,3.2vw,3rem)]'}`}>
      {value}{unit && <span className="ml-1 text-[0.42em] tracking-normal text-textSecondary">{unit}</span>}
    </span>
  );
  return (
    <div className="flex h-full flex-col rounded-[28px] border border-white/[0.07] bg-[#1A1A1A] p-6">
      <div className="flex items-start justify-between gap-3">
        <span className="text-sm font-medium text-textSecondary">{label}</span>
        {accent && <span className="rounded-lg bg-uv px-2 py-0.5 text-[10px] font-medium text-ink-900">{accent}</span>}
      </div>
      {icon ? (
        <div className="mt-auto flex items-end justify-between gap-3 pt-8">
          <span className="icon-chip h-11 w-11 shrink-0">{createElement(icon, { className: 'h-4 w-4 text-white', strokeWidth: 1.5, 'aria-hidden': true })}</span>
          {number}
        </div>
      ) : <div className="mt-6">{number}</div>}
      {note && <div className={`mt-2 text-xs text-textMuted ${icon ? 'mt-3 text-right' : ''}`}>{note}</div>}
    </div>
  );
}

/** Rises into place once, when scrolled into view. */
export function Reveal({ children, delay = 0, className = '' }) {
  return (
    <motion.div
      className={className}
      // Starts readable: if the animation never runs (stalled script, throttled tab), the content is still there.
      initial={{ opacity: 0.55, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-40px' }}
      transition={{ duration: DURATION.slow, delay, ease: EASE }}
    >
      {children}
    </motion.div>
  );
}
