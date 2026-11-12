import { lazy, Suspense } from 'react';
import Wordmark from './Wordmark';

const Specimen3D = lazy(() => import('./Specimen3D'));

/** Split sign-in layout: the specimen fills the left, the form sits on a ruled panel on the right. */
export default function AuthLayout({ title, lede, children, footer, tone = 'analyst' }) {
  return (
    <main className="uv-ground grid min-h-screen lg:grid-cols-[minmax(0,7fr)_minmax(420px,5fr)]">
      <section className="stars relative hidden overflow-hidden lg:block" aria-hidden>
        <div className="pointer-events-none absolute inset-0" style={{ background: 'radial-gradient(55% 45% at 50% 36%, rgba(160,130,255,0.55), rgba(71,0,214,0.32) 45%, transparent 75%)' }} />
        {/* The slide sits in the upper part so the headline below never covers it. */}
        <Suspense fallback={null}>
          <Specimen3D className="absolute inset-x-0 top-[6%] h-[60%]" loop />
        </Suspense>
        <div className="absolute left-10 top-10"><Wordmark size={22} /></div>
        <div className="absolute bottom-10 left-10 max-w-[30rem]">
          <p className="font-display text-[clamp(2rem,3.4vw,3.4rem)] font-medium leading-[0.95]">
            Every row on the slide.<br /><span className="serif-accent">The suspicious ones glow.</span>
          </p>
          <p className="mt-4 max-w-[44ch] text-textSecondary">
            Veritas ranks training rows by how likely their label is wrong or their features were tampered with, and reads model files without ever loading them.
          </p>
        </div>
      </section>

      <section className="flex items-center justify-center border-l border-borderHairline bg-ink-950/70 px-6 py-12 backdrop-blur-sm">
        <div className="w-full max-w-[400px]">
          <div className="mb-10 lg:hidden"><Wordmark size={22} /></div>
          {tone === 'admin' && (
            <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-statusCritical/40 px-3 py-1 text-[12px] text-statusCritical">
              Administrator access
            </div>
          )}
          <h1 className="font-display text-[34px] font-medium leading-tight">{title}</h1>
          {lede && <p className="mt-2 text-textSecondary">{lede}</p>}
          <div className="mt-8">{children}</div>
          {footer && <div className="mt-8 flex justify-between border-t border-borderHairline pt-2 text-sm [&_a]:inline-flex [&_a]:min-h-[44px] [&_a]:items-center">{footer}</div>}
        </div>
      </section>
    </main>
  );
}

// The sign-in pages import Field from here; it is the shared control from ui.jsx.
export { Field } from './ui';
