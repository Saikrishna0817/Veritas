import { createElement, lazy, Suspense, useMemo, useRef } from 'react';
import { Link } from 'react-router-dom';
import { motion, useScroll, useTransform } from 'framer-motion';
import { ArrowUpRight, FileCode2, ScanLine, ShieldCheck, Tags, Waves, Terminal } from 'lucide-react';
import { LogoMark } from '../components/Wordmark';

const Specimen3D = lazy(() => import('../components/Specimen3D'));

const EASE = [0.19, 1, 0.22, 1];
const BENCH = [
  { label: 'Row AUPRC at 5% poison', note: 'random ranking scores 0.05', rows: [['veritas2', 0.69], ['cleanlab', 0.54], ['Two-sample test', 0.49], ['Old Veritas', 0.11]] },
  { label: 'Batches caught at 5% false alarms', note: '12 held-out OpenML datasets', rows: [['veritas2', 0.78], ['cleanlab', 0.55], ['Old Veritas', 0.33], ['Two-sample test', 0.14]] },
];

function WhiteButton({ to, children, className = '' }) {
  return (
    <Link
      to={to}
      className={`group inline-flex items-center gap-2 whitespace-nowrap rounded-2xl bg-white px-6 py-3.5 text-sm font-medium text-ink-900 transition-colors duration-300 hover:bg-[#EDC5FC] ${className}`}
    >
      {children}
      <ArrowUpRight className="h-4 w-4 transition-transform duration-300 group-hover:translate-x-0.5 group-hover:-translate-y-0.5" aria-hidden />
    </Link>
  );
}

function PillNav() {
  return (
    <nav aria-label="Main" className="fixed left-1/2 top-4 z-50 -translate-x-1/2">
      <div className="flex items-center gap-1 rounded-2xl bg-white/10 p-2 backdrop-blur-xl">
        <a href="#top" aria-label="Veritas" className="flex h-9 w-9 items-center justify-center rounded-xl hover:bg-white/10"><LogoMark size={22} /></a>
        <a href="#how" className="whitespace-nowrap rounded-xl px-3 py-1.5 text-sm font-medium text-white hover:bg-white/10">How it works</a>
        <a href="#benchmarks" className="hidden rounded-xl px-3 py-1.5 text-sm font-medium text-white hover:bg-white/10 sm:block">Benchmarks</a>
        <Link to="/login" className="whitespace-nowrap rounded-xl px-3 py-1.5 text-sm font-medium text-white hover:bg-white/10">Sign in</Link>
      </div>
    </nav>
  );
}

function IconPill({ icon }) {
  return (
    <span className="mx-[0.18em] inline-flex h-[0.86em] w-[1.5em] translate-y-[0.06em] items-center justify-center rounded-[0.3em] border border-[#B9A3FF] bg-white/[0.04] align-baseline" aria-hidden>
      {createElement(icon, { className: 'h-[0.42em] w-[0.42em] text-white/80', strokeWidth: 1.4 })}
    </span>
  );
}

/** Paragraph whose words light up from grey to white as it scrolls through the viewport. */
function FillWord({ progress, range, children }) {
  const opacity = useTransform(progress, range, [0.28, 1]);
  return <motion.span style={{ opacity }}>{children} </motion.span>;
}
function ScrollFill({ text, className }) {
  const ref = useRef(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ['start 0.85', 'end 0.45'] });
  const words = useMemo(() => text.split(' '), [text]);
  return (
    <p ref={ref} className={className}>
      {words.map((w, i) => (
        <FillWord key={i} progress={scrollYProgress} range={[i / words.length, (i + 1) / words.length]}>{w}</FillWord>
      ))}
    </p>
  );
}

function Rise({ children, delay = 0, className = '' }) {
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y: 40 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-80px' }}
      transition={{ duration: 1.4, delay, ease: EASE }}
    >
      {children}
    </motion.div>
  );
}

function StepCard({ icon, title, children, className = '', style }) {
  return (
    <motion.div style={style} className={`glass-card w-full self-start lg:max-w-[19rem] ${className}`}>
      <div className="px-7 py-9">
        <span className="icon-chip h-12 w-12">{createElement(icon, { className: 'h-5 w-5 text-white', strokeWidth: 1.4, 'aria-hidden': true })}</span>
        <h3 className="mt-6 text-xl font-medium leading-tight">{title}</h3>
        <p className="mt-3 text-sm text-white/70">{children}</p>
      </div>
    </motion.div>
  );
}

function HowItWorks() {
  const ref = useRef(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ['start end', 'end start'] });
  const y1 = useTransform(scrollYProgress, [0, 1], [90, -90]);
  const y2 = useTransform(scrollYProgress, [0, 1], [160, -160]);
  const y3 = useTransform(scrollYProgress, [0, 1], [50, -120]);
  // Parallax only where the cards float over the headline; stacked cards on small screens stay put.
  const wide = typeof window !== 'undefined' && window.matchMedia('(min-width: 1024px)').matches;
  return (
    <section id="how" ref={ref} className="relative px-6 py-28 md:px-20 lg:py-40">
      <h2 className="font-display text-center text-[clamp(3.6rem,13vw,13.2rem)] font-semibold leading-[1.05] tracking-[-0.06em]">
        <Rise>Find the</Rise>
        <Rise delay={0.1}><span className="serif-accent block lg:ml-[22%]">Poison</span></Rise>
      </h2>
      <div className="relative z-10 mt-12 grid gap-6 lg:-mt-[16vw] lg:grid-cols-3 lg:gap-10">
        <StepCard icon={ScanLine} title="Give it a CSV and say which rows you trust" style={wide ? { y: y1 } : undefined} className="lg:mt-10">
          A separate trusted file, rows before a cut-off date, or the first 70% of the file.
        </StepCard>
        <StepCard icon={Waves} title="Veritas checks labels and distribution shift" style={wide ? { y: y2 } : undefined} className="lg:mt-56 lg:justify-self-center">
          Two checks, calibrated against your own clean rows, become one verdict with a p-value behind it.
        </StepCard>
        <StepCard icon={Tags} title="Review the ranked rows first" style={wide ? { y: y3 } : undefined} className="lg:mt-24 lg:justify-self-end">
          Every incoming row is ranked by evidence, so the first hundred you read are the ones most likely to be wrong.
        </StepCard>
      </div>
    </section>
  );
}

function Features() {
  return (
    <section className="px-6 py-24 md:px-20">
      <div className="mx-auto grid max-w-[1280px] lg:grid-cols-3 lg:divide-x lg:divide-white/15">
        <Rise className="flex flex-col items-center gap-8 px-8 py-10 text-center">
          <div className="w-56 rounded-3xl bg-white/[0.06] p-4 text-left font-mono text-[11px]" aria-hidden>
            {[['2,163', 0.4, false], ['2,164', 2.8, true], ['2,165', 0.2, false], ['2,177', 2.8, true], ['2,178', 0.7, false]].map(([row, ev, f]) => (
              <div key={row} className="flex items-center gap-3 py-1">
                <span className="w-10 text-white/50">{row}</span>
                <span className="h-1.5 flex-1 rounded-full bg-white/10"><span className="block h-full rounded-full" style={{ width: `${(ev / 2.8) * 100}%`, background: f ? 'var(--flag)' : '#4A494B' }} /></span>
              </div>
            ))}
          </div>
          <div>
            <h3 className="text-[26px] font-medium leading-tight">Row-level label check</h3>
            <p className="mx-auto mt-3 max-w-[30ch] text-white/70">Finds labels a model trained on your clean rows would not have given.</p>
          </div>
        </Rise>
        <Rise delay={0.1} className="flex flex-col items-center gap-8 border-t border-white/15 px-8 py-10 text-center lg:border-t-0">
          <div>
            <h3 className="text-[26px] font-medium leading-tight">Calibrated verdicts</h3>
            <p className="mx-auto mt-3 max-w-[30ch] text-white/70">Each verdict is a false-alarm rate, measured on clean batches.</p>
          </div>
          <div className="w-64 rounded-[2rem] border border-[#B9A3FF]/40 p-5" aria-hidden>
            {[['Clean', 'p > 0.05', 'var(--status-safe)'], ['Low risk', 'p ≤ 0.05', 'var(--status-info)'], ['Suspicious', 'p ≤ 0.01', 'var(--status-warn)'], ['Confirmed', 'p ≤ 1e-4', 'var(--status-critical)']].map(([n, p, c]) => (
              <div key={n} className="flex items-center justify-between py-1.5 text-sm">
                <span className="flex items-center gap-2"><span className="h-2 w-2 rounded-full" style={{ background: c }} />{n}</span>
                <span className="font-mono text-xs text-white/60">{p}</span>
              </div>
            ))}
          </div>
        </Rise>
        <div className="border-t border-white/15 lg:border-t-0">
          <Rise delay={0.2} className="flex items-center gap-6 px-8 py-10">
            <span className="icon-chip h-14 w-14 shrink-0"><FileCode2 className="h-6 w-6 text-[#EDC5FC]" strokeWidth={1.4} aria-hidden /></span>
            <div>
              <h3 className="text-[26px] font-medium leading-tight">Model-file scan</h3>
              <p className="mt-2 text-white/70">Says whether loading a file could run code. The file is read as bytes and never loaded.</p>
            </div>
          </Rise>
          <Rise delay={0.3} className="flex items-center gap-6 border-t border-white/15 px-8 py-10">
            <span className="icon-chip h-14 w-14 shrink-0"><Terminal className="h-6 w-6 text-[#EDC5FC]" strokeWidth={1.4} aria-hidden /></span>
            <div>
              <h3 className="text-[26px] font-medium leading-tight">CLI and API keys</h3>
              <p className="mt-2 text-white/70">Run the same checks from a terminal or a CI job and fail the build on a bad verdict.</p>
            </div>
          </Rise>
        </div>
      </div>
    </section>
  );
}

function MockData() {
  return (
    <div className="w-full max-w-[30rem] rounded-[28px] border border-white/10 bg-[#141314]/90 p-7 backdrop-blur" aria-hidden>
      <div className="flex items-baseline justify-between">
        <span className="text-2xl font-medium tracking-tight">Examination <span className="text-white/40">claims_batch.csv</span></span>
        <span className="rounded-lg bg-white/10 px-2 py-1 text-[10px]">Example run</span>
      </div>
      <div className="mt-5 grid grid-cols-3 gap-3">
        {[['Verdict', 'Suspicious', 'var(--status-warn)'], ['Flagged rows', '27', 'var(--flag)'], ['Rows checked', '900', '#fff']].map(([l, v, c]) => (
          <div key={l} className="rounded-2xl bg-white/[0.05] p-4">
            <div className="text-[11px] text-white/60">{l}</div>
            <div className="mt-2 text-2xl font-medium tracking-tight" style={{ color: c }}>{v}</div>
          </div>
        ))}
      </div>
      <div className="mt-4 space-y-2.5 rounded-2xl bg-white/[0.05] p-4">
        {[[1, 0.72, true], [1, 0.72, true], [0.89, 0.6, true], [0.5, 0.2, false]].map(([w, , f], i) => (
          <div key={i} className="h-2.5 rounded-full bg-white/10"><div className="h-full rounded-full" style={{ width: `${w * 100}%`, background: f ? '#DDB5FC' : '#4A494B' }} /></div>
        ))}
      </div>
    </div>
  );
}

function MockScan() {
  return (
    <div className="w-full max-w-[30rem] rounded-[28px] border border-white/10 bg-[#141314]/90 p-7 backdrop-blur" aria-hidden>
      <div className="flex items-baseline justify-between">
        <span className="text-2xl font-medium tracking-tight">Scan <span className="text-white/40">classifier_v3.pkl</span></span>
        <span className="rounded-lg bg-white/10 px-2 py-1 text-[10px]">Example run</span>
      </div>
      <div className="mt-5 text-5xl font-medium tracking-[-0.05em] text-[#FF2327]">Dangerous</div>
      <div className="mt-5 flex items-start gap-3 rounded-2xl bg-white/[0.05] p-4 text-sm">
        <span className="rounded-lg border border-[#FF2327] px-2 py-0.5 font-mono text-[10px] font-bold uppercase text-[#FF2327]">critical</span>
        <span className="text-white/80">pickle imports <span className="font-mono">os.system</span>, which can run code or access the system</span>
      </div>
      <div className="mt-3 font-mono text-[11px] leading-5 text-white/40">000000&nbsp;&nbsp;63 6f 73 0a 73 79 73 74 65 6d 0a 28 53 27 65 63</div>
    </div>
  );
}

function StackCard({ label, title, children, mock, top, cta }) {
  return (
    <div className="sticky" style={{ top }}>
      <Rise>
        <div className="edge-card">
          <div className="edge-inner stars relative grid items-center gap-10 px-8 py-14 md:px-16 lg:grid-cols-2 lg:px-[7.4rem] lg:py-[7.9rem]">
            <div aria-hidden className="pointer-events-none absolute inset-0" style={{ background: 'radial-gradient(60% 80% at 85% 70%, rgba(71,0,214,0.55), transparent 70%), radial-gradient(40% 50% at 70% 100%, rgba(185,163,255,0.35), transparent 70%)' }} />
            <div className="relative">
              <div className="rule-label text-sm">{label}</div>
              <h2 className="font-display mt-4 text-[clamp(2rem,3.4vw,3.2rem)] font-medium leading-[1.1]">{title}</h2>
              <p className="mt-5 max-w-[44ch] text-white/80">{children}</p>
              {cta}
            </div>
            <div className="relative flex justify-center lg:justify-end">{mock}</div>
          </div>
        </div>
      </Rise>
    </div>
  );
}

function Benchmarks() {
  return (
    <section id="benchmarks" className="px-6 py-28 md:px-20">
      <div className="mx-auto max-w-[1280px]">
        <Rise>
          <h2 className="font-display text-[clamp(2.4rem,6vw,5.6rem)] font-medium leading-[1.02]">Measured on data <span className="serif-accent">it never saw</span></h2>
          <p className="mt-6 max-w-[62ch] text-lg text-white/70">
            18 real OpenML datasets, six injected attacks. Tuned on 6, reported on the 12 held out. On real human label noise it matches cleanlab rather than beating it, and at 1% contamination every method struggles.
          </p>
        </Rise>
        <div className="mt-14 grid gap-6 lg:grid-cols-2">
          {BENCH.map((b, bi) => (
            <Rise key={b.label} delay={bi * 0.1}>
              <div className="panel h-full p-8">
                <h3 className="text-xl font-medium">{b.label}</h3>
                <p className="mt-1 text-sm text-white/50">{b.note}</p>
                <div className="mt-8 space-y-4">
                  {b.rows.map(([name, v], i) => (
                    <div key={name} className="grid grid-cols-[8.5rem_1fr_2.6rem] items-center gap-3 text-sm">
                      <span className={i === 0 ? 'font-medium' : 'text-white/60'}>{name}</span>
                      <div className="h-3 overflow-hidden rounded-full bg-white/[0.07]">
                        <motion.div
                          className="h-full rounded-full"
                          style={{ background: i === 0 ? '#DDB5FC' : '#4A494B' }}
                          initial={{ width: 0 }}
                          whileInView={{ width: `${v * 100}%` }}
                          viewport={{ once: true }}
                          transition={{ duration: 1.6, delay: 0.2 + i * 0.1, ease: EASE }}
                        />
                      </div>
                      <span className={`text-right font-mono ${i === 0 ? 'text-[#DDB5FC]' : 'text-white/60'}`}>{v.toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </Rise>
          ))}
        </div>
      </div>
    </section>
  );
}

/** Wavy striped form behind the wordmark, drawn as SVG lines. */
function Stripes() {
  const paths = useMemo(() => Array.from({ length: 46 }, (_, i) => {
    const x = 40 + i * 20;
    let d = `M ${x} 0`;
    for (let y = 0; y <= 520; y += 20) {
      const dx = Math.sin(y / 70 + i / 6) * 34 + Math.sin(y / 31 + i / 3) * 9;
      d += ` L ${(x + dx).toFixed(1)} ${y}`;
    }
    return d;
  }), []);
  return (
    <svg viewBox="0 0 1000 520" className="absolute left-1/2 top-1/2 h-[120%] w-[min(1100px,120%)] -translate-x-1/2 -translate-y-1/2" aria-hidden
      style={{ maskImage: 'radial-gradient(closest-side, #000 35%, transparent 100%)', WebkitMaskImage: 'radial-gradient(closest-side, #000 35%, transparent 100%)' }}>
      <defs>
        <linearGradient id="stripe" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0" stopColor="#4700D6" /><stop offset="0.6" stopColor="#7B4DFF" /><stop offset="1" stopColor="#B9A3FF" />
        </linearGradient>
      </defs>
      {paths.map((d, i) => <path key={i} d={d} fill="none" stroke="url(#stripe)" strokeWidth="7" strokeLinejoin="round" />)}
    </svg>
  );
}

export default function LandingPage() {
  const heroRef = useRef(null);
  const { scrollYProgress } = useScroll({ target: heroRef, offset: ['start start', 'end start'] });
  const heroY = useTransform(scrollYProgress, [0, 1], [0, 160]);
  const heroFade = useTransform(scrollYProgress, [0, 0.8], [1, 0]);

  return (
    <div id="top" className="relative overflow-x-clip bg-bgSurface text-white">
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[70] focus:rounded-lg focus:bg-white focus:px-3 focus:py-2 focus:text-ink-900">Skip to content</a>
      <PillNav />

      {/* Hero */}
      <header ref={heroRef} className="stars relative flex h-[100svh] min-h-[640px] flex-col items-center overflow-hidden px-7">
        <div aria-hidden className="pointer-events-none absolute inset-0" style={{ background: 'radial-gradient(55% 60% at 50% 38%, rgba(160,130,255,0.75), rgba(71,0,214,0.42) 38%, transparent 72%)' }} />
        <motion.div style={{ opacity: heroFade }} className="relative z-10 mt-[13vh] text-center">
          <motion.h1
            initial={{ opacity: 0, y: 30, letterSpacing: '0em' }}
            animate={{ opacity: 1, y: 0, letterSpacing: '-0.07em' }}
            transition={{ duration: 1.8, ease: EASE }}
            className="font-display text-[clamp(3.6rem,7vw,7rem)] font-medium leading-[1.2]"
          >
            veritas
          </motion.h1>
          <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 1.4, delay: 0.5 }} className="text-lg font-medium text-white/90">
            Training-data and model-file checks
          </motion.p>
        </motion.div>
        <motion.div style={{ y: heroY }} className="absolute inset-x-0 bottom-0 top-[30%]">
          <Suspense fallback={null}><Specimen3D className="absolute inset-0" loop /></Suspense>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1.6, delay: 0.9, ease: EASE }}
          className="absolute bottom-5 left-5 right-5 z-10 flex items-center justify-between gap-5 rounded-[22px] bg-[#1A1A1A]/80 p-2.5 pl-5 backdrop-blur-xl sm:left-auto sm:right-7"
        >
          <div className="leading-tight">
            <div className="font-medium">Every row on the slide</div>
            <div className="text-sm text-white/60">The suspicious ones glow</div>
          </div>
          <WhiteButton to="/signup">Create account</WhiteButton>
        </motion.div>
      </header>

      <main id="main">
        {/* Statement */}
        <section className="px-6 pb-10 pt-32 md:px-20 lg:pt-48">
          <h2 className="font-display mx-auto max-w-[1100px] text-center text-[clamp(2.6rem,7.4vw,7rem)] font-medium leading-[1.12]">
            <Rise>Trust<IconPill icon={ShieldCheck} />Your</Rise>
            <Rise delay={0.08}>Data<IconPill icon={ScanLine} /><span className="serif-accent">Verify</span></Rise>
            <Rise delay={0.16}>Your<IconPill icon={FileCode2} />Models</Rise>
          </h2>
          <ScrollFill
            className="mt-24 max-w-[30ch] text-[clamp(1.4rem,2.2vw,2rem)] font-medium leading-[1.25] tracking-[-0.03em] md:ml-[8%] lg:mt-36"
            text="Poisoned rows pass every ordinary test. Veritas compares each incoming batch with rows you already trust, ranks what looks wrong, and tells you how often a clean batch would look this bad. It is a triage aid, not a certificate."
          />
        </section>

        <HowItWorks />
        <Features />

        {/* Stacking product cards */}
        <section className="space-y-10 px-6 py-24 md:px-20">
          <StackCard
            top="6rem"
            label="For ML engineers"
            title={<>Audit the batch before you train on it</>}
            mock={<MockData />}
            cta={<WhiteButton to="/signup" className="mt-8">Examine a batch</WhiteButton>}
          >
            Upload a labelled CSV. Veritas returns one verdict, the rows to review first, and the evidence for both checks. Reliable batch verdicts need about 5% contamination; below that, work from the ranked rows.
          </StackCard>
          <StackCard
            top="7.5rem"
            label="For security analysts"
            title={<>Know what a model file does before anything loads it</>}
            mock={<MockScan />}
            cta={<WhiteButton to="/signup" className="mt-8">Scan a model file</WhiteButton>}
          >
            Pickle, joblib, PyTorch, safetensors, ONNX, NumPy and GGUF files are read in an isolated process. You get the imports the file would perform and a plain verdict: dangerous, needs review, no code execution found, or safe format.
          </StackCard>
        </section>

        <Benchmarks />

        {/* Name */}
        <section className="relative flex h-[80vh] min-h-[420px] items-center justify-center overflow-hidden px-6">
          <Stripes />
          <Rise className="relative text-center">
            <p className="font-medium text-white/90">Review these rows first. Do not load this file.</p>
            <div className="font-display text-[clamp(4rem,13vw,13rem)] font-medium leading-none tracking-[-0.07em]">veritas</div>
          </Rise>
        </section>
      </main>

      {/* Footer */}
      <footer className="relative overflow-hidden px-6 pt-16 md:px-11">
        <div aria-hidden className="pointer-events-none absolute inset-x-0 bottom-0 h-[70%]" style={{ background: 'radial-gradient(60% 90% at 55% 100%, rgba(221,181,252,0.9), rgba(123,77,255,0.75) 30%, rgba(71,0,214,0.5) 55%, transparent 80%)' }} />
        <div className="relative flex flex-col gap-10 md:flex-row md:items-start md:justify-between">
          <div>
            <div className="text-sm font-medium text-white/80">Start with one file</div>
            <Link to="/signup" className="font-display mt-2 block text-[clamp(2rem,3.4vw,3.2rem)] font-medium leading-tight hover:text-[#EDC5FC]">Create an account</Link>
          </div>
          <div>
            <div className="text-xl font-medium">Already checking data?</div>
            <p className="mt-1 text-sm text-white/80">Pick up where your last examination left off.</p>
            <WhiteButton to="/login" className="mt-4">Sign in</WhiteButton>
          </div>
        </div>
        <div className="relative mt-12 flex flex-wrap justify-between gap-4 border-b border-white/15 pb-9 text-xl font-medium">
          <a href="#how" className="hover:text-[#EDC5FC]">How it works</a>
          <a href="#benchmarks" className="hover:text-[#EDC5FC]">Benchmarks</a>
          <Link to="/login" className="hover:text-[#EDC5FC]">Sign in</Link>
          <Link to="/admin/login" className="hover:text-[#EDC5FC]">Admin</Link>
        </div>
        <div className="font-display relative select-none pt-6 text-center text-[clamp(3.4rem,15.6vw,15rem)] font-medium uppercase leading-none tracking-[-0.06em]" aria-hidden>Veritas</div>
        <div className="relative flex flex-wrap justify-between gap-3 pb-6 pt-2 text-xs text-white/90">
          <span>Veritas is a triage aid. It is not a certification, proof of an attack, or a compliance claim.</span>
          <span>Benchmark numbers: docs/model-card.md</span>
        </div>
      </footer>
    </div>
  );
}
