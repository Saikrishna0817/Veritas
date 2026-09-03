import { useState, useEffect } from 'react';
import { api } from '../services/api';
import { motion, AnimatePresence } from 'framer-motion';
import Marquee from 'react-fast-marquee';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ComposedChart
} from 'recharts';
import { Shield, AlertTriangle, CheckCircle, Play, Loader, Eye, ChevronRight } from 'lucide-react';
import Tactile3DHero from '../components/Tactile3DHero';

// ── Sub-components for Metrics ────────────────────────────────────────────────

function MetricCard({ label, value, unit = '', color = '#E8622C', sublabel }) {
  return (
    <div className="bg-slateLighter p-5 relative overflow-hidden rounded-2xl border border-white/5">
      <div className="absolute top-0 left-0 w-1 h-full" style={{ backgroundColor: color }} />
      <div className="text-xs uppercase tracking-widest text-textMutedDark mb-2">{label}</div>
      <div className="text-3xl font-extrabold tracking-tight text-textLight flex items-baseline gap-1 font-mono">
        <span style={{ color }}>{value}</span>
        {unit && <span className="text-sm text-textMutedDark font-normal">{unit}</span>}
      </div>
      {sublabel && <div className="text-xs text-textMutedDark mt-1.5 font-medium">{sublabel}</div>}
    </div>
  );
}

// ── Expandable Feature Cards (How SPECTRA Helps) ─────────────────────────────

const features = [
  {
    id: 'f1',
    title: 'Model Scanning',
    desc: 'Deep inspection of neural network weights for embedded backdoors.',
    icon: '🧠',
  },
  {
    id: 'f2',
    title: 'Poison Forensics',
    desc: 'Identify specific training samples causing statistical shifts.',
    icon: '🔍',
  },
  {
    id: 'f3',
    title: 'Live Threat Defense',
    desc: 'Active interception of adversarial inputs and data poisoning attacks.',
    icon: '🛡️',
  },
];

function ExpandableFeatures() {
  const [expandedId, setExpandedId] = useState('f2');

  return (
    <div className="flex flex-col md:flex-row gap-4 w-full h-80">
      {features.map((feat) => {
        const isExpanded = expandedId === feat.id;
        return (
          <motion.div
            key={feat.id}
            layout
            onClick={() => setExpandedId(feat.id)}
            className={`cursor-pointer rounded-3xl overflow-hidden p-6 relative border border-white/5 flex flex-col justify-end
              ${isExpanded ? 'bg-softYellow text-slateDark flex-[3]' : 'bg-slateLighter text-textLight flex-1 hover:bg-white/5'}
            `}
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
          >
            <motion.div layout="position" className="text-4xl mb-4">
              {feat.icon}
            </motion.div>
            <motion.h3 layout="position" className="text-xl font-bold tracking-tight mb-2 whitespace-nowrap">
              {feat.title}
            </motion.h3>
            <AnimatePresence>
              {isExpanded && (
                <motion.p
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="text-slateDark/80 text-sm leading-relaxed"
                >
                  {feat.desc}
                </motion.p>
              )}
            </AnimatePresence>
          </motion.div>
        );
      })}
    </div>
  );
}

// ── Main Dashboard ────────────────────────────────────────────────────────────

export default function Dashboard({ wsEvents = [] }) {
  const [demoResult, setDemoResult] = useState(null);
  const [trustScore, setTrustScore] = useState(null);
  const [timeline, setTimeline] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [analystMode, setAnalystMode] = useState(false);

  useEffect(() => {
    api.getTrustScore().then(setTrustScore).catch(() => {});
    api.getAttackTimeline().then((d) => setTimeline(d.timeline || [])).catch(() => {});
  }, []);

  useEffect(() => {
    if (wsEvents.some((e) => e.event === 'attack_confirmed')) {
      api.getTrustScore().then(setTrustScore).catch(() => {});
    }
  }, [wsEvents]);

  const handleRunDemo = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.runDemo();
      setDemoResult(result);
      const ts = await api.getTrustScore();
      setTrustScore(ts);
      setTimeline(result.timeline || []);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full">
      {/* ── 1. Hero Section (Light Theme) ────────────────────────────────────── */}
      <section className="bg-cream pt-20 pb-24 px-6 md:px-12 rounded-t-[28px]">
        <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          <div className="space-y-8 relative z-10">
            <h1 className="text-5xl md:text-7xl font-black tracking-tighter text-textDark leading-[1.05]">
              Uncompromising AI Security.
            </h1>
            <p className="text-lg md:text-xl text-textDark/80 max-w-lg leading-relaxed">
              SPECTRA/VERITAS provides enterprise-grade data poisoning detection, proxy impact analysis, and continuous model security.
            </p>
            <div className="flex items-center gap-6">
              <button
                onClick={handleRunDemo}
                disabled={loading}
                className="group flex items-center justify-center gap-2 bg-burntOrange text-white px-8 py-4 rounded-full font-bold text-lg hover:bg-[#d95625] transition-colors disabled:opacity-50"
              >
                {loading ? <Loader className="w-5 h-5 animate-spin" /> : 'Run Demo'}
                {!loading && <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />}
              </button>
              
              {/* Analyst Mode Toggle */}
              <label className="flex items-center gap-3 cursor-pointer group">
                <div className="relative">
                  <input type="checkbox" className="sr-only" checked={analystMode} onChange={() => setAnalystMode(!analystMode)} />
                  <div className={`block w-14 h-8 rounded-full transition-colors ${analystMode ? 'bg-frameBlack' : 'bg-black/10'}`}></div>
                  <div className={`dot absolute left-1 top-1 bg-white w-6 h-6 rounded-full transition-transform ${analystMode ? 'transform translate-x-6' : ''}`}></div>
                </div>
                <div className="text-sm font-bold text-textDark group-hover:text-burntOrange transition-colors">
                  Analyst Mode
                </div>
              </label>
            </div>
          </div>
          
          <div className="relative h-[400px] w-full flex items-center justify-center">
             {/* Realistic 3D Element Placeholder / Component */}
             <Tactile3DHero intensity={analystMode ? 2.0 : 0.5} />
          </div>
        </div>
      </section>

      {/* ── 2. How Can SPECTRA Help? (Dark Theme) ─────────────────────────────── */}
      <section className="bg-slateDark py-24 px-6 md:px-12 text-textLight">
        <div className="max-w-7xl mx-auto space-y-12">
          <div className="flex justify-between items-end">
            <h2 className="text-4xl md:text-5xl font-black tracking-tight max-w-lg">
              How Can SPECTRA Help?
            </h2>
            <p className="text-textMutedDark max-w-sm text-right hidden md:block">
              Interactive threat modeling and live detection tools designed for advanced security operations.
            </p>
          </div>
          
          <ExpandableFeatures />
        </div>
      </section>

      {/* ── 3. Our Partners (Dark Theme Marquee) ──────────────────────────────── */}
      <section className="bg-slateDark border-t border-white/5 py-12 overflow-hidden">
        <Marquee speed={40} gradient={true} gradientColor={[13, 13, 15]} gradientWidth={100} autoFill>
          {['NVIDIA Inception', 'Microsoft Security', 'Google Cloud', 'AWS Partner', 'OpenAI', 'HuggingFace'].map((partner, i) => (
            <div key={i} className="mx-8 text-2xl font-black tracking-tighter text-textLight/20 uppercase hover:text-softYellow transition-colors duration-300">
              {partner}
            </div>
          ))}
        </Marquee>
      </section>

      {/* ── 4. Dashboard Metrics (Dark Theme) ─────────────────────────────────── */}
      <section className="bg-slateDark py-16 px-6 md:px-12 rounded-b-[28px] border-t border-white/5">
         <div className="max-w-7xl mx-auto space-y-8">
            <h3 className="text-2xl font-bold text-textLight tracking-tight">Live Threat Intelligence</h3>
            
            {error && (
              <div className="p-4 bg-danger/10 border border-danger/30 text-danger text-sm font-mono rounded-xl">
                ⚠️ Connection Error: {error}
              </div>
            )}

            {trustScore && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
                <MetricCard label="Data Quality" value={trustScore.dataset_trust.data_quality} color="#4DE8FF" />
                <MetricCard label="Poison Risk" value={trustScore.dataset_trust.poison_risk} unit="%" color="#E8622C" />
                <MetricCard label="Behavioral Trust" value={trustScore.dataset_trust.behavioral_trust} color="#F2E85C" />
                <div className="bg-slateLighter p-5 relative overflow-hidden rounded-2xl border border-white/5">
                  <div className="absolute top-0 left-0 w-1 h-full bg-softYellow" />
                  <div className="text-xs uppercase tracking-widest text-textMutedDark mb-2">Model Grade</div>
                  <div className="text-5xl font-extrabold tracking-tight text-softYellow font-mono">
                    {trustScore.model_safety.grade}
                  </div>
                </div>
              </div>
            )}

            {timeline.length > 0 && (
              <div className="bg-slateLighter p-6 lg:p-8 rounded-3xl border border-white/5 mt-8">
                <div className="text-sm font-bold text-textLight mb-6">Threat Timeline Analysis</div>
                <ResponsiveContainer width="100%" height={300}>
                  <ComposedChart data={timeline.slice(-24)} margin={{ top: 10, right: 30, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="timestamp" tick={{ fill: '#9CA3AF', fontSize: 11 }} tickFormatter={(v) => v.slice(11, 16)} />
                    <YAxis yAxisId="left" tick={{ fill: '#9CA3AF', fontSize: 11 }} domain={[0.7, 1.0]} />
                    <YAxis yAxisId="right" orientation="right" tick={{ fill: '#9CA3AF', fontSize: 11 }} />
                    <Tooltip contentStyle={{ backgroundColor: '#121214', borderColor: 'rgba(255,255,255,0.1)' }} />
                    <Bar yAxisId="right" dataKey="poison_count" fill="rgba(232, 98, 44, 0.25)" stroke="#E8622C" name="Poison Count" radius={[4, 4, 0, 0]} />
                    <Line yAxisId="left" type="monotone" dataKey="accuracy" stroke="#F2E85C" strokeWidth={2.5} dot={false} name="Accuracy" />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            )}
         </div>
      </section>
    </div>
  );
}
