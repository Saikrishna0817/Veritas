import { useState, useEffect } from 'react';
import { api } from '../services/api';
import { AnimatePresence } from 'framer-motion';
import Marquee from 'react-fast-marquee';
import { Line, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ComposedChart
} from 'recharts';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { ChevronRight, Loader, BrainCircuit, FileText, ShieldAlert, User } from 'lucide-react';
import Tactile3DHero from '../components/Tactile3DHero';

// ── Sub-components for Metrics ────────────────────────────────────────────────

function MetricCard({ label, value, unit = '', color = 'var(--red-primary)', sublabel }) {
  return (
    <div className="bg-bgPanel p-5 relative overflow-hidden rounded-[20px] border border-borderHairline group hover:bg-bgPanelRaised transition-colors">
      <div className="absolute top-0 left-0 w-1 h-full" style={{ backgroundColor: color }} />
      <div className="text-[13px] font-bold tracking-[0.15em] text-textMuted mb-2">{label}</div>
      <div className="text-[40px] font-mono font-extrabold tracking-tight text-textPrimary flex items-baseline gap-1">
        <span style={{ color }}>{value}</span>
        {unit && <span className="text-sm text-textMuted font-normal">{unit}</span>}
      </div>
      {sublabel && <div className="text-xs text-textMuted mt-1.5 font-medium">{sublabel}</div>}
    </div>
  );
}

// ── Expandable Feature Cards (How SPECTRA Helps) ─────────────────────────────

const features = [
  {
    id: 'f1',
    title: 'Detect Data & Model Poisoning',
    desc: 'Upload a dataset or a trained .pkl model and run it through all 5 detection layers to catch poisoned samples before they ship.',
    icon: <BrainCircuit className="w-8 h-8 text-white" />,
  },
  {
    id: 'f2',
    title: 'Court-Admissible Forensic Reports',
    desc: 'Generate evidence packages mapped to NIST AI RMF and EU AI Act compliance, ready for audit or litigation.',
    icon: <FileText className="w-8 h-8 text-white" />,
  },
  {
    id: 'f3',
    title: '24/7 Blue Team Operations',
    desc: 'A live SOC dashboard with human-in-the-loop review, red-team resilience testing, and real-time threat-level status.',
    icon: <ShieldAlert className="w-8 h-8 text-white" />,
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
            className={`cursor-pointer rounded-[20px] overflow-hidden p-6 relative border border-borderHairline flex flex-col justify-end
              ${isExpanded ? 'bg-redDim/20 border-redPrimary/30 flex-[3]' : 'bg-bgPanel text-textPrimary flex-1 hover:bg-bgPanelRaised'}
            `}
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
          >
            <motion.div layout="position" className="mb-4 w-12 h-12 rounded-[10px] border border-borderHairline flex items-center justify-center bg-redPrimary">
              {feat.icon}
            </motion.div>
            <motion.h3 layout="position" className="text-xl font-display font-bold tracking-tight mb-2 whitespace-nowrap text-textPrimary">
              {feat.title}
            </motion.h3>
            <AnimatePresence>
              {isExpanded && (
                <motion.p
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="text-textSecondary text-[18px] leading-relaxed"
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
  const { user } = useAuth();
  const navigate = useNavigate();
  const [trustScore, setTrustScore] = useState(null);
  const [timeline, setTimeline] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [analystMode, setAnalystMode] = useState(false);

  useEffect(() => {
    if (user?.role === 'admin') {
      navigate('/admin', { replace: true });
    }
  }, [user, navigate]);

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
      {/* ── 1. Hero Section (Dark Theme) ────────────────────────────────────── */}
      <section className="bg-transparent pt-16 pb-12 px-6 md:px-12 rounded-t-[28px]" style={{ overflow: 'visible' }}>
        <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-0 items-center" style={{ overflow: 'visible', position: 'relative' }}>
          <div className="space-y-6 relative z-10 py-12">
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 font-mono text-xs font-bold tracking-wider uppercase">
              <User className="w-3.5 h-3.5" /> USER DASHBOARD
            </div>
            <h1 className="text-[80px] md:text-[96px] font-display font-bold tracking-tighter text-textPrimary leading-[1.05]">
              Uncompromising<br/>
              <span className="text-redPrimary border-b-[4px] border-redPrimary">AI Security.</span>
            </h1>

            <p className="text-[18px] md:text-[19px] text-textSecondary max-w-lg leading-relaxed">
              SPECTRA/VERITAS provides enterprise-grade data poisoning detection, proxy impact analysis, and continuous model security.
            </p>
            <div className="flex items-center gap-6">
              <button
                onClick={handleRunDemo}
                disabled={loading}
                className="group flex items-center justify-center gap-2 bg-redPrimary text-white px-8 py-4 rounded-xl font-bold text-lg hover:bg-redBright hover:shadow-red-glow transition-all disabled:opacity-50"
              >
                {loading ? <Loader className="w-5 h-5 animate-spin" /> : 'Run Demo'}
                {!loading && <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />}
              </button>
              
              {/* Analyst Mode Toggle */}
              <label className="flex items-center gap-3 cursor-pointer group">
                <div className="relative">
                  <input type="checkbox" className="sr-only" checked={analystMode} onChange={() => setAnalystMode(!analystMode)} />
                  <div className={`block w-14 h-8 rounded-full transition-colors ${analystMode ? 'bg-redPrimary' : 'bg-bgPanelRaised'}`}></div>
                  <div className={`dot absolute left-1 top-1 w-6 h-6 rounded-full transition-transform ${analystMode ? 'transform translate-x-6 bg-white' : 'bg-textSecondary'}`}></div>
                </div>
                <div className="text-sm font-bold text-textPrimary group-hover:text-redPrimary transition-colors">
                  Analyst Mode
                </div>
              </label>
            </div>
          </div>
          
          {/* 3D Animation — absolutely positioned, oversized so rings are never clipped */}
          <div style={{
            position: 'absolute',
            right: '-15%',
            top: '50%',
            transform: 'translateY(-50%)',
            width: '75vw',
            height: '900px',
            maxWidth: '1000px',
            overflow: 'visible',
            pointerEvents: 'none',
            zIndex: 0
          }}>
             <Tactile3DHero intensity={analystMode ? 2.0 : 0.5} />
          </div>
        </div>
      </section>

      {/* ── 2. How Can SPECTRA Help? (Dark Theme) ─────────────────────────────── */}
      <section className="bg-bgVoid py-24 px-6 md:px-12 text-textPrimary">
        <div className="max-w-7xl mx-auto space-y-12">
          <div className="flex justify-between items-end">
            <h2 className="text-[56px] md:text-[64px] font-display font-bold tracking-tight max-w-lg leading-tight">
              How Can SPECTRA Help?
            </h2>
            <p className="text-[18px] text-textSecondary max-w-sm text-right hidden md:block">
              Interactive threat modeling and live detection tools designed for advanced security operations.
            </p>
          </div>
          
          <ExpandableFeatures />
        </div>
      </section>

      {/* ── 3. Our Partners (Dark Theme Marquee) ──────────────────────────────── */}
      <section className="bg-bgVoid border-t border-borderHairline py-12 overflow-hidden">
        <Marquee speed={40} gradient={true} gradientColor={[0, 0, 0]} gradientWidth={100} autoFill>
          {['NIST AI RMF', 'EU AI Act', 'MITRE ATLAS', 'ISO/IEC 42001', 'OWASP Top 10 for LLMs'].map((partner, i) => (
            <div key={i} className="mx-12 text-xl font-bold tracking-wider text-textMuted uppercase hover:text-textPrimary transition-colors duration-300">
              {partner}
            </div>
          ))}
        </Marquee>
      </section>

      {/* ── 4. Dashboard Metrics (Dark Theme) ─────────────────────────────────── */}
      <section className="bg-bgSurface py-16 px-6 md:px-12 rounded-b-[28px] border-t border-borderHairline">
         <div className="max-w-7xl mx-auto space-y-8">
            <h3 className="text-[56px] font-display font-bold text-textPrimary tracking-tight">Live Threat Intelligence</h3>
            
            {error && (
              <div className="p-4 bg-redDim border border-redPrimary text-textPrimary text-[18px] font-mono rounded-xl shadow-red-glow">
                ⚠️ Connection Error: {error}
              </div>
            )}

            {trustScore && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
                <MetricCard label="Data Quality" value={trustScore.dataset_trust.data_quality} color="var(--status-safe)" />
                <MetricCard label="Poison Risk" value={trustScore.dataset_trust.poison_risk} unit="%" color="var(--red-primary)" />
                <MetricCard label="Behavioral Trust" value={trustScore.dataset_trust.behavioral_trust} color="var(--status-warn)" />
                <div className="bg-bgPanel p-5 relative overflow-hidden rounded-[20px] border border-borderHairline group hover:bg-bgPanelRaised transition-colors">
                  <div className="absolute top-0 left-0 w-1 h-full bg-redPrimary" />
                  <div className="text-[13px] font-bold tracking-[0.15em] text-textMuted mb-2">Model Grade</div>
                  <div className="text-[40px] font-extrabold tracking-tight text-textPrimary font-mono">
                    {trustScore.model_safety.grade}
                  </div>
                </div>
              </div>
            )}

            {timeline.length > 0 && (
              <div className="bg-bgPanel p-6 lg:p-8 rounded-[20px] border border-borderHairline mt-8 hover:bg-bgPanelRaised transition-colors">
                <div className="text-[13px] font-bold tracking-[0.15em] text-textMuted mb-6">Threat Timeline Analysis</div>
                <ResponsiveContainer width="100%" height={300}>
                  <ComposedChart data={timeline.slice(-24)} margin={{ top: 10, right: 30, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                    <XAxis dataKey="timestamp" tick={{ fill: 'var(--text-muted)', fontSize: 13, fontFamily: 'JetBrains Mono' }} tickFormatter={(v) => v.slice(11, 16)} />
                    <YAxis yAxisId="left" tick={{ fill: 'var(--text-muted)', fontSize: 13, fontFamily: 'JetBrains Mono' }} domain={[0.7, 1.0]} />
                    <YAxis yAxisId="right" orientation="right" tick={{ fill: 'var(--text-muted)', fontSize: 13, fontFamily: 'JetBrains Mono' }} />
                    <Tooltip contentStyle={{ backgroundColor: 'var(--bg-panel)', borderColor: 'var(--border-hairline)', borderRadius: '12px' }} />
                    <Bar yAxisId="right" dataKey="poison_count" fill="var(--red-glow)" stroke="var(--red-primary)" name="Poison Count" radius={[4, 4, 0, 0]} />
                    <Line yAxisId="left" type="monotone" dataKey="accuracy" stroke="var(--status-safe)" strokeWidth={2.5} dot={false} name="Accuracy" />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            )}
         </div>
      </section>
    </div>
  );
}
