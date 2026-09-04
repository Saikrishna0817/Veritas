import { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import { 
    ShieldAlert, AlertTriangle, Shield, CheckCircle, 
    Lock, Zap, ClipboardList, Target, ShieldCheck, 
    Check, User, Bot, RefreshCw, Radio, BookOpen, Activity 
} from 'lucide-react';

// ─── Constants ────────────────────────────────────────────────────────────────

const THREAT_CONFIG = {
    CRITICAL: { color: 'var(--status-critical)', bg: 'rgba(228,36,43,0.1)', border: 'rgba(228,36,43,0.4)', label: 'CRITICAL', icon: ShieldAlert, pulse: true },
    ELEVATED: { color: 'var(--status-warn)', bg: 'rgba(242,184,75,0.1)', border: 'rgba(242,184,75,0.4)', label: 'ELEVATED', icon: AlertTriangle, pulse: true },
    GUARDED: { color: '#3b82f6', bg: 'rgba(59,130,246,0.1)', border: 'rgba(59,130,246,0.4)', label: 'GUARDED', icon: Shield, pulse: false },
    NOMINAL: { color: 'var(--status-safe)', bg: 'rgba(61,220,132,0.1)', border: 'rgba(61,220,132,0.4)', label: 'NOMINAL', icon: CheckCircle, pulse: false },
};

const ATTACK_COLORS = {
    label_flip: 'var(--status-warn)', backdoor: 'var(--status-critical)',
    clean_label: '#a855f7', gradient_poisoning: '#06b6d4', boiling_frog: 'var(--status-safe)',
};

const SEVERITY_CONFIG = {
    critical: { color: 'var(--status-critical)', bg: 'rgba(228,36,43,0.1)', label: 'CRITICAL' },
    high: { color: 'var(--status-warn)', bg: 'rgba(242,184,75,0.1)', label: 'HIGH' },
    medium: { color: '#3b82f6', bg: 'rgba(59,130,246,0.1)', label: 'MEDIUM' },
    info: { color: 'var(--text-muted)', bg: 'var(--bg-void)', label: 'INFO' },
};

// ─── Sub-components ───────────────────────────────────────────────────────────

function StatCard({ icon: Icon, label, value, sub, color = 'var(--red-primary)' }) {
    return (
        <div className="bg-bgPanel border rounded-xl p-5 flex-1 min-w-[120px] border-borderHairline hover:border-redPrimary/30 transition-colors animate-flipIn" style={{ animationDelay: `${Math.random() * 0.4}s` }}>
            <div className="mb-3"><Icon className="w-6 h-6" style={{ color }} /></div>
            <div className="text-[40px] font-mono font-black leading-none tracking-tighter" style={{ color }}>{value}</div>
            <div className="text-xs text-textSecondary mt-3 font-bold uppercase tracking-widest">{label}</div>
            {sub && <div className="text-[10px] text-textMuted mt-1 font-mono uppercase">{sub}</div>}
        </div>
    );
}

function SectionHeader({ title, icon: Icon }) {
    return (
        <div className="text-xs text-redPrimary uppercase tracking-widest mb-5 font-bold pb-3 border-b border-borderHairline flex items-center gap-2">
            {Icon && <Icon className="w-4 h-4" />}
            {title}
        </div>
    );
}

function ThreatBanner({ status }) {
    if (!status) return null;
    const cfg = THREAT_CONFIG[status.threat_level] || THREAT_CONFIG.NOMINAL;
    const ThreatIcon = cfg.icon;
    
    return (
        <div className="rounded-2xl p-6 md:p-8 mb-8 flex flex-col md:flex-row items-center gap-6 relative overflow-hidden border" style={{ background: cfg.bg, borderColor: cfg.border }}>
            {cfg.pulse && (
                <div className="absolute inset-0 bg-redPrimary/5 pointer-events-none animate-pulse" style={{ background: `radial-gradient(ellipse at right, ${cfg.color}15 0%, transparent 70%)` }} />
            )}
            <div className={cfg.pulse ? "drop-shadow-red-glow" : ""}>
                <ThreatIcon className="w-14 h-14" style={{ color: cfg.color }} />
            </div>
            <div className="flex-1 text-center md:text-left z-10">
                <div className="flex flex-col md:flex-row items-center gap-3 mb-2">
                    <span className="text-2xl font-mono font-black tracking-widest uppercase" style={{ color: cfg.color }}>
                        THREAT LEVEL: {cfg.label}
                    </span>
                    {cfg.pulse && (
                        <span className="text-[10px] px-3 py-1 rounded-full border uppercase tracking-widest font-bold animate-pulse" style={{ background: `${cfg.color}20`, color: cfg.color, borderColor: `${cfg.color}50` }}>
                            LIVE
                        </span>
                    )}
                </div>
                <div className="text-sm font-mono text-textSecondary flex flex-wrap items-center justify-center md:justify-start gap-3 mt-3">
                    <span>Verdict: <strong style={{ color: cfg.color }}>{status.current_verdict}</strong></span>
                    <span className="text-borderHairline">|</span>
                    <span>Suspicion: <strong style={{ color: cfg.color }}>{Math.round((status.suspicion_score || 0) * 100)}%</strong></span>
                    <span className="text-borderHairline">|</span>
                    <span>Mode: <strong className="text-textPrimary uppercase">{status.defense_mode}</strong></span>
                </div>
            </div>
            <div className="text-center md:text-right z-10">
                <div className="text-[10px] text-textMuted uppercase tracking-widest font-bold">Updated</div>
                <div className="text-xs text-textSecondary font-mono mt-1">
                    {status.updated_at ? new Date(status.updated_at).toLocaleTimeString() : '—'}
                </div>
            </div>
        </div>
    );
}

function DefenseStats({ status }) {
    if (!status) return null;
    const rt = status.red_team || {};
    return (
        <div className="flex flex-wrap gap-4 mb-8">
            <StatCard icon={Lock} label="Total Quarantined" value={status.total_quarantined || 0} color="var(--status-critical)" />
            <StatCard icon={Zap} label="Defense Actions" value={status.n_defense_actions || 0} color="var(--status-warn)" />
            <StatCard icon={ClipboardList} label="HITL Queue" value={status.hitl_queue_depth || 0} sub="pending analyst review" color="#a855f7" />
            <StatCard icon={Target} label="Red Team Tests" value={rt.total_simulations || 0} sub={`${rt.attacks_caught || 0} caught`} color="#06b6d4" />
            <StatCard icon={ShieldCheck} label="Resilience" value={`${rt.resilience_pct ?? 100}%`} sub="attack catch rate" color="var(--status-safe)" />
        </div>
    );
}

function HITLQueue({ cases, onDecide }) {
    if (!cases || cases.length === 0) {
        return (
            <div className="p-8 text-center text-textMuted text-sm font-mono border-2 border-dashed border-borderHairline rounded-xl flex flex-col items-center">
                <CheckCircle className="w-8 h-8 mb-3 opacity-20" />
                No pending cases — queue is clear
            </div>
        );
    }
    return (
        <div className="flex flex-col gap-3">
            {cases.map(c => (
                <div key={c.case_id} className="bg-bgVoid border border-statusWarn/30 rounded-xl p-4 flex flex-col md:flex-row md:items-center gap-4">
                    <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                            <span className="text-sm font-mono font-bold text-textPrimary">{c.case_id?.slice(0, 8)}…</span>
                            <span className="text-[10px] px-2 py-0.5 rounded-md bg-statusWarn/20 text-statusWarn border border-statusWarn/30 font-bold uppercase tracking-wider">PENDING</span>
                        </div>
                        <div className="text-xs font-mono text-textSecondary mb-2 flex gap-3 flex-wrap items-center">
                            <span>Suspicion: <strong className="text-statusWarn">{Math.round((c.suspicion_score || 0) * 100)}%</strong></span>
                            <span className="text-borderHairline">|</span>
                            <span>Samples: <strong>{c.n_samples}</strong></span>
                            <span className="text-borderHairline">|</span>
                            <span>Attack: <strong className="capitalize" style={{ color: ATTACK_COLORS[c.evidence_summary?.attack_type] || 'var(--text-muted)' }}>
                                {(c.evidence_summary?.attack_type || 'unknown').replace(/_/g, ' ')}
                            </strong></span>
                        </div>
                        <div className="text-[11px] font-mono text-textMuted flex gap-3">
                            <span>KL Div: {c.evidence_summary?.kl_divergence?.toFixed(3) || '—'}</span>
                            <span className="text-borderHairline">|</span>
                            <span>Causal: {c.evidence_summary?.causal_effect?.toFixed(3) || '—'}</span>
                        </div>
                    </div>
                    <div className="flex gap-2 flex-shrink-0">
                        <button onClick={() => onDecide(c.case_id, 'approve_quarantine')}
                            className="flex items-center gap-2 px-4 py-2 rounded-lg border border-statusCritical/30 bg-statusCritical/10 text-statusCritical text-xs font-bold hover:bg-statusCritical/20 transition-colors uppercase tracking-wider">
                            <Lock className="w-4 h-4" /> Quarantine
                        </button>
                        <button onClick={() => onDecide(c.case_id, 'mark_safe')}
                            className="flex items-center gap-2 px-4 py-2 rounded-lg border border-statusSafe/30 bg-statusSafe/10 text-statusSafe text-xs font-bold hover:bg-statusSafe/20 transition-colors uppercase tracking-wider">
                            <Check className="w-4 h-4" /> Safe
                        </button>
                    </div>
                </div>
            ))}
        </div>
    );
}

function IncidentLog({ incidents }) {
    const [expanded, setExpanded] = useState(null);
    if (!incidents || incidents.length === 0) {
        return (
            <div className="p-8 text-center text-textMuted text-sm font-mono border-2 border-dashed border-borderHairline rounded-xl flex flex-col items-center">
                <Radio className="w-8 h-8 mb-3 opacity-20" />
                No incidents logged yet.
            </div>
        );
    }
    return (
        <div className="flex flex-col gap-2">
            {incidents.map((inc, i) => {
                const sev = SEVERITY_CONFIG[inc.severity] || SEVERITY_CONFIG.info;
                const isOpen = expanded === i;
                return (
                    <div key={i} onClick={() => setExpanded(isOpen ? null : i)} className={`p-3 rounded-lg cursor-pointer transition-all border ${isOpen ? 'bg-bgVoid border-borderHairline' : 'bg-transparent border-transparent hover:bg-bgVoid/50'}`}>
                        <div className="flex items-center gap-3">
                            <span className="text-[10px] font-bold px-2 py-0.5 rounded border uppercase tracking-wider whitespace-nowrap" style={{ background: sev.bg, color: sev.color, borderColor: `${sev.color}40` }}>
                                {sev.label}
                            </span>
                            <span className="text-xs font-mono text-textPrimary flex-1 capitalize flex items-center gap-2">
                                {inc.type === 'human_decision' ? <User className="w-3 h-3 text-textMuted" /> : <Bot className="w-3 h-3 text-textMuted" />}
                                {(inc.action || '').replace(/_/g, ' ')}
                                {inc.samples_affected ? <span className="text-textMuted">— {inc.samples_affected} samples</span> : null}
                                {inc.reviewer ? <span className="text-redPrimary">by {inc.reviewer}</span> : null}
                            </span>
                            <span className="text-[10px] text-textMuted font-mono whitespace-nowrap">
                                {inc.timestamp ? new Date(inc.timestamp).toLocaleTimeString() : '—'}
                            </span>
                        </div>
                        {isOpen && inc.reason && (
                            <div className="mt-3 text-xs font-mono text-textSecondary pl-3 border-l-2 border-redPrimary/50 py-1 bg-redPrimary/5 rounded-r">
                                {inc.reason}
                            </div>
                        )}
                    </div>
                );
            })}
        </div>
    );
}

function ResiliencePanel({ resilience }) {
    if (!resilience) return <div className="text-textMuted text-sm font-mono p-4">Loading metrics...</div>;
    if (resilience.total_tests === 0) {
        return (
            <div className="p-8 text-center border-2 border-dashed border-borderHairline rounded-xl flex flex-col items-center">
                <Target className="w-10 h-10 mb-3 opacity-20 text-textMuted" />
                <div className="text-textMuted text-sm font-mono">{resilience.message || "No resilience data available."}</div>
            </div>
        );
    }
    return (
        <div>
            <div className="flex flex-wrap gap-4 mb-6">
                <div className="flex-1 min-w-[100px] bg-statusSafe/10 border border-statusSafe/20 rounded-xl p-4 text-center">
                    <div className="text-4xl font-black font-mono text-statusSafe">{resilience.overall_resilience_pct}%</div>
                    <div className="text-[10px] text-textSecondary uppercase tracking-widest font-bold mt-2">Overall Catch Rate</div>
                </div>
                <div className="flex-1 min-w-[100px] bg-redPrimary/10 border border-redPrimary/20 rounded-xl p-4 text-center">
                    <div className="text-4xl font-black font-mono text-redPrimary">{resilience.avg_detection_ms}ms</div>
                    <div className="text-[10px] text-textSecondary uppercase tracking-widest font-bold mt-2">Avg Detection Time</div>
                </div>
                <div className="flex-1 min-w-[100px] bg-statusCritical/10 border border-statusCritical/20 rounded-xl p-4 text-center">
                    <div className="text-4xl font-black font-mono text-statusCritical">{resilience.total_missed}</div>
                    <div className="text-[10px] text-textSecondary uppercase tracking-widest font-bold mt-2">Attacks Missed</div>
                </div>
            </div>
            <div className="space-y-4">
                {Object.entries(resilience.by_attack_type || {}).map(([type, stats]) => {
                    const color = ATTACK_COLORS[type] || 'var(--red-primary)';
                    const pct = stats.catch_rate_pct || 0;
                    return (
                        <div key={type}>
                            <div className="flex justify-between items-center text-xs font-bold uppercase tracking-wider mb-2">
                                <span className="text-textSecondary">{type.replace(/_/g, ' ')}</span>
                                <span className="font-mono text-[11px]" style={{ color }}>
                                    {stats.caught}/{stats.total_tests} · {pct}% · {stats.avg_detection_ms}ms
                                </span>
                            </div>
                            <div className="h-1.5 bg-bgVoid rounded-full overflow-hidden">
                                <div className="h-full rounded-full transition-all duration-1000 ease-out" style={{ width: `${pct}%`, background: color, boxShadow: `0 0 10px ${color}80` }} />
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}

function PlaybookPanel() {
    const [playbooks, setPlaybooks] = useState([]);
    const [selected, setSelected] = useState(null);
    const [detail, setDetail] = useState(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        api.listPlaybooks().then(d => setPlaybooks(d.playbooks || [])).catch(() => { });
    }, []);

    const loadPlaybook = async (id) => {
        if (selected === id) { setSelected(null); setDetail(null); return; }
        setSelected(id); setLoading(true);
        try {
            const d = await api.getPlaybook(id);
            setDetail(d);
        } catch (e) { console.error(e); }
        finally { setLoading(false); }
    };

    return (
        <div>
            <div className="flex gap-2 flex-wrap mb-6">
                {playbooks.map(p => {
                    const isSelected = selected === p.id;
                    return (
                        <button key={p.id} onClick={() => loadPlaybook(p.id)} 
                            className={`px-4 py-2 rounded-full text-[11px] uppercase tracking-widest font-bold border transition-all ${isSelected ? 'bg-redPrimary/20 text-redPrimary border-redPrimary' : 'bg-transparent text-textMuted border-borderHairline hover:border-textMuted hover:text-textSecondary'}`}>
                            {p.attack}
                        </button>
                    );
                })}
            </div>

            {loading && <div className="text-textMuted text-sm font-mono animate-pulse">Loading playbook...</div>}

            {detail && !loading && (
                <div className="bg-bgVoid border border-redPrimary/30 rounded-xl p-6 animate-fadeInUp">
                    <div className="flex items-center gap-3 mb-5">
                        <div className="text-lg font-black tracking-wider text-redPrimary uppercase">{detail.attack}</div>
                        <span className="text-[9px] px-2 py-0.5 rounded border border-redPrimary/40 bg-redPrimary/10 text-redPrimary uppercase tracking-widest font-bold">
                            {detail.severity}
                        </span>
                    </div>
                    <div className="text-sm font-mono text-textSecondary mb-6 leading-relaxed border-l-2 border-borderHairline pl-4 italic">
                        "{detail.description}"
                    </div>

                    <div className="space-y-6">
                        {[
                            { title: 'Immediate Response', key: 'immediate_steps', icon: AlertTriangle },
                            { title: 'Investigation', key: 'investigation_steps', icon: Activity },
                            { title: 'Remediation', key: 'remediation', icon: ShieldCheck },
                        ].map(section => (
                            <div key={section.key}>
                                <div className="text-[11px] text-redPrimary font-bold mb-3 uppercase tracking-widest flex items-center gap-2">
                                    <section.icon className="w-3.5 h-3.5" /> {section.title}
                                </div>
                                <div className="space-y-2">
                                    {(detail[section.key] || []).map((step, i) => (
                                        <div key={i} className="text-xs font-mono text-textPrimary py-1.5 px-3 border-l-2 border-redPrimary/50 bg-bgPanel rounded-r">
                                            {step}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>

                    {detail.regulatory && (
                        <div className="mt-6 text-[11px] font-mono text-textSecondary bg-bgPanel rounded-lg p-3 border border-borderHairline flex items-center gap-3">
                            <BookOpen className="w-4 h-4 text-textMuted" />
                            {detail.regulatory}
                        </div>
                    )}
                </div>
            )}

            {!detail && !loading && playbooks.length > 0 && (
                <div className="text-textMuted text-sm font-mono p-4 border-2 border-dashed border-borderHairline rounded-xl text-center">
                    Select an attack type above to view the step-by-step response playbook.
                </div>
            )}
        </div>
    );
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function BlueTeamPage() {
    const [status, setStatus] = useState(null);
    const [incidents, setIncidents] = useState([]);
    const [resilience, setResilience] = useState(null);
    const [loading, setLoading] = useState(true);
    const [decisionMsg, setDecisionMsg] = useState(null);

    const loadAll = useCallback(async () => {
        try {
            const [s, inc, res] = await Promise.all([
                api.getBlueTeamStatus(),
                api.getBlueTeamIncidents(),
                api.getBlueTeamResilience(),
            ]);
            setStatus(s);
            setIncidents(inc.incidents || []);
            setResilience(res);
        } catch (e) { console.error(e); }
        finally { setLoading(false); }
    }, []);

    useEffect(() => { loadAll(); const t = setInterval(loadAll, 15000); return () => clearInterval(t); }, [loadAll]);

    const handleDecide = async (caseId, decision) => {
        try {
            await api.submitReviewDecision(caseId, decision, 'analyst');
            setDecisionMsg(`Case ${caseId.slice(0, 8)}… → ${decision.replace(/_/g, ' ')}`);
            setTimeout(() => setDecisionMsg(null), 3000);
            await loadAll();
        } catch (e) { console.error(e); }
    };

    if (loading) {
        return (
            <div className="flex flex-col items-center justify-center min-h-[calc(100vh-80px)]">
                <Shield className="w-16 h-16 text-redPrimary animate-pulse mb-6" />
                <div className="text-textMuted font-mono text-sm tracking-widest uppercase">Initialising Blue Team SOC...</div>
            </div>
        );
    }

    return (
        <div className="relative z-10 px-6 md:px-12 py-12 max-w-7xl mx-auto flex flex-col gap-8 min-h-[calc(100vh-80px)] animate-fadeInUp">
            {/* Header */}
            <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
                <div>
                    <div className="font-mono text-xs text-redPrimary tracking-widest uppercase mb-3 flex items-center gap-2 font-bold">
                        <Shield className="w-4 h-4" /> Blue Team
                    </div>
                    <h1 className="text-[48px] font-display font-bold text-textPrimary m-0 tracking-tight leading-none">
                        Security Operations <span className="text-redPrimary">Centre</span>
                    </h1>
                    <p className="font-mono text-[13px] text-textMuted mt-4 uppercase tracking-widest">
                        Real-time defense status // HITL review queue // Resilience metrics
                    </p>
                </div>
                <button onClick={loadAll} className="flex items-center gap-2 px-6 py-2.5 border border-redPrimary/30 bg-redPrimary/10 text-redPrimary font-mono text-sm font-bold rounded-xl hover:bg-redPrimary/20 transition-all uppercase tracking-widest">
                    <RefreshCw className="w-4 h-4" /> Refresh
                </button>
            </div>

            {decisionMsg && (
                <div className="bg-statusSafe/10 border border-statusSafe/30 rounded-xl px-5 py-4 text-statusSafe text-sm font-bold font-mono flex items-center gap-3">
                    <CheckCircle className="w-5 h-5" /> Decision recorded: {decisionMsg}
                </div>
            )}

            {/* Threat Banner */}
            <ThreatBanner status={status} />

            {/* Stats Row */}
            <DefenseStats status={status} />

            {/* Main grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-4">
                {/* HITL Queue */}
                <div className="bg-bgPanel border border-borderHairline rounded-[24px] p-6 lg:p-8">
                    <SectionHeader title={`Human Review Queue (${(status?.pending_cases || []).length} pending)`} icon={ClipboardList} />
                    <div className="max-h-[400px] overflow-y-auto pr-2 custom-scrollbar">
                        <HITLQueue cases={status?.pending_cases || []} onDecide={handleDecide} />
                    </div>
                </div>

                {/* Resilience */}
                <div className="bg-bgPanel border border-borderHairline rounded-[24px] p-6 lg:p-8">
                    <SectionHeader title="Red Team Resilience Metrics" icon={Target} />
                    <ResiliencePanel resilience={resilience} />
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Incident Log */}
                <div className="bg-bgPanel border border-borderHairline rounded-[24px] p-6 lg:p-8">
                    <SectionHeader title={`Defense Incident Log (${incidents.length} events)`} icon={Radio} />
                    <div className="max-h-[400px] overflow-y-auto pr-2 custom-scrollbar">
                        <IncidentLog incidents={incidents} />
                    </div>
                </div>

                {/* Playbooks */}
                <div className="bg-bgPanel border border-borderHairline rounded-[24px] p-6 lg:p-8">
                    <SectionHeader title="Incident Response Playbooks" icon={BookOpen} />
                    <PlaybookPanel />
                </div>
            </div>

            <style>{`
                @keyframes fadeInUp { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:translateY(0); } }
                @keyframes flipIn {
                    0% { transform: perspective(400px) rotateX(90deg); opacity: 0; }
                    100% { transform: perspective(400px) rotateX(0deg); opacity: 1; }
                }
                .animate-flipIn {
                    animation: flipIn 0.6s cubic-bezier(0.4, 0, 0.2, 1) forwards;
                    opacity: 0;
                }
                .custom-scrollbar::-webkit-scrollbar { width: 6px; }
                .custom-scrollbar::-webkit-scrollbar-track { background: var(--bg-void); border-radius: 4px; }
                .custom-scrollbar::-webkit-scrollbar-thumb { background: var(--border-hairline); border-radius: 4px; }
                .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: var(--red-primary); }
            `}</style>
        </div>
    );
}
