import { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';

// ─── Constants ────────────────────────────────────────────────────────────────

const THREAT_CONFIG = {
    CRITICAL: { color: '#ef4444', bg: 'rgba(239,68,68,0.12)', border: 'rgba(239,68,68,0.4)', label: 'CRITICAL', icon: '🚨', pulse: true },
    ELEVATED: { color: '#f59e0b', bg: 'rgba(245,158,11,0.1)', border: 'rgba(245,158,11,0.35)', label: 'ELEVATED', icon: '⚠️', pulse: true },
    GUARDED: { color: '#3b82f6', bg: 'rgba(59,130,246,0.1)', border: 'rgba(59,130,246,0.3)', label: 'GUARDED', icon: '🔵', pulse: false },
    NOMINAL: { color: '#22c55e', bg: 'rgba(34,197,94,0.1)', border: 'rgba(34,197,94,0.25)', label: 'NOMINAL', icon: '🟢', pulse: false },
};

const ATTACK_COLORS = {
    label_flip: '#f59e0b', backdoor: '#ef4444',
    clean_label: '#a855f7', gradient_poisoning: '#06b6d4', boiling_frog: '#22c55e',
};

const SEVERITY_CONFIG = {
    critical: { color: '#ef4444', bg: 'rgba(239,68,68,0.1)', label: '🚨 CRITICAL' },
    high: { color: '#f59e0b', bg: 'rgba(245,158,11,0.1)', label: '⚠️ HIGH' },
    medium: { color: '#3b82f6', bg: 'rgba(59,130,246,0.1)', label: '🔵 MEDIUM' },
    info: { color: '#64748b', bg: 'rgba(100,116,139,0.1)', label: 'ℹ️ INFO' },
};

// ─── Sub-components ───────────────────────────────────────────────────────────

function StatCard({ icon, label, value, sub, color = '#6366f1' }) {
    return (
        <div style={{ background: 'rgba(255,255,255,0.03)', border: `1px solid ${color}22`, borderRadius: 12, padding: '16px 18px', flex: 1, minWidth: 120 }}>
            <div style={{ fontSize: 22, marginBottom: 4 }}>{icon}</div>
            <div style={{ fontSize: 26, fontWeight: 800, color, fontFamily: 'monospace', lineHeight: 1 }}>{value}</div>
            <div style={{ fontSize: 11, color: '#475569', marginTop: 4 }}>{label}</div>
            {sub && <div style={{ fontSize: 10, color: '#334155', marginTop: 2 }}>{sub}</div>}
        </div>
    );
}

function SectionHeader({ title }) {
    return (
        <div style={{ fontSize: 11, color: '#475569', textTransform: 'uppercase', letterSpacing: 2, marginBottom: 12, fontWeight: 700, paddingBottom: 6, borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
            {title}
        </div>
    );
}

function ThreatBanner({ status }) {
    if (!status) return null;
    const cfg = THREAT_CONFIG[status.threat_level] || THREAT_CONFIG.NOMINAL;
    return (
        <div style={{ background: cfg.bg, border: `1px solid ${cfg.border}`, borderRadius: 16, padding: '20px 28px', marginBottom: 24, display: 'flex', alignItems: 'center', gap: 20, position: 'relative', overflow: 'hidden' }}>
            {cfg.pulse && (
                <div style={{ position: 'absolute', top: 0, right: 0, width: '100%', height: '100%', background: `radial-gradient(ellipse at right, ${cfg.color}08 0%, transparent 70%)`, pointerEvents: 'none' }} />
            )}
            <div style={{ fontSize: 52, filter: cfg.pulse ? 'drop-shadow(0 0 12px currentColor)' : 'none' }}>{cfg.icon}</div>
            <div style={{ flex: 1 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 4 }}>
                    <span style={{ fontSize: 22, fontWeight: 900, color: cfg.color, letterSpacing: 3, fontFamily: 'monospace' }}>
                        THREAT LEVEL: {cfg.label}
                    </span>
                    {cfg.pulse && (
                        <span style={{ fontSize: 10, padding: '2px 8px', borderRadius: 20, background: `${cfg.color}20`, color: cfg.color, border: `1px solid ${cfg.color}44`, animation: 'blink 1.5s infinite' }}>
                            LIVE
                        </span>
                    )}
                </div>
                <div style={{ color: '#94a3b8', fontSize: 13 }}>
                    Current Verdict: <strong style={{ color: cfg.color }}>{status.current_verdict}</strong>
                    &nbsp;·&nbsp; Suspicion: <strong style={{ color: cfg.color }}>{Math.round((status.suspicion_score || 0) * 100)}%</strong>
                    &nbsp;·&nbsp; Mode: <strong style={{ color: '#f1f5f9', textTransform: 'capitalize' }}>{status.defense_mode}</strong>
                </div>
            </div>
            <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: 11, color: '#475569' }}>Updated</div>
                <div style={{ fontSize: 12, color: '#64748b', fontFamily: 'monospace' }}>
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
        <div style={{ display: 'flex', gap: 12, marginBottom: 24, flexWrap: 'wrap' }}>
            <StatCard icon="🔒" label="Total Quarantined" value={status.total_quarantined || 0} color="#ef4444" />
            <StatCard icon="⚡" label="Defense Actions" value={status.n_defense_actions || 0} color="#f59e0b" />
            <StatCard icon="📋" label="HITL Queue" value={status.hitl_queue_depth || 0} sub="pending analyst review" color="#a855f7" />
            <StatCard icon="🎯" label="Red Team Tests" value={rt.total_simulations || 0} sub={`${rt.attacks_caught || 0} caught`} color="#06b6d4" />
            <StatCard icon="🛡️" label="Resilience" value={`${rt.resilience_pct ?? 100}%`} sub="attack catch rate" color="#22c55e" />
        </div>
    );
}

function HITLQueue({ cases, onDecide }) {
    if (!cases || cases.length === 0) {
        return (
            <div style={{ padding: '20px', textAlign: 'center', color: '#475569', fontSize: 13 }}>
                ✅ No pending cases — queue is clear
            </div>
        );
    }
    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {cases.map(c => (
                <div key={c.case_id} style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(245,158,11,0.2)', borderRadius: 10, padding: '14px 16px' }}>
                    <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
                        <div style={{ flex: 1 }}>
                            <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 4 }}>
                                <span style={{ fontSize: 11, fontFamily: 'monospace', color: '#64748b' }}>{c.case_id?.slice(0, 8)}…</span>
                                <span style={{ fontSize: 10, padding: '1px 6px', borderRadius: 10, background: 'rgba(245,158,11,0.15)', color: '#f59e0b', border: '1px solid rgba(245,158,11,0.3)' }}>PENDING</span>
                            </div>
                            <div style={{ fontSize: 12, color: '#cbd5e1', marginBottom: 6 }}>
                                Suspicion: <strong style={{ color: '#f59e0b' }}>{Math.round((c.suspicion_score || 0) * 100)}%</strong>
                                &nbsp;·&nbsp; Samples: <strong>{c.n_samples}</strong>
                                &nbsp;·&nbsp; Attack: <strong style={{ color: ATTACK_COLORS[c.evidence_summary?.attack_type] || '#94a3b8', textTransform: 'capitalize' }}>
                                    {(c.evidence_summary?.attack_type || 'unknown').replace(/_/g, ' ')}
                                </strong>
                            </div>
                            <div style={{ fontSize: 11, color: '#475569' }}>
                                KL Divergence: {c.evidence_summary?.kl_divergence?.toFixed(3) || '—'} &nbsp;·&nbsp;
                                Causal Effect: {c.evidence_summary?.causal_effect?.toFixed(3) || '—'}
                            </div>
                        </div>
                        <div style={{ display: 'flex', gap: 6, flexShrink: 0 }}>
                            <button onClick={() => onDecide(c.case_id, 'approve_quarantine')}
                                style={{ padding: '6px 12px', borderRadius: 8, border: '1px solid rgba(239,68,68,0.3)', background: 'rgba(239,68,68,0.1)', color: '#fca5a5', fontSize: 12, cursor: 'pointer', fontWeight: 600 }}>
                                🔒 Quarantine
                            </button>
                            <button onClick={() => onDecide(c.case_id, 'mark_safe')}
                                style={{ padding: '6px 12px', borderRadius: 8, border: '1px solid rgba(34,197,94,0.3)', background: 'rgba(34,197,94,0.1)', color: '#86efac', fontSize: 12, cursor: 'pointer', fontWeight: 600 }}>
                                ✅ Safe
                            </button>
                        </div>
                    </div>
                </div>
            ))}
        </div>
    );
}

function IncidentLog({ incidents }) {
    const [expanded, setExpanded] = useState(null);
    if (!incidents || incidents.length === 0) {
        return <div style={{ padding: 20, textAlign: 'center', color: '#475569', fontSize: 13 }}>No incidents logged yet. Run a demo or trigger a defense action.</div>;
    }
    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            {incidents.map((inc, i) => {
                const sev = SEVERITY_CONFIG[inc.severity] || SEVERITY_CONFIG.info;
                const isOpen = expanded === i;
                return (
                    <div key={i} onClick={() => setExpanded(isOpen ? null : i)} style={{ padding: '10px 14px', borderRadius: 8, cursor: 'pointer', background: isOpen ? 'rgba(255,255,255,0.04)' : 'rgba(255,255,255,0.01)', border: `1px solid ${isOpen ? 'rgba(255,255,255,0.1)' : 'rgba(255,255,255,0.04)'}`, transition: 'all 0.2s' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                            <span style={{ fontSize: 10, padding: '2px 7px', borderRadius: 8, background: sev.bg, color: sev.color, border: `1px solid ${sev.color}33`, whiteSpace: 'nowrap' }}>
                                {sev.label}
                            </span>
                            <span style={{ fontSize: 12, color: '#cbd5e1', flex: 1, textTransform: 'capitalize' }}>
                                {inc.type === 'human_decision' ? '👤' : '🤖'}
                                &nbsp;{(inc.action || '').replace(/_/g, ' ')}
                                {inc.samples_affected ? ` — ${inc.samples_affected} samples` : ''}
                                {inc.reviewer ? ` by ${inc.reviewer}` : ''}
                            </span>
                            <span style={{ fontSize: 10, color: '#475569', fontFamily: 'monospace' }}>
                                {inc.timestamp ? new Date(inc.timestamp).toLocaleTimeString() : '—'}
                            </span>
                        </div>
                        {isOpen && inc.reason && (
                            <div style={{ marginTop: 8, fontSize: 11, color: '#64748b', paddingLeft: 4, borderLeft: '2px solid rgba(255,255,255,0.08)' }}>
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
    if (!resilience) return <div style={{ color: '#475569', fontSize: 13, padding: 16 }}>Loading...</div>;
    if (resilience.total_tests === 0) {
        return (
            <div style={{ padding: 20, textAlign: 'center' }}>
                <div style={{ fontSize: 32, marginBottom: 8 }}>🎯</div>
                <div style={{ color: '#64748b', fontSize: 13 }}>{resilience.message}</div>
            </div>
        );
    }
    return (
        <div>
            <div style={{ display: 'flex', gap: 10, marginBottom: 16, flexWrap: 'wrap' }}>
                <div style={{ flex: 1, minWidth: 100, background: 'rgba(34,197,94,0.08)', border: '1px solid rgba(34,197,94,0.2)', borderRadius: 10, padding: '12px 14px', textAlign: 'center' }}>
                    <div style={{ fontSize: 28, fontWeight: 800, color: '#22c55e', fontFamily: 'monospace' }}>{resilience.overall_resilience_pct}%</div>
                    <div style={{ fontSize: 10, color: '#475569', marginTop: 2 }}>Overall Catch Rate</div>
                </div>
                <div style={{ flex: 1, minWidth: 100, background: 'rgba(99,102,241,0.08)', border: '1px solid rgba(99,102,241,0.2)', borderRadius: 10, padding: '12px 14px', textAlign: 'center' }}>
                    <div style={{ fontSize: 28, fontWeight: 800, color: '#818cf8', fontFamily: 'monospace' }}>{resilience.avg_detection_ms}ms</div>
                    <div style={{ fontSize: 10, color: '#475569', marginTop: 2 }}>Avg Detection Time</div>
                </div>
                <div style={{ flex: 1, minWidth: 100, background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: 10, padding: '12px 14px', textAlign: 'center' }}>
                    <div style={{ fontSize: 28, fontWeight: 800, color: '#ef4444', fontFamily: 'monospace' }}>{resilience.total_missed}</div>
                    <div style={{ fontSize: 10, color: '#475569', marginTop: 2 }}>Attacks Missed</div>
                </div>
            </div>
            {Object.entries(resilience.by_attack_type || {}).map(([type, stats]) => {
                const color = ATTACK_COLORS[type] || '#6366f1';
                const pct = stats.catch_rate_pct || 0;
                return (
                    <div key={type} style={{ marginBottom: 10 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                            <span style={{ color: '#94a3b8', textTransform: 'capitalize' }}>{type.replace(/_/g, ' ')}</span>
                            <span style={{ fontFamily: 'monospace', color }}>
                                {stats.caught}/{stats.total_tests} · {pct}% · {stats.avg_detection_ms}ms
                            </span>
                        </div>
                        <div style={{ height: 6, background: 'rgba(255,255,255,0.05)', borderRadius: 3, overflow: 'hidden' }}>
                            <div style={{ height: '100%', width: `${pct}%`, background: `linear-gradient(90deg, ${color}88, ${color})`, borderRadius: 3, transition: 'width 1s ease' }} />
                        </div>
                    </div>
                );
            })}
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
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 16 }}>
                {playbooks.map(p => (
                    <button key={p.id} onClick={() => loadPlaybook(p.id)} style={{ padding: '7px 14px', borderRadius: 20, border: `1px solid ${selected === p.id ? p.color : 'rgba(255,255,255,0.1)'}`, background: selected === p.id ? `${p.color}18` : 'rgba(255,255,255,0.03)', color: selected === p.id ? p.color : '#64748b', fontSize: 12, cursor: 'pointer', fontWeight: selected === p.id ? 700 : 400, transition: 'all 0.2s' }}>
                        {p.attack}
                    </button>
                ))}
            </div>

            {loading && <div style={{ color: '#64748b', fontSize: 13 }}>Loading playbook...</div>}

            {detail && !loading && (
                <div style={{ background: 'rgba(0,0,0,0.2)', border: `1px solid ${detail.color}33`, borderRadius: 12, padding: 20, animation: 'fadeIn 0.3s ease' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
                        <div style={{ fontSize: 16, fontWeight: 800, color: detail.color }}>{detail.attack}</div>
                        <span style={{ fontSize: 10, padding: '2px 8px', borderRadius: 10, background: `${detail.color}15`, color: detail.color, border: `1px solid ${detail.color}33`, textTransform: 'uppercase', letterSpacing: 1 }}>
                            {detail.severity}
                        </span>
                    </div>
                    <div style={{ fontSize: 12, color: '#64748b', marginBottom: 14, lineHeight: 1.6 }}>{detail.description}</div>

                    {[
                        { title: '🚨 Immediate Response', key: 'immediate_steps' },
                        { title: '🔍 Investigation', key: 'investigation_steps' },
                        { title: '🔧 Remediation', key: 'remediation' },
                    ].map(section => (
                        <div key={section.key} style={{ marginBottom: 14 }}>
                            <div style={{ fontSize: 11, color: detail.color, fontWeight: 700, marginBottom: 6, textTransform: 'uppercase', letterSpacing: 1 }}>{section.title}</div>
                            {(detail[section.key] || []).map((step, i) => (
                                <div key={i} style={{ fontSize: 12, color: '#94a3b8', padding: '5px 10px', borderLeft: `2px solid ${detail.color}44`, marginBottom: 4, lineHeight: 1.5, background: 'rgba(255,255,255,0.02)', borderRadius: '0 6px 6px 0' }}>
                                    {step}
                                </div>
                            ))}
                        </div>
                    ))}

                    {detail.regulatory && (
                        <div style={{ fontSize: 11, color: '#334155', background: 'rgba(255,255,255,0.03)', borderRadius: 8, padding: '8px 12px', borderLeft: `2px solid #475569` }}>
                            📜 {detail.regulatory}
                        </div>
                    )}
                </div>
            )}

            {!detail && !loading && playbooks.length > 0 && (
                <div style={{ color: '#475569', fontSize: 13, padding: 12 }}>
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
    const [deciding, setDeciding] = useState(null);
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
        setDeciding(caseId);
        try {
            await api.submitReviewDecision(caseId, decision, 'analyst');
            setDecisionMsg(`Case ${caseId.slice(0, 8)}… → ${decision.replace(/_/g, ' ')}`);
            setTimeout(() => setDecisionMsg(null), 3000);
            await loadAll();
        } catch (e) { console.error(e); }
        finally { setDeciding(null); }
    };

    if (loading) {
        return (
            <div style={{ padding: '60px 40px', textAlign: 'center' }}>
                <div style={{ fontSize: 48, marginBottom: 16 }}>🛡️</div>
                <div style={{ color: '#64748b', fontSize: 14 }}>Initialising Blue Team SOC...</div>
            </div>
        );
    }

    return (
        <div style={{ padding: '32px 40px', maxWidth: 1200, margin: '0 auto' }}>
            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 24 }}>
                <div>
                    <h1 style={{ fontSize: 26, fontWeight: 800, color: '#f1f5f9', margin: 0, display: 'flex', alignItems: 'center', gap: 10 }}>
                        🛡️ Blue Team — Security Operations Centre
                    </h1>
                    <p style={{ color: '#64748b', marginTop: 6, fontSize: 13 }}>
                        Real-time defense status · HITL review queue · Incident log · Resilience metrics · Incident playbooks
                    </p>
                </div>
                <button onClick={loadAll} style={{ padding: '8px 14px', borderRadius: 10, border: '1px solid rgba(255,255,255,0.1)', background: 'rgba(255,255,255,0.04)', color: '#64748b', cursor: 'pointer', fontSize: 13 }}>
                    🔄 Refresh
                </button>
            </div>

            {decisionMsg && (
                <div style={{ background: 'rgba(34,197,94,0.1)', border: '1px solid rgba(34,197,94,0.3)', borderRadius: 10, padding: '10px 16px', color: '#86efac', marginBottom: 16, fontSize: 13 }}>
                    ✅ Decision recorded: {decisionMsg}
                </div>
            )}

            {/* Threat Banner */}
            <ThreatBanner status={status} />

            {/* Stats Row */}
            <DefenseStats status={status} />

            {/* Main grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
                {/* HITL Queue */}
                <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 14, padding: 20 }}>
                    <SectionHeader title={`📋 Human Review Queue (${(status?.pending_cases || []).length} pending)`} />
                    <HITLQueue cases={status?.pending_cases || []} onDecide={handleDecide} />
                </div>

                {/* Resilience */}
                <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 14, padding: 20 }}>
                    <SectionHeader title="🎯 Red Team Resilience Metrics" />
                    <ResiliencePanel resilience={resilience} />
                </div>
            </div>

            {/* Incident Log */}
            <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 14, padding: 20, marginBottom: 16 }}>
                <SectionHeader title={`📡 Defense Incident Log (${incidents.length} events)`} />
                <div style={{ maxHeight: 280, overflowY: 'auto' }}>
                    <IncidentLog incidents={incidents} />
                </div>
            </div>

            {/* Playbooks */}
            <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 14, padding: 20 }}>
                <SectionHeader title="📖 Incident Response Playbooks — Step-by-Step Defense Procedures" />
                <PlaybookPanel />
            </div>

            <style>{`
                @keyframes fadeIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
                @keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0.4; } }
            `}</style>
        </div>
    );
}
