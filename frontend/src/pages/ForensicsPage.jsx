import { useState, useEffect } from 'react';
import { api } from '../services/api';
import { Search, Upload, Target } from 'lucide-react';

function NarrativeBox({ narrative }) {
    if (!narrative) return null;
    return (
        <div className="bg-bg2 border border-border rounded-lg p-6 font-mono text-xs">
            <div className="text-accent uppercase tracking-widest mb-3 text-xs">Attack Reconstruction Report</div>
            <pre className="text-text2 whitespace-pre-wrap leading-relaxed">{narrative}</pre>
        </div>
    );
}

function AttackClassCard({ classification }) {
    if (!classification) return null;
    const severityColor = {
        critical: 'danger', high: 'orange', medium: 'yellow', low: 'accent3'
    }[classification.severity] || 'text2';

    return (
        <div className="bg-surface border border-border rounded-lg p-6">
            <div className="font-mono text-xs text-accent uppercase tracking-widest mb-4">Attack Classification</div>
            <div className="space-y-3">
                <div className="flex justify-between">
                    <span className="font-mono text-xs text-text3">Attack Type</span>
                    <span className="font-mono text-sm font-bold text-danger capitalize">
                        {classification.attack_type?.replace(/_/g, ' ')}
                    </span>
                </div>
                <div className="flex justify-between">
                    <span className="font-mono text-xs text-text3">Subtype</span>
                    <span className="font-mono text-xs text-text2 capitalize">
                        {classification.attack_subtype?.replace(/_/g, ' ')}
                    </span>
                </div>
                <div className="flex justify-between">
                    <span className="font-mono text-xs text-text3">Confidence</span>
                    <span className="font-mono text-xs font-bold text-accent">
                        {(classification.confidence * 100).toFixed(1)}%
                    </span>
                </div>
                <div className="flex justify-between">
                    <span className="font-mono text-xs text-text3">Severity</span>
                    <span className={`font-mono text-xs font-bold text-${severityColor} uppercase`}>
                        {classification.severity}
                    </span>
                </div>
                <div className="pt-2 border-t border-border">
                    <div className="font-mono text-xs text-text3 mb-2">Probability Distribution</div>
                    {Object.entries(classification.probabilities || {}).map(([type, prob]) => (
                        <div key={type} className="mb-1.5">
                            <div className="flex justify-between font-mono text-xs mb-0.5">
                                <span className="text-text3 capitalize">{type.replace(/_/g, ' ')}</span>
                                <span className="text-text2">{(prob * 100).toFixed(1)}%</span>
                            </div>
                            <div className="h-1 bg-border rounded-full overflow-hidden">
                                <div className="h-full bg-accent/60 rounded-full"
                                    style={{ width: `${prob * 100}%` }} />
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

function SophisticationCard({ sophistication }) {
    if (!sophistication) return null;
    const score = sophistication.sophistication_score || 0;
    const color = score >= 8 ? 'danger' : score >= 4 ? 'orange' : 'yellow';

    return (
        <div className="bg-surface border border-border rounded-lg p-6">
            <div className="font-mono text-xs text-accent uppercase tracking-widest mb-4">
                Attacker Sophistication
            </div>
            <div className="flex items-center gap-4 mb-4">
                <div className={`text-5xl font-bold text-${color}`}>{score}</div>
                <div>
                    <div className="font-mono text-xs text-text3">/ 10</div>
                    <div className={`font-mono text-xs text-${color} mt-1`}>{sophistication.level}</div>
                </div>
            </div>
            <div className="space-y-2">
                {Object.entries(sophistication.factors || {}).map(([key, val]) => (
                    <div key={key} className="flex justify-between font-mono text-xs">
                        <span className="text-text3 capitalize">{key.replace(/_/g, ' ')}</span>
                        <span className="text-text2">{(val * 100).toFixed(0)}%</span>
                    </div>
                ))}
            </div>
        </div>
    );
}

function PatternCard({ pattern }) {
    if (!pattern) return null;
    return (
        <div className="bg-surface border border-border rounded-lg p-6">
            <div className="font-mono text-xs text-accent uppercase tracking-widest mb-4">
                Injection Pattern
            </div>
            <div className="space-y-2 font-mono text-xs">
                {[
                    { label: 'Poisoned Samples', value: pattern.n_poisoned_samples },
                    { label: 'Affected Batches', value: pattern.n_batches },
                    { label: 'Injection Schedule', value: pattern.injection_schedule?.replace(/_/g, ' ') },
                    { label: 'Sigma Shift', value: pattern.sigma_shift != null ? `${pattern.sigma_shift}σ` : null },
                    { label: 'Primary Client', value: pattern.primary_client },
                    { label: 'First Injection', value: pattern.first_injection?.slice(0, 19) },
                    { label: 'Last Injection', value: pattern.last_injection?.slice(0, 19) },
                ].map(({ label, value }) => value != null && (
                    <div key={label} className="flex justify-between py-1.5 border-b border-border/40">
                        <span className="text-text3">{label}</span>
                        <span className="text-text2 font-medium">{value}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}

function VerdictBadge({ verdict, score }) {
    const cfg = {
        CONFIRMED_POISONED: { color: '#ef4444', bg: 'rgba(239,68,68,0.1)', icon: '☠️' },
        SUSPICIOUS: { color: '#f59e0b', bg: 'rgba(245,158,11,0.1)', icon: '⚠️' },
        LOW_RISK: { color: '#3b82f6', bg: 'rgba(59,130,246,0.1)', icon: '🔵' },
        CLEAN: { color: '#22c55e', bg: 'rgba(34,197,94,0.1)', icon: '✅' },
    }[verdict] || { color: '#64748b', bg: 'rgba(100,116,139,0.1)', icon: '❓' };

    return (
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '8px 16px', borderRadius: 10, background: cfg.bg, border: `1px solid ${cfg.color}44` }}>
            <span style={{ fontSize: 20 }}>{cfg.icon}</span>
            <div>
                <div style={{ fontSize: 13, fontWeight: 800, color: cfg.color, fontFamily: 'monospace' }}>{verdict}</div>
                {score != null && <div style={{ fontSize: 11, color: '#64748b' }}>Suspicion: {(score * 100).toFixed(1)}%</div>}
            </div>
        </div>
    );
}

function DatasetInfoBanner({ info, dataSource }) {
    if (!info) return null;
    return (
        <div style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 10, padding: '10px 16px', fontSize: 12, color: '#94a3b8', display: 'flex', gap: 16, flexWrap: 'wrap' }}>
            <span>📂 <strong style={{ color: '#f1f5f9' }}>{info.filename}</strong></span>
            <span>Rows: <strong style={{ color: '#f1f5f9' }}>{info.n_rows?.toLocaleString()}</strong></span>
            <span>Features: <strong style={{ color: '#f1f5f9' }}>{info.n_features}</strong></span>
            <span>Mode: <strong style={{ color: '#6366f1' }}>{info.detection_mode}</strong></span>
            {dataSource && <span>Source: <strong style={{ color: '#06b6d4' }}>{dataSource}</strong></span>}
        </div>
    );
}

export default function ForensicsPage() {
    const [activeTab, setActiveTab] = useState('auto');   // 'auto' | 'demo' | 'upload'
    const [forensics, setForensics] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const load = async (src = activeTab) => {
        setLoading(true);
        setError(null);
        try {
            const data = await api.getLatestForensics(src);
            setForensics(data);
        } catch (e) {
            setError(
                src === 'upload'
                    ? 'No uploaded dataset analysis found. Upload a CSV first from the Upload Dataset page.'
                    : 'No forensic data yet. Run the demo from the Trust Dashboard first.'
            );
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => { load(activeTab); }, [activeTab]);

    const TABS = [
        { id: 'auto', label: 'Latest (Auto)', icon: Target },
        { id: 'demo', label: 'Demo Analysis', icon: Target },
        { id: 'upload', label: 'My Upload', icon: Upload },
    ];

    return (
        <div className="p-8 space-y-8 animate-in">
            <div className="flex items-start justify-between">
                <div>
                    <div className="font-mono text-xs text-accent3 tracking-widest uppercase mb-2 flex items-center gap-2">
                        <Search className="w-3 h-3" /> Forensic Analysis
                    </div>
                    <h1 className="text-4xl font-bold tracking-tight">
                        Poison <span className="text-accent3">Forensics</span>
                    </h1>
                    <p className="font-mono text-sm text-text2 mt-2">
                        WHY was it flagged · HOW was it injected · WHAT damage did it cause
                    </p>
                </div>
                <button onClick={() => load(activeTab)} disabled={loading}
                    className="px-4 py-2 border border-accent3/30 bg-accent3/10 text-accent3 font-mono text-xs rounded-lg hover:bg-accent3/20 transition-all">
                    {loading ? 'Loading...' : 'Refresh'}
                </button>
            </div>

            {/* Source Tabs */}
            <div style={{ display: 'flex', gap: 8 }}>
                {TABS.map(({ id, label, icon: Icon }) => (
                    <button key={id} onClick={() => setActiveTab(id)}
                        style={{
                            display: 'flex', alignItems: 'center', gap: 6,
                            padding: '7px 14px', borderRadius: 20, fontSize: 12, fontFamily: 'monospace',
                            cursor: 'pointer', transition: 'all 0.2s',
                            background: activeTab === id ? 'rgba(0,229,255,0.1)' : 'rgba(255,255,255,0.03)',
                            border: activeTab === id ? '1px solid rgba(0,229,255,0.4)' : '1px solid rgba(255,255,255,0.08)',
                            color: activeTab === id ? '#00e5ff' : '#64748b',
                            fontWeight: activeTab === id ? 700 : 400,
                        }}>
                        <Icon style={{ width: 12, height: 12 }} /> {label}
                    </button>
                ))}
            </div>

            {error && (
                <div className="bg-yellow/10 border border-yellow/30 rounded-lg p-4 font-mono text-sm text-yellow">
                    ⚠ {error}
                </div>
            )}

            {!forensics && !error && !loading && (
                <div className="text-center py-20 text-text3 font-mono text-sm">
                    No forensic data yet. Run the demo or upload a CSV first.
                </div>
            )}

            {forensics && (
                <>
                    {/* Verdict + dataset info */}
                    <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
                        <VerdictBadge verdict={forensics.verdict} score={forensics.overall_suspicion_score} />
                        <DatasetInfoBanner info={forensics.dataset_info} dataSource={forensics.source} />
                    </div>

                    {/* Attack Narrative */}
                    <NarrativeBox narrative={forensics.injection_pattern?.narrative} />

                    {/* Cards Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <AttackClassCard classification={forensics.attack_classification} />
                        <SophisticationCard sophistication={forensics.sophistication} />
                        <PatternCard pattern={forensics.injection_pattern} />
                    </div>

                    {/* Counterfactual */}
                    {forensics.counterfactual && (
                        <div className="bg-surface border border-border rounded-lg p-6">
                            <div className="font-mono text-xs text-purple uppercase tracking-widest mb-4">
                                Counterfactual Impact Simulator — "What if we hadn't detected this?"
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                                {forensics.counterfactual.counterfactual_projections?.map(p => (
                                    <div key={p.days} className="bg-bg3 border border-purple/20 rounded-lg p-4">
                                        <div className="font-mono text-xs text-text3 mb-2">{p.days}-Day Projection</div>
                                        <div className="text-2xl font-bold text-purple">{(p.projected_accuracy * 100).toFixed(1)}%</div>
                                        <div className="font-mono text-xs text-danger mt-1">
                                            -{(p.accuracy_loss * 100).toFixed(1)}% accuracy
                                        </div>
                                        <div className="font-mono text-xs text-text3 mt-1">
                                            ~{p.estimated_harm.toLocaleString()} predictions affected
                                        </div>
                                    </div>
                                ))}
                            </div>
                            <div className="bg-accent3/5 border border-accent3/20 rounded-lg p-4 font-mono text-sm text-accent3">
                                ✓ {forensics.counterfactual.detection_value}
                            </div>
                        </div>
                    )}
                </>
            )}
        </div>
    );
}
