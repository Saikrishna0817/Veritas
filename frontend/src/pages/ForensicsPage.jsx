import { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import { Search, Upload, Target, ShieldAlert, Activity, Check, HelpCircle, Folder, AlertTriangle, Fingerprint, Bug, BrainCircuit } from 'lucide-react';

function NarrativeBox({ narrative }) {
    if (!narrative) return null;
    return (
        <div className="bg-bgPanel border border-borderHairline rounded-xl p-6 font-mono text-sm relative overflow-hidden group">
            <div className="absolute top-0 left-0 w-1 h-full bg-redPrimary shadow-red-glow"></div>
            <div className="text-redPrimary uppercase tracking-widest mb-4 text-xs font-bold flex items-center gap-2">
                <Search className="w-4 h-4" /> Heuristic Investigation Summary
            </div>
            <pre className="text-textSecondary whitespace-pre-wrap leading-relaxed bg-bgVoid p-4 rounded-lg border border-borderHairline">{narrative}</pre>
        </div>
    );
}

function AttackClassCard({ classification }) {
    if (!classification) return null;
    const severityColor = {
        critical: 'text-statusCritical', high: 'text-statusWarn', medium: 'text-yellow-500', low: 'text-statusSafe'
    }[classification.severity] || 'text-textMuted';

    return (
        <div className="bg-bgPanel border border-borderHairline rounded-xl p-6 hover:border-redPrimary/30 transition-colors">
            <div className="font-mono text-xs text-redPrimary uppercase tracking-widest mb-6 font-bold flex items-center gap-2">
                <Bug className="w-4 h-4" /> Classification
            </div>
            <div className="space-y-4">
                <div className="flex justify-between items-center pb-3 border-b border-borderHairline">
                    <span className="font-mono text-xs text-textMuted">Attack Type</span>
                    <span className="font-mono text-sm font-bold text-statusCritical capitalize">
                        {classification.attack_type?.replace(/_/g, ' ')}
                    </span>
                </div>
                <div className="flex justify-between items-center pb-3 border-b border-borderHairline">
                    <span className="font-mono text-xs text-textMuted">Subtype</span>
                    <span className="font-mono text-xs text-textSecondary capitalize">
                        {classification.attack_subtype?.replace(/_/g, ' ')}
                    </span>
                </div>
                <div className="flex justify-between items-center pb-3 border-b border-borderHairline">
                    <span className="font-mono text-xs text-textMuted">Confidence</span>
                    <span className="font-mono text-xs font-bold text-redPrimary">
                        {(classification.confidence * 100).toFixed(1)}%
                    </span>
                </div>
                <div className="flex justify-between items-center pb-3 border-b border-borderHairline">
                    <span className="font-mono text-xs text-textMuted">Severity</span>
                    <span className={`font-mono text-xs font-bold ${severityColor} uppercase`}>
                        {classification.severity}
                    </span>
                </div>
                <div className="pt-2">
                    <div className="font-mono text-xs text-textMuted mb-3">Probability Distribution</div>
                    {Object.entries(classification.probabilities || {}).map(([type, prob]) => (
                        <div key={type} className="mb-2.5">
                            <div className="flex justify-between font-mono text-[11px] mb-1">
                                <span className="text-textSecondary capitalize">{type.replace(/_/g, ' ')}</span>
                                <span className="text-textPrimary">{(prob * 100).toFixed(1)}%</span>
                            </div>
                            <div className="h-1.5 bg-bgVoid rounded-full overflow-hidden">
                                <div className="h-full bg-redPrimary shadow-red-glow rounded-full transition-all"
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
    const colorClass = score >= 8 ? 'text-statusCritical' : score >= 4 ? 'text-statusWarn' : 'text-statusSafe';

    return (
        <div className="bg-bgPanel border border-borderHairline rounded-xl p-6 hover:border-redPrimary/30 transition-colors">
            <div className="font-mono text-xs text-redPrimary uppercase tracking-widest mb-6 font-bold flex items-center gap-2">
                <BrainCircuit className="w-4 h-4" /> Sophistication
            </div>
            <div className="flex items-center gap-5 mb-6 bg-bgVoid p-4 rounded-lg border border-borderHairline">
                <div className={`text-6xl font-mono font-bold ${colorClass}`}>{score}</div>
                <div>
                    <div className="font-mono text-[10px] text-textMuted uppercase tracking-widest">/ 10 Score</div>
                    <div className={`font-mono text-sm font-bold ${colorClass} mt-1 uppercase`}>{sophistication.level}</div>
                </div>
            </div>
            <div className="space-y-3">
                {Object.entries(sophistication.factors || {}).map(([key, val]) => (
                    <div key={key} className="flex justify-between font-mono text-xs pb-2 border-b border-borderHairline last:border-0">
                        <span className="text-textMuted capitalize">{key.replace(/_/g, ' ')}</span>
                        <span className="text-textSecondary font-bold">{(val * 100).toFixed(0)}%</span>
                    </div>
                ))}
            </div>
        </div>
    );
}

function PatternCard({ pattern }) {
    if (!pattern) return null;
    return (
        <div className="bg-bgPanel border border-borderHairline rounded-xl p-6 hover:border-redPrimary/30 transition-colors">
            <div className="font-mono text-xs text-redPrimary uppercase tracking-widest mb-6 font-bold flex items-center gap-2">
                <Fingerprint className="w-4 h-4" /> Injection Pattern
            </div>
            <div className="space-y-0 font-mono text-xs">
                {[
                    { label: 'Poisoned Samples', value: pattern.n_poisoned_samples },
                    { label: 'Affected Batches', value: pattern.n_batches },
                    { label: 'Injection Schedule', value: pattern.injection_schedule?.replace(/_/g, ' ') },
                    { label: 'Sigma Shift', value: pattern.sigma_shift != null ? `${pattern.sigma_shift}σ` : null },
                    { label: 'Primary Client', value: pattern.primary_client },
                    { label: 'First Injection', value: pattern.first_injection?.slice(0, 19) },
                    { label: 'Last Injection', value: pattern.last_injection?.slice(0, 19) },
                ].map(({ label, value }) => value != null && (
                    <div key={label} className="flex justify-between py-3 border-b border-borderHairline last:border-0 items-center">
                        <span className="text-textMuted">{label}</span>
                        <span className="text-textPrimary font-bold bg-bgVoid px-2 py-1 rounded">{value}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}

function VerdictBadge({ verdict, score }) {
    const cfg = {
        CONFIRMED_POISONED: { color: 'var(--status-critical)', bg: 'rgba(228,36,43,0.1)', icon: <ShieldAlert className="w-6 h-6 text-[var(--status-critical)]" /> },
        SUSPICIOUS: { color: 'var(--status-warn)', bg: 'rgba(242,184,75,0.1)', icon: <Activity className="w-6 h-6 text-[var(--status-warn)]" /> },
        LOW_RISK: { color: '#3b82f6', bg: 'rgba(59,130,246,0.1)', icon: <Check className="w-6 h-6 text-blue-500" /> },
        CLEAN: { color: 'var(--status-safe)', bg: 'rgba(61,220,132,0.1)', icon: <Check className="w-6 h-6 text-[var(--status-safe)]" /> },
    }[verdict] || { color: 'var(--text-muted)', bg: 'var(--bg-panel)', icon: <HelpCircle className="w-6 h-6 text-textMuted" /> };

    return (
        <div className="flex items-center gap-4 px-5 py-3 rounded-xl border" style={{ background: cfg.bg, borderColor: `${cfg.color}40` }}>
            {cfg.icon}
            <div>
                <div className="text-sm font-bold font-mono tracking-wide" style={{ color: cfg.color }}>{verdict}</div>
                {score != null && <div className="text-[10px] font-mono text-textMuted uppercase tracking-widest mt-1">Suspicion: {(score * 100).toFixed(1)}%</div>}
            </div>
        </div>
    );
}

function DatasetInfoBanner({ info, dataSource }) {
    if (!info) return null;
    return (
        <div className="bg-bgPanel border border-borderHairline rounded-xl px-5 py-4 text-xs font-mono text-textSecondary flex gap-6 flex-wrap items-center">
            <span className="flex items-center gap-2"><Folder className="w-4 h-4 text-redPrimary" /> <strong className="text-textPrimary">{info.filename}</strong></span>
            <span className="text-borderHairline">|</span>
            <span>Rows: <strong className="text-textPrimary">{info.n_rows?.toLocaleString()}</strong></span>
            <span className="text-borderHairline">|</span>
            <span>Features: <strong className="text-textPrimary">{info.n_features}</strong></span>
            <span className="text-borderHairline">|</span>
            <span>Mode: <strong className="text-redPrimary">{info.detection_mode}</strong></span>
            {dataSource && (
                <>
                    <span className="text-borderHairline">|</span>
                    <span>Source: <strong className="text-redPrimary">{dataSource}</strong></span>
                </>
            )}
        </div>
    );
}

export default function ForensicsPage() {
    const [activeTab, setActiveTab] = useState('auto');   // 'auto' | 'demo' | 'upload'
    const [forensics, setForensics] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const load = useCallback(async (src = activeTab) => {
        setLoading(true);
        setError(null);
        try {
            const data = await api.getLatestForensics(src);
            setForensics(data);
        } catch {
            setError(
                src === 'upload'
                    ? 'No uploaded dataset analysis found. Upload a CSV first from the Upload Dataset page.'
                    : 'No forensic data yet. Run the demo from the Trust Dashboard first.'
            );
        } finally {
            setLoading(false);
        }
    }, [activeTab]);

    useEffect(() => { load(activeTab); }, [activeTab, load]);

    const TABS = [
        { id: 'auto', label: 'Latest (Auto)', icon: Target },
        { id: 'demo', label: 'Demo Analysis', icon: Target },
        { id: 'upload', label: 'My Upload', icon: Upload },
    ];

    return (
        <div className="relative z-10 px-6 md:px-12 py-12 max-w-6xl mx-auto flex flex-col gap-8 min-h-[calc(100vh-80px)] animate-fadeInUp">
            <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
                <div>
                    <div className="font-mono text-xs text-redPrimary tracking-widest uppercase mb-3 flex items-center gap-2 font-bold">
                        <Search className="w-4 h-4" /> Forensic Analysis
                    </div>
                    <h1 className="text-[48px] font-display font-bold text-textPrimary m-0 tracking-tight leading-none">
                        Poisoning <span className="text-redPrimary">Investigation</span>
                    </h1>
                    <p className="font-mono text-[13px] text-textMuted mt-4 uppercase tracking-widest">
                        Heuristic pattern analysis // Threat signals
                    </p>
                </div>
                <button onClick={() => load(activeTab)} disabled={loading}
                    className="px-6 py-2.5 border border-redPrimary/30 bg-redPrimary/10 text-redPrimary font-mono text-sm font-bold rounded-xl hover:bg-redPrimary/20 transition-all uppercase tracking-widest disabled:opacity-50">
                    {loading ? 'Analyzing...' : 'Refresh Scan'}
                </button>
            </div>

            {/* Source Tabs */}
            <div className="flex gap-3 overflow-x-auto pb-2 border-b border-borderHairline/50">
                {TABS.map(({ id, label, icon: TabIcon }) => (
                    <button key={id} onClick={() => setActiveTab(id)}
                        className={`flex items-center gap-2 px-5 py-2.5 rounded-t-xl font-mono text-xs uppercase tracking-widest font-bold transition-all
                        ${activeTab === id 
                            ? 'bg-redPrimary/10 text-redPrimary border-b-2 border-redPrimary' 
                            : 'bg-transparent text-textMuted hover:text-textPrimary hover:bg-bgPanel'}`}>
                        <TabIcon className="w-4 h-4" /> {label}
                    </button>
                ))}
            </div>

            {error && (
                <div className="bg-statusWarn/10 border border-statusWarn/30 rounded-xl p-5 font-mono text-sm text-statusWarn flex items-center gap-3">
                    <AlertTriangle className="w-5 h-5" /> {error}
                </div>
            )}

            {!forensics && !error && !loading && (
                <div className="flex flex-col items-center justify-center py-32 text-textMuted font-mono text-sm border-2 border-dashed border-borderHairline rounded-2xl">
                    <Search className="w-12 h-12 mb-4 opacity-20" />
                    No forensic data yet. Run the demo or upload a CSV first.
                </div>
            )}

            {forensics && (
                <div className="space-y-6">
                    {/* Verdict + dataset info */}
                    <div className="flex gap-4 items-center flex-wrap">
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

                    {/* Proxy comparison */}
                    {forensics.counterfactual && (
                        <div className="bg-bgPanel border border-borderHairline rounded-xl p-6">
                            <div className="font-mono text-xs text-redPrimary uppercase tracking-widest mb-4 font-bold flex items-center gap-2">
                                <Activity className="w-4 h-4" /> Proxy Comparison and Limits
                            </div>
                            <div className="bg-bgVoid border border-borderHairline rounded-lg p-5 font-mono text-sm">
                                <div className="text-textPrimary font-bold">Proxy accuracy effect: <span className="text-redPrimary">{((forensics.counterfactual.proxy_accuracy_effect || 0) * 100).toFixed(1)}%</span></div>
                                <div className="text-textMuted mt-3 text-xs leading-relaxed border-t border-borderHairline pt-3">{forensics.counterfactual.limitation}</div>
                            </div>
                        </div>
                    )}
                </div>
            )}
            
            <style>{`@keyframes fadeInUp { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:translateY(0); } }`}</style>
        </div>
    );
}
