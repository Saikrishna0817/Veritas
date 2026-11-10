import { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import { PreviewTag } from '../components/ui';
import { Search, Upload, Target, ShieldAlert, Activity, Check, HelpCircle, Folder, AlertTriangle, Fingerprint, Bug, BrainCircuit } from 'lucide-react';

function NarrativeBox({ narrative }) {
    if (!narrative) return null;
    return (
        <div className="bg-bgPanel border border-borderHairline rounded-[28px] p-6 font-mono text-sm relative overflow-hidden group">
            <div className="text-redPrimary tracking-wide mb-4 text-xs font-medium flex items-center gap-2">
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
        <div className="bg-bgPanel border border-borderHairline rounded-[28px] p-6 hover:border-redPrimary/30 transition-colors">
            <div className="text-xs text-redPrimary mb-6 font-medium flex items-center gap-2">
                <Bug className="w-4 h-4" /> Classification
            </div>
            <div className="space-y-4">
                <div className="flex justify-between items-center pb-3 border-b border-borderHairline">
                    <span className="font-mono text-xs text-textMuted">Attack Type</span>
                    <span className="text-sm font-medium text-statusCritical capitalize">
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
                    <span className="text-xs font-medium text-redPrimary">
                        {(classification.confidence * 100).toFixed(1)}%
                    </span>
                </div>
                <div className="flex justify-between items-center pb-3 border-b border-borderHairline">
                    <span className="font-mono text-xs text-textMuted">Severity</span>
                    <span className={`text-xs font-medium ${severityColor} uppercase`}>
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
                                <div className="h-full bg-redPrimary shadow-lift rounded-full transition-all"
                                    style={{ width: `${prob * 100}%` }} />
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

function PatternCard({ pattern }) {
    if (!pattern) return null;
    return (
        <div className="bg-bgPanel border border-borderHairline rounded-[28px] p-6 hover:border-redPrimary/30 transition-colors">
            <div className="text-xs text-redPrimary mb-6 font-medium flex items-center gap-2">
                <Fingerprint className="w-4 h-4" /> Flagged Rows
            </div>
            <div className="space-y-0 font-mono text-xs">
                {[
                    { label: 'Poisoned Samples', value: pattern.n_poisoned_samples },
                    { label: 'Sigma Shift', value: pattern.sigma_shift != null ? `${pattern.sigma_shift}σ` : null },
                ].map(({ label, value }) => value != null && (
                    <div key={label} className="flex justify-between py-3 border-b border-borderHairline last:border-0 items-center">
                        <span className="text-textMuted">{label}</span>
                        <span className="text-textPrimary font-medium bg-bgVoid px-2 py-1 rounded">{value}</span>
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
        <div className="flex items-center gap-4 px-5 py-3 rounded-2xl border" style={{ background: cfg.bg, borderColor: `${cfg.color}40` }}>
            {cfg.icon}
            <div>
                <div className="text-sm font-medium" style={{ color: cfg.color }}>{verdict}</div>
                {score != null && <div className="text-[10px] text-textMuted mt-1">Suspicion: {(score * 100).toFixed(1)}%</div>}
            </div>
        </div>
    );
}

function DatasetInfoBanner({ info, dataSource }) {
    if (!info) return null;
    return (
        <div className="bg-bgPanel border border-borderHairline rounded-2xl px-5 py-4 text-xs font-mono text-textSecondary flex gap-6 flex-wrap items-center">
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

const TABS = [
    { id: 'auto', label: 'Latest (Auto)', icon: Target },
    { id: 'demo', label: 'Demo Analysis', icon: Target },
    { id: 'upload', label: 'My Upload', icon: Upload },
];

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

    return (
        <div className="relative z-10 px-5 md:px-10 py-10 md:py-14 max-w-[1360px] mx-auto flex flex-col gap-8 min-h-[calc(100vh-80px)] animate-fadeInUp">
            <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
                <div className="max-w-3xl">
                    <h1 className="font-display font-medium text-[clamp(2.2rem,5vw,4.2rem)] leading-[0.95] text-textPrimary">Forensics{" "}<PreviewTag className="ml-3" /></h1>
                    <p className="mt-4 text-textSecondary text-[17px] leading-relaxed max-w-[62ch]">Heuristic reading of the latest flagged batch: which pattern the flagged rows resemble and how they arrived. Treat it as a lead to check, not a finding.</p>
                </div>
                <button onClick={() => load(activeTab)} disabled={loading}
                    className="px-6 py-2.5 border border-redPrimary/30 bg-redPrimary/10 text-redPrimary text-sm font-medium rounded-2xl hover:bg-redPrimary/20 transition-all disabled:opacity-50">
                    {loading ? 'Analyzing...' : 'Refresh Scan'}
                </button>
            </div>

            {/* Source Tabs */}
            <div className="flex gap-3 overflow-x-auto pb-2 border-b border-borderHairline/50">
                {TABS.map((tab) => {
                    const TabIcon = tab.icon;
                    return (
                        <button key={tab.id} onClick={() => setActiveTab(tab.id)}
                            className={`flex items-center gap-2 px-4 py-2 rounded-2xl border text-sm font-medium transition-colors ${activeTab === tab.id ? 'border-uv bg-uv/[0.08] text-uv' : 'border-borderHairline text-textSecondary hover:border-borderStrong hover:text-textPrimary'}`}>
                            <TabIcon className="w-4 h-4" /> {tab.label}
                        </button>
                    );
                })}
            </div>

            {error && (
                <div className="bg-statusWarn/10 border border-statusWarn/30 rounded-2xl p-5 font-mono text-sm text-statusWarn flex items-center gap-3">
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
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <AttackClassCard classification={forensics.attack_classification} />
                        <PatternCard pattern={forensics.injection_pattern} />
                    </div>
                </div>
            )}
            
            <style>{`@keyframes fadeInUp { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:translateY(0); } }`}</style>
        </div>
    );
}
