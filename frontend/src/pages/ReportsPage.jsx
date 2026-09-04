import { useState } from 'react';
import { api } from '../services/api';
import { FileText, Download, CheckCircle, Loader, Upload, Target, Activity } from 'lucide-react';

const SOURCE_CONFIG = {
    auto: { label: 'Latest (Auto)', icon: Activity, desc: 'Uses your most recent upload; falls back to demo if none.' },
    upload: { label: 'My Uploaded Dataset', icon: Upload, desc: 'Generate report from the CSV you uploaded.' },
    demo: { label: 'Demo Analysis', icon: Target, desc: 'Generate report from the built-in demo run.' },
};

export default function ReportsPage() {
    const [report, setReport] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [source, setSource] = useState('auto');

    const generateReport = async () => {
        setLoading(true);
        setError(null);
        try {
            const r = await api.generateReport(source);
            setReport(r);
        } catch {
            setError(
                source === 'upload'
                    ? 'No uploaded dataset found. Upload a CSV from the Upload Dataset page first.'
                    : source === 'demo'
                        ? 'No demo run found. Click "Run Demo" on the Trust Dashboard first.'
                        : 'No analysis results found. Upload a CSV or run the demo first.'
            );
        } finally {
            setLoading(false);
        }
    };

    const downloadReport = () => {
        if (!report) return;
        const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `forensic_report_${report.report_id?.slice(0, 8)}.json`;
        a.click();
        URL.revokeObjectURL(url);
    };

    const srcCfg = SOURCE_CONFIG[source];

    return (
        <div className="relative z-10 px-6 md:px-12 py-12 max-w-7xl mx-auto flex flex-col gap-8 min-h-[calc(100vh-80px)] animate-fadeInUp">
            {/* Header */}
            <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
                <div>
                    <div className="font-mono text-xs text-redPrimary tracking-widest uppercase mb-3 flex items-center gap-2 font-bold">
                        <FileText className="w-4 h-4" /> Evidence Package
                    </div>
                    <h1 className="text-[48px] font-display font-bold text-textPrimary m-0 tracking-tight leading-none">
                        Forensic <span className="text-redPrimary">Reports</span>
                    </h1>
                    <p className="font-mono text-[13px] text-textMuted mt-4 uppercase tracking-widest">
                        Court-admissible evidence // NIST AI RMF // EU AI Act compliance
                    </p>
                </div>
                <div className="flex gap-3">
                    <button onClick={generateReport} disabled={loading}
                        className="flex items-center gap-2 px-6 py-2.5 bg-redPrimary/10 border border-redPrimary/40 text-redPrimary font-mono text-sm font-bold rounded-xl hover:bg-redPrimary/20 transition-all uppercase tracking-widest disabled:opacity-50">
                        {loading ? <Loader className="w-4 h-4 animate-spin" /> : <FileText className="w-4 h-4" />}
                        Generate Report
                    </button>
                    {report && (
                        <button onClick={downloadReport}
                            className="flex items-center gap-2 px-6 py-2.5 border border-borderHairline text-textSecondary font-mono text-sm font-bold rounded-xl hover:border-textSecondary hover:text-textPrimary transition-all uppercase tracking-widest">
                            <Download className="w-4 h-4" /> Download JSON
                        </button>
                    )}
                </div>
            </div>

            {/* Source Selector */}
            <div className="bg-bgPanel border border-borderHairline rounded-[24px] p-6 lg:p-8">
                <div className="font-mono text-xs text-redPrimary font-bold uppercase tracking-widest mb-4 flex items-center gap-2">
                    <Activity className="w-4 h-4" /> Report Data Source
                </div>
                <div className="flex gap-3 flex-wrap mb-4">
                    {Object.entries(SOURCE_CONFIG).map(([id, cfg]) => {
                        const isSelected = source === id;
                        const Icon = cfg.icon;
                        return (
                            <button key={id} onClick={() => { setSource(id); setReport(null); setError(null); }}
                                className={`flex items-center gap-2 px-5 py-3 rounded-xl font-mono text-xs font-bold transition-all uppercase tracking-widest border
                                ${isSelected ? 'bg-redPrimary/10 border-redPrimary/50 text-redPrimary' : 'bg-transparent border-borderHairline text-textMuted hover:border-textMuted hover:text-textSecondary'}`}>
                                <Icon className="w-4 h-4" />
                                {cfg.label}
                            </button>
                        );
                    })}
                </div>
                <p className="font-mono text-xs text-textSecondary bg-bgVoid px-4 py-3 rounded-lg border border-borderHairline inline-block">{srcCfg.desc}</p>
            </div>

            {error && (
                <div className="bg-statusWarn/10 border border-statusWarn/30 rounded-xl p-5 font-mono text-sm text-statusWarn font-bold flex items-center gap-3">
                    <AlertTriangle className="w-5 h-5" /> {error}
                </div>
            )}

            {!report && !error && !loading && (
                <div className="mt-4">
                    <div className="text-center py-12 text-textMuted font-mono text-sm uppercase tracking-widest">
                        Select a data source above and click "Generate Report".
                    </div>
                    {/* Skeleton Preview */}
                    <div className="space-y-6 opacity-30 pointer-events-none filter grayscale">
                        <div className="h-32 bg-bgPanel border border-borderHairline rounded-[24px] animate-pulse"></div>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
                            <div className="h-28 bg-bgPanel border border-borderHairline rounded-[24px] animate-pulse"></div>
                            <div className="h-28 bg-bgPanel border border-borderHairline rounded-[24px] animate-pulse"></div>
                            <div className="h-28 bg-bgPanel border border-borderHairline rounded-[24px] animate-pulse"></div>
                            <div className="h-28 bg-bgPanel border border-borderHairline rounded-[24px] animate-pulse"></div>
                            <div className="h-28 bg-bgPanel border border-borderHairline rounded-[24px] animate-pulse"></div>
                            <div className="h-28 bg-bgPanel border border-borderHairline rounded-[24px] animate-pulse"></div>
                        </div>
                    </div>
                </div>
            )}

            {loading && (
                <div className="mt-4">
                    <div className="text-center py-12 text-redPrimary font-mono text-sm uppercase tracking-widest animate-pulse flex flex-col items-center justify-center gap-4">
                        <Loader className="w-8 h-8 animate-spin" />
                        Generating forensic evidence package...
                    </div>
                    {/* Skeleton Preview */}
                    <div className="space-y-6 opacity-50 pointer-events-none">
                        <div className="h-32 bg-bgPanel border border-borderHairline rounded-[24px] animate-pulse border-redPrimary/20"></div>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
                            <div className="h-28 bg-bgPanel border border-borderHairline rounded-[24px] animate-pulse"></div>
                            <div className="h-28 bg-bgPanel border border-borderHairline rounded-[24px] animate-pulse"></div>
                            <div className="h-28 bg-bgPanel border border-borderHairline rounded-[24px] animate-pulse"></div>
                        </div>
                    </div>
                </div>
            )}

            {report && (
                <div className="space-y-6 animate-fadeInUp">
                    {/* Report Header */}
                    <div className="bg-bgPanel border border-redPrimary/30 rounded-[24px] p-8 relative overflow-hidden">
                        <div className="absolute top-0 right-0 w-64 h-64 bg-redPrimary/5 rounded-full blur-3xl -mr-20 -mt-20"></div>
                        <div className="flex items-start justify-between mb-4 relative z-10">
                            <div>
                                <div className="font-mono text-[10px] text-redPrimary font-bold uppercase tracking-widest mb-2 px-2 py-0.5 rounded border border-redPrimary/30 bg-redPrimary/10 inline-block">
                                    {report.platform}
                                </div>
                                <h2 className="text-3xl font-black text-textPrimary tracking-tight mb-2">{report.title}</h2>
                                <div className="font-mono text-[11px] text-textSecondary uppercase tracking-widest flex items-center gap-3">
                                    <span>ID: {report.report_id}</span>
                                    <span className="text-borderHairline">|</span>
                                    <span>{report.generated_at ? new Date(report.generated_at.endsWith('Z') ? report.generated_at : report.generated_at + 'Z').toLocaleString() : '—'}</span>
                                </div>
                                <div className="font-mono text-[11px] text-statusSafe mt-2 uppercase tracking-widest">
                                    Source: {report.data_source} · {report.dataset_info?.filename || 'demo dataset'}
                                    {report.dataset_info?.n_rows && ` · ${report.dataset_info.n_rows.toLocaleString()} rows`}
                                </div>
                            </div>
                            <CheckCircle className="w-12 h-12 text-statusSafe drop-shadow-md" />
                        </div>
                    </div>

                    {/* Executive Summary */}
                    <div className="bg-bgPanel border border-borderHairline rounded-[24px] p-8">
                        <div className="font-mono text-xs text-redPrimary font-bold uppercase tracking-widest mb-6 flex items-center gap-2">
                            <Target className="w-4 h-4" /> Executive Summary
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                            {[
                                { label: 'Verdict', value: report.executive_summary?.verdict, color: report.executive_summary?.verdict === 'CONFIRMED_POISONED' ? 'statusCritical' : report.executive_summary?.verdict === 'CLEAN' ? 'statusSafe' : 'statusWarn' },
                                { label: 'Attack Type', value: (report.executive_summary?.attack_type || '—').replace(/_/g, ' '), color: 'textSecondary' },
                                { label: 'Confidence', value: `${((report.executive_summary?.confidence || 0) * 100).toFixed(1)}%`, color: 'textPrimary' },
                                { label: 'Causal Effect', value: `${((report.executive_summary?.causal_effect || 0) * 100).toFixed(1)}%`, color: 'statusWarn' },
                                { label: 'Sophistication', value: `${report.executive_summary?.sophistication_score || 0}/10`, color: 'statusCritical' },
                                { label: 'Models Impacted', value: report.executive_summary?.blast_radius_summary?.models ?? '—', color: 'redPrimary' },
                            ].map(({ label, value, color }) => (
                                <div key={label} className="bg-bgVoid border border-borderHairline rounded-xl p-5 hover:border-redPrimary/20 transition-all">
                                    <div className="font-mono text-[10px] text-textMuted uppercase tracking-widest font-bold mb-2">{label}</div>
                                    <div className={`font-mono text-lg font-black uppercase tracking-wider text-${color}`}>{value}</div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Layer Evidence */}
                    {report.layer_scores && (
                        <div className="bg-bgPanel border border-borderHairline rounded-[24px] p-8">
                            <div className="font-mono text-xs text-redPrimary font-bold uppercase tracking-widest mb-6 flex items-center gap-2">
                                <Activity className="w-4 h-4" /> Detection Layer Evidence
                            </div>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                {Object.entries(report.layer_scores).map(([layer, score]) => {
                                    const pct = Math.round((score || 0) * 100);
                                    const color = pct > 70 ? 'var(--status-critical)' : pct > 40 ? 'var(--status-warn)' : 'var(--status-safe)';
                                    return (
                                        <div key={layer} className="bg-bgVoid border border-borderHairline rounded-xl p-5">
                                            <div className="font-mono text-[10px] text-textMuted uppercase tracking-widest font-bold mb-3 truncate">{layer.replace(/_/g, ' ')}</div>
                                            <div className="font-mono text-3xl font-black mb-4" style={{ color }}>{pct}%</div>
                                            <div className="h-1.5 bg-bgPanel rounded-full overflow-hidden">
                                                <div className="h-full rounded-full transition-all duration-1000" style={{ width: `${pct}%`, background: color }} />
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    )}

                    {/* Compliance */}
                    <div className="bg-bgPanel border border-borderHairline rounded-[24px] p-8">
                        <div className="font-mono text-xs text-redPrimary font-bold uppercase tracking-widest mb-6 flex items-center gap-2">
                            <ShieldCheck className="w-4 h-4" /> Regulatory Compliance
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="bg-bgVoid border border-borderHairline rounded-xl p-5">
                                <div className="font-mono text-[10px] text-textMuted uppercase tracking-widest font-bold mb-3">NIST AI RMF</div>
                                <div className="font-mono text-xs text-statusSafe">{report.compliance?.nist_ai_rmf}</div>
                            </div>
                            <div className="bg-bgVoid border border-borderHairline rounded-xl p-5">
                                <div className="font-mono text-[10px] text-textMuted uppercase tracking-widest font-bold mb-3">EU AI Act</div>
                                <div className="font-mono text-xs text-statusSafe">{report.compliance?.eu_ai_act}</div>
                            </div>
                            <div className="bg-bgVoid border border-borderHairline rounded-xl p-5 md:col-span-2">
                                <div className="font-mono text-[10px] text-textMuted uppercase tracking-widest font-bold mb-3">Audit Hash</div>
                                <div className="font-mono text-xs text-textSecondary break-all">{report.compliance?.audit_hash}</div>
                            </div>
                        </div>
                    </div>

                    {/* Attack Narrative */}
                    {report.attack_narrative && (
                        <div className="bg-bgVoid border border-redPrimary/20 rounded-[24px] p-8">
                            <div className="font-mono text-xs text-redPrimary font-bold uppercase tracking-widest mb-4 flex items-center gap-2">
                                <FileText className="w-4 h-4" /> Attack Narrative
                            </div>
                            <pre className="font-mono text-[13px] text-textSecondary whitespace-pre-wrap leading-loose border-l-2 border-redPrimary/50 pl-5">
                                {report.attack_narrative}
                            </pre>
                        </div>
                    )}

                    {/* Defense Actions */}
                    {report.defense_actions?.length > 0 && (
                        <div className="bg-bgPanel border border-borderHairline rounded-[24px] p-8">
                            <div className="font-mono text-xs text-redPrimary font-bold uppercase tracking-widest mb-6 flex items-center gap-2">
                                <Shield className="w-4 h-4" /> Defense Actions Taken
                            </div>
                            <div className="space-y-4">
                                {report.defense_actions.map((action, i) => (
                                    <div key={i} className="bg-bgVoid border border-borderHairline rounded-xl p-5 hover:border-redPrimary/30 transition-all">
                                        <div className="flex items-center justify-between mb-3">
                                            <span className="font-mono text-xs font-bold text-redPrimary uppercase tracking-widest">{action.action}</span>
                                            <span className="font-mono text-[10px] text-textMuted">{action.timestamp?.slice(0, 19)}</span>
                                        </div>
                                        <div className="font-mono text-sm text-textSecondary mb-2">{action.reason}</div>
                                        <div className="font-mono text-[11px] text-textMuted uppercase tracking-widest">
                                            {action.samples_affected} samples affected
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}
            
            <style>{`
                @keyframes fadeInUp { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:translateY(0); } }
            `}</style>
        </div>
    );
}
