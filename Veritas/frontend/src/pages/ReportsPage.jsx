import { useState } from 'react';
import { api } from '../services/api';
import { FileText, Download, CheckCircle, Loader } from 'lucide-react';

export default function ReportsPage() {
    const [report, setReport] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const generateReport = async () => {
        setLoading(true);
        setError(null);
        try {
            const r = await api.generateReport();
            setReport(r);
        } catch (e) {
            setError('Run the demo first from the Dashboard.');
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

    return (
        <div className="p-8 space-y-8 animate-in">
            <div className="flex items-start justify-between">
                <div>
                    <div className="font-mono text-xs text-yellow tracking-widest uppercase mb-2 flex items-center gap-2">
                        <FileText className="w-3 h-3" /> Evidence Package
                    </div>
                    <h1 className="text-4xl font-bold tracking-tight">
                        Forensic <span className="text-yellow">Reports</span>
                    </h1>
                    <p className="font-mono text-sm text-text2 mt-2">
                        Court-admissible evidence · NIST AI RMF · EU AI Act compliance
                    </p>
                </div>
                <div className="flex gap-3">
                    <button onClick={generateReport} disabled={loading}
                        className="flex items-center gap-2 px-5 py-2.5 bg-yellow/10 border border-yellow/40 text-yellow font-mono text-xs rounded-lg hover:bg-yellow/20 transition-all disabled:opacity-50">
                        {loading ? <Loader className="w-3 h-3 animate-spin" /> : <FileText className="w-3 h-3" />}
                        Generate Report
                    </button>
                    {report && (
                        <button onClick={downloadReport}
                            className="flex items-center gap-2 px-5 py-2.5 bg-accent/10 border border-accent/40 text-accent font-mono text-xs rounded-lg hover:bg-accent/20 transition-all">
                            <Download className="w-3 h-3" /> Download JSON
                        </button>
                    )}
                </div>
            </div>

            {error && (
                <div className="bg-yellow/10 border border-yellow/30 rounded-lg p-4 font-mono text-sm text-yellow">
                    ⚠ {error}
                </div>
            )}

            {!report && !error && (
                <div className="text-center py-20 text-text3 font-mono text-sm">
                    <FileText className="w-12 h-12 mx-auto mb-4 opacity-20" />
                    Click "Generate Report" to create a forensic evidence package.
                </div>
            )}

            {report && (
                <div className="space-y-6">
                    {/* Report Header */}
                    <div className="bg-surface border border-yellow/20 rounded-lg p-6">
                        <div className="flex items-start justify-between mb-4">
                            <div>
                                <div className="font-mono text-xs text-yellow uppercase tracking-widest mb-1">
                                    {report.platform}
                                </div>
                                <h2 className="text-2xl font-bold text-text1">{report.title}</h2>
                                <div className="font-mono text-xs text-text3 mt-1">
                                    Report ID: {report.report_id} · Generated: {report.generated_at?.slice(0, 19)} UTC
                                </div>
                            </div>
                            <CheckCircle className="w-8 h-8 text-accent3" />
                        </div>
                    </div>

                    {/* Executive Summary */}
                    <div className="bg-surface border border-border rounded-lg p-6">
                        <div className="font-mono text-xs text-accent uppercase tracking-widest mb-4">
                            Executive Summary
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                            {[
                                { label: 'Verdict', value: report.executive_summary?.verdict, color: report.executive_summary?.verdict === 'CONFIRMED_POISONED' ? 'danger' : 'accent3' },
                                { label: 'Attack Type', value: report.executive_summary?.attack_type?.replace(/_/g, ' '), color: 'text2' },
                                { label: 'Confidence', value: `${(report.executive_summary?.confidence * 100).toFixed(1)}%`, color: 'accent' },
                                { label: 'Causal Effect', value: `${(report.executive_summary?.causal_effect * 100).toFixed(1)}%`, color: 'danger' },
                                { label: 'Sophistication', value: `${report.executive_summary?.sophistication_score}/10`, color: 'orange' },
                                { label: 'Models Impacted', value: report.executive_summary?.blast_radius_summary?.models, color: 'yellow' },
                            ].map(({ label, value, color }) => (
                                <div key={label} className="bg-bg3 rounded-lg p-4">
                                    <div className="font-mono text-xs text-text3 mb-1">{label}</div>
                                    <div className={`font-bold text-${color} capitalize`}>{value}</div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Compliance */}
                    <div className="bg-surface border border-border rounded-lg p-6">
                        <div className="font-mono text-xs text-accent3 uppercase tracking-widest mb-4">
                            Regulatory Compliance
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="bg-bg3 rounded-lg p-4">
                                <div className="font-mono text-xs text-text3 mb-2">NIST AI RMF</div>
                                <div className="font-mono text-xs text-accent3">{report.compliance?.nist_ai_rmf}</div>
                            </div>
                            <div className="bg-bg3 rounded-lg p-4">
                                <div className="font-mono text-xs text-text3 mb-2">EU AI Act</div>
                                <div className="font-mono text-xs text-accent3">{report.compliance?.eu_ai_act}</div>
                            </div>
                            <div className="bg-bg3 rounded-lg p-4 md:col-span-2">
                                <div className="font-mono text-xs text-text3 mb-2">Audit Hash</div>
                                <div className="font-mono text-xs text-text2 break-all">{report.compliance?.audit_hash}</div>
                            </div>
                        </div>
                    </div>

                    {/* Attack Narrative */}
                    {report.attack_narrative && (
                        <div className="bg-bg2 border border-border rounded-lg p-6">
                            <div className="font-mono text-xs text-accent uppercase tracking-widest mb-3">
                                Attack Narrative
                            </div>
                            <pre className="font-mono text-xs text-text2 whitespace-pre-wrap leading-relaxed">
                                {report.attack_narrative}
                            </pre>
                        </div>
                    )}

                    {/* Defense Actions */}
                    {report.defense_actions?.length > 0 && (
                        <div className="bg-surface border border-border rounded-lg p-6">
                            <div className="font-mono text-xs text-purple uppercase tracking-widest mb-4">
                                Defense Actions Taken
                            </div>
                            <div className="space-y-3">
                                {report.defense_actions.map((action, i) => (
                                    <div key={i} className="bg-bg3 rounded-lg p-4 border border-purple/20">
                                        <div className="flex items-center justify-between mb-2">
                                            <span className="font-mono text-xs font-bold text-purple uppercase">{action.action}</span>
                                            <span className="font-mono text-xs text-text3">{action.timestamp?.slice(0, 19)}</span>
                                        </div>
                                        <div className="font-mono text-xs text-text3">{action.reason}</div>
                                        <div className="font-mono text-xs text-text2 mt-1">
                                            {action.samples_affected} samples affected
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
