import { useState, useEffect } from 'react';
import { api } from '../services/api';
import {
    LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, ReferenceLine, ComposedChart, Area
} from 'recharts';
import { Shield, AlertTriangle, CheckCircle, Play, Loader, TrendingDown, Zap, Eye } from 'lucide-react';

// ── Sub-components ────────────────────────────────────────────────────────────

function MetricCard({ label, value, unit = '', color = 'accent', sublabel }) {
    return (
        <div className="bg-surface border border-border rounded-lg p-5 relative overflow-hidden">
            <div className={`absolute top-0 left-0 w-0.5 h-full bg-${color}`} />
            <div className="font-mono text-xs text-text3 uppercase tracking-widest mb-2">{label}</div>
            <div className={`text-3xl font-bold tracking-tight text-${color}`}>
                {value}<span className="text-lg ml-1 text-text3">{unit}</span>
            </div>
            {sublabel && <div className="font-mono text-xs text-text3 mt-1">{sublabel}</div>}
        </div>
    );
}

function VerdictBadge({ verdict }) {
    const cfg = {
        CONFIRMED_POISONED: { color: 'danger', icon: AlertTriangle, label: 'ATTACK CONFIRMED' },
        SUSPICIOUS: { color: 'yellow', icon: Eye, label: 'SUSPICIOUS' },
        CLEAN: { color: 'accent3', icon: CheckCircle, label: 'CLEAN' },
    }[verdict] || { color: 'text3', icon: Shield, label: 'UNKNOWN' };

    const Icon = cfg.icon;
    return (
        <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-md border border-${cfg.color}/40 bg-${cfg.color}/10`}>
            <Icon className={`w-4 h-4 text-${cfg.color}`} />
            <span className={`font-mono text-sm font-bold text-${cfg.color} tracking-widest`}>{cfg.label}</span>
        </div>
    );
}

function LayerScoreBar({ name, score }) {
    const pct = Math.round(score * 100);
    const color = pct > 70 ? '#ff4d6a' : pct > 40 ? '#ffd166' : '#00ffc8';
    return (
        <div className="mb-3">
            <div className="flex justify-between font-mono text-xs mb-1">
                <span className="text-text2">{name}</span>
                <span style={{ color }}>{pct}%</span>
            </div>
            <div className="h-1.5 bg-border rounded-full overflow-hidden">
                <div className="h-full rounded-full transition-all duration-1000"
                    style={{ width: `${pct}%`, background: color }} />
            </div>
        </div>
    );
}

const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    return (
        <div className="bg-bg2 border border-border rounded-lg p-3 font-mono text-xs">
            <div className="text-text3 mb-2">{label}</div>
            {payload.map((p, i) => (
                <div key={i} style={{ color: p.color }} className="flex justify-between gap-4">
                    <span>{p.name}</span>
                    <span>{typeof p.value === 'number' ? p.value.toFixed(3) : p.value}</span>
                </div>
            ))}
        </div>
    );
};

// ── Main Dashboard ────────────────────────────────────────────────────────────

export default function Dashboard({ wsEvents = [] }) {
    const [demoResult, setDemoResult] = useState(null);
    const [trustScore, setTrustScore] = useState(null);
    const [timeline, setTimeline] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Load trust score on mount
    useEffect(() => {
        api.getTrustScore().then(setTrustScore).catch(() => { });
        api.getAttackTimeline().then(d => setTimeline(d.timeline || [])).catch(() => { });
    }, []);

    // Update trust score when demo runs
    useEffect(() => {
        if (wsEvents.some(e => e.event === 'attack_confirmed')) {
            api.getTrustScore().then(setTrustScore).catch(() => { });
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

    const layerScores = demoResult?.layer_scores || {};
    const verdict = demoResult?.verdict || 'CLEAN';
    const suspicion = demoResult?.overall_suspicion_score || 0;

    return (
        <div className="p-8 space-y-8 animate-in">
            {/* Header */}
            <div className="flex items-start justify-between">
                <div>
                    <div className="font-mono text-xs text-accent tracking-widest uppercase mb-2 flex items-center gap-2">
                        <span className="pulse-dot" /> Real-Time AI Immune System
                    </div>
                    <h1 className="text-4xl font-bold tracking-tight">
                        Trust <span className="text-accent">Dashboard</span>
                    </h1>
                    <p className="font-mono text-sm text-text2 mt-2 max-w-lg">
                        5-layer poisoning detection · Causal proof engine · Auto-defense
                    </p>
                </div>
                <button
                    onClick={handleRunDemo}
                    disabled={loading}
                    className="flex items-center gap-2 px-6 py-3 bg-accent/10 border border-accent/40 rounded-lg
                     font-mono text-sm text-accent hover:bg-accent/20 transition-all duration-200
                     disabled:opacity-50 disabled:cursor-not-allowed glow-accent"
                >
                    {loading ? <Loader className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                    {loading ? 'Analyzing...' : 'Run Demo Analysis'}
                </button>
            </div>

            {error && (
                <div className="bg-danger/10 border border-danger/30 rounded-lg p-4 font-mono text-sm text-danger">
                    ⚠ Backend not running. Start it with: <code className="bg-bg3 px-2 py-0.5 rounded">cd backend && uvicorn app.main:app --reload</code>
                    <div className="mt-1 text-xs text-text3">{error}</div>
                </div>
            )}

            {/* Verdict Banner */}
            {demoResult && (
                <div className={`rounded-lg border p-6 ${verdict === 'CONFIRMED_POISONED' ? 'border-danger/40 bg-danger/5' :
                        verdict === 'SUSPICIOUS' ? 'border-yellow/40 bg-yellow/5' :
                            'border-accent3/40 bg-accent3/5'
                    }`}>
                    <div className="flex items-center justify-between flex-wrap gap-4">
                        <div>
                            <VerdictBadge verdict={verdict} />
                            <p className="font-mono text-sm text-text2 mt-2">
                                Overall suspicion score: <span className="text-text1 font-bold">{(suspicion * 100).toFixed(1)}%</span>
                                {' · '}{demoResult.n_samples} samples analyzed in {demoResult.elapsed_ms}ms
                            </p>
                        </div>
                        {demoResult.attack_classification && (
                            <div className="text-right">
                                <div className="font-mono text-xs text-text3 uppercase tracking-widest">Attack Type</div>
                                <div className="font-bold text-danger text-lg capitalize">
                                    {demoResult.attack_classification.attack_type?.replace(/_/g, ' ')}
                                </div>
                                <div className="font-mono text-xs text-text3">
                                    {(demoResult.attack_classification.confidence * 100).toFixed(1)}% confidence
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* Trust Score Cards */}
            {trustScore && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <MetricCard
                        label="Data Quality"
                        value={trustScore.dataset_trust.data_quality}
                        color="accent"
                        sublabel="Feature integrity"
                    />
                    <MetricCard
                        label="Poison Risk"
                        value={trustScore.dataset_trust.poison_risk}
                        unit="%"
                        color={trustScore.dataset_trust.poison_risk > 50 ? 'danger' : trustScore.dataset_trust.poison_risk > 20 ? 'yellow' : 'accent3'}
                        sublabel="Contamination level"
                    />
                    <MetricCard
                        label="Behavioral Trust"
                        value={trustScore.dataset_trust.behavioral_trust}
                        color="accent3"
                        sublabel="Client consistency"
                    />
                    <div className="bg-surface border border-border rounded-lg p-5 relative overflow-hidden">
                        <div className="absolute top-0 left-0 w-0.5 h-full bg-purple" />
                        <div className="font-mono text-xs text-text3 uppercase tracking-widest mb-2">Model Grade</div>
                        <div className="text-5xl font-bold tracking-tight text-purple">
                            {trustScore.model_safety.grade}
                        </div>
                        <div className="font-mono text-xs text-text3 mt-1">
                            Backdoor: <span className={
                                trustScore.model_safety.backdoor_risk === 'HIGH' ? 'text-danger' :
                                    trustScore.model_safety.backdoor_risk === 'MEDIUM' ? 'text-yellow' : 'text-accent3'
                            }>{trustScore.model_safety.backdoor_risk}</span>
                        </div>
                    </div>
                </div>
            )}

            {/* 5-Layer Detection Scores */}
            {demoResult && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="bg-surface border border-border rounded-lg p-6">
                        <div className="font-mono text-xs text-accent uppercase tracking-widest mb-4">
                            5-Layer Detection Scores
                        </div>
                        <LayerScoreBar name="Layer 1 · Statistical Shift" score={layerScores.statistical || 0} />
                        <LayerScoreBar name="Layer 2 · Spectral Activation" score={layerScores.spectral || 0} />
                        <LayerScoreBar name="Layer 3 · Ensemble Anomaly" score={layerScores.ensemble || 0} />
                        <LayerScoreBar name="Layer 4 · Causal Proof" score={layerScores.causal || 0} />
                        <LayerScoreBar name="Layer 5 · Federated Trust" score={layerScores.federated || 0} />
                        <LayerScoreBar name="SHAP Drift Monitor" score={layerScores.shap_drift || 0} />
                    </div>

                    {/* Causal Proof Box */}
                    {demoResult.layer_results?.layer4_causal && (
                        <div className="bg-surface border border-border rounded-lg p-6">
                            <div className="font-mono text-xs text-accent3 uppercase tracking-widest mb-4">
                                Causal Proof Engine
                            </div>
                            <div className="space-y-3">
                                {[
                                    { label: 'Causal Effect (Δ Accuracy)', value: `${(demoResult.layer_results.layer4_causal.causal_effect * 100).toFixed(1)}%`, color: 'danger' },
                                    { label: 'Accuracy with Poison', value: `${(demoResult.layer_results.layer4_causal.accuracy_with_poison * 100).toFixed(1)}%`, color: 'text2' },
                                    { label: 'Accuracy without Poison', value: `${(demoResult.layer_results.layer4_causal.accuracy_without_poison * 100).toFixed(1)}%`, color: 'accent3' },
                                    { label: 'Placebo Test', value: demoResult.layer_results.layer4_causal.placebo_passed ? '✓ PASSED' : '✗ FAILED', color: demoResult.layer_results.layer4_causal.placebo_passed ? 'accent3' : 'danger' },
                                    { label: 'Statistically Significant', value: demoResult.layer_results.layer4_causal.statistically_significant ? 'YES' : 'NO', color: 'accent' },
                                    { label: 'Proof Valid', value: demoResult.layer_results.layer4_causal.proof_valid ? '✓ VERIFIED' : 'PENDING', color: demoResult.layer_results.layer4_causal.proof_valid ? 'accent3' : 'yellow' },
                                ].map(({ label, value, color }) => (
                                    <div key={label} className="flex justify-between items-center py-2 border-b border-border/50">
                                        <span className="font-mono text-xs text-text3">{label}</span>
                                        <span className={`font-mono text-xs font-bold text-${color}`}>{value}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Attack Timeline Chart */}
            {timeline.length > 0 && (
                <div className="bg-surface border border-border rounded-lg p-6">
                    <div className="font-mono text-xs text-accent uppercase tracking-widest mb-6">
                        Attack Timeline — Accuracy vs SHAP Drift vs Poison Count
                    </div>
                    <ResponsiveContainer width="100%" height={280}>
                        <ComposedChart data={timeline.slice(-24)} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1e3a52" />
                            <XAxis dataKey="timestamp" tick={{ fill: '#4a7a9b', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                                tickFormatter={v => v.slice(11, 16)} />
                            <YAxis yAxisId="left" tick={{ fill: '#4a7a9b', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                                domain={[0.7, 1.0]} />
                            <YAxis yAxisId="right" orientation="right" tick={{ fill: '#4a7a9b', fontSize: 10, fontFamily: 'JetBrains Mono' }} />
                            <Tooltip content={<CustomTooltip />} />
                            <Bar yAxisId="right" dataKey="poison_count" fill="rgba(255,77,106,0.3)"
                                stroke="#ff4d6a" name="Poison Count" />
                            <Line yAxisId="left" type="monotone" dataKey="accuracy" stroke="#00e5ff"
                                strokeWidth={2} dot={false} name="Accuracy" />
                            <Line yAxisId="right" type="monotone" dataKey="shap_drift" stroke="#ffd166"
                                strokeWidth={1.5} dot={false} strokeDasharray="4 2" name="SHAP Drift" />
                            <Line yAxisId="right" type="monotone" dataKey="trust_score" stroke="#bd93f9"
                                strokeWidth={1.5} dot={false} name="Trust Score" />
                        </ComposedChart>
                    </ResponsiveContainer>
                    <div className="flex gap-6 mt-4 font-mono text-xs text-text3">
                        <span className="flex items-center gap-2"><span className="w-4 h-0.5 bg-accent inline-block" /> Accuracy</span>
                        <span className="flex items-center gap-2"><span className="w-4 h-0.5 bg-yellow inline-block" /> SHAP Drift</span>
                        <span className="flex items-center gap-2"><span className="w-4 h-0.5 bg-purple inline-block" /> Trust Score</span>
                        <span className="flex items-center gap-2"><span className="w-4 h-2 bg-danger/40 inline-block" /> Poison Count</span>
                    </div>
                </div>
            )}

            {/* Blast Radius */}
            {demoResult?.blast_radius && (
                <div className="bg-surface border border-border rounded-lg p-6">
                    <div className="font-mono text-xs text-danger uppercase tracking-widest mb-4">
                        Blast Radius Analysis
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        {[
                            { label: 'Poisoned Samples', value: demoResult.blast_radius.n_poisoned_samples, color: 'danger' },
                            { label: 'Batches Affected', value: demoResult.blast_radius.n_batches_affected, color: 'orange' },
                            { label: 'Models Impacted', value: demoResult.blast_radius.n_models_affected, color: 'yellow' },
                            { label: 'Prediction Impact', value: `${demoResult.blast_radius.prediction_impact_pct}%`, color: 'purple' },
                        ].map(({ label, value, color }) => (
                            <div key={label} className={`bg-bg3 border border-${color}/20 rounded-lg p-4 text-center`}>
                                <div className={`text-2xl font-bold text-${color}`}>{value}</div>
                                <div className="font-mono text-xs text-text3 mt-1">{label}</div>
                            </div>
                        ))}
                    </div>
                    {demoResult.blast_radius.downstream_harm && (
                        <div className="mt-4 p-4 bg-bg3 rounded-lg border border-danger/10">
                            <div className="font-mono text-xs text-text3 mb-2">Downstream Harm Estimate (Medical Domain)</div>
                            <div className="grid grid-cols-3 gap-4 font-mono text-xs">
                                <div>
                                    <span className="text-text3">Est. Misdiagnoses: </span>
                                    <span className="text-danger font-bold">{demoResult.blast_radius.downstream_harm.estimated_misdiagnoses}</span>
                                </div>
                                <div>
                                    <span className="text-text3">Accuracy Loss: </span>
                                    <span className="text-orange font-bold">{demoResult.blast_radius.downstream_harm.accuracy_loss_pct}%</span>
                                </div>
                                <div>
                                    <span className="text-text3">Financial Impact: </span>
                                    <span className="text-yellow font-bold">${demoResult.blast_radius.downstream_harm.financial_impact_usd?.toLocaleString()}</span>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Live WS Events */}
            {wsEvents.length > 0 && (
                <div className="bg-surface border border-border rounded-lg p-6">
                    <div className="font-mono text-xs text-accent uppercase tracking-widest mb-4">
                        Live Detection Events
                    </div>
                    <div className="space-y-2">
                        {wsEvents.slice(0, 6).map((evt, i) => (
                            <div key={i} className={`flex items-start gap-3 p-3 rounded-md border-l-2 font-mono text-xs ${evt.event === 'attack_confirmed' ? 'border-danger bg-danger/5 text-danger' :
                                    evt.event === 'defense_triggered' ? 'border-purple bg-purple/5 text-purple' :
                                        evt.event === 'human_review_required' ? 'border-yellow bg-yellow/5 text-yellow' :
                                            'border-accent bg-accent/5 text-accent'
                                }`}>
                                <span className="font-bold uppercase tracking-widest flex-shrink-0">
                                    {evt.event?.replace(/_/g, ' ')}
                                </span>
                                <span className="text-text3">
                                    {evt.event === 'sample_analyzed' && `${evt.data?.n_samples} samples · score: ${(evt.data?.suspicion_score * 100).toFixed(1)}%`}
                                    {evt.event === 'attack_confirmed' && `${evt.data?.attack_type} · causal effect: ${(evt.data?.causal_effect * 100).toFixed(1)}%`}
                                    {evt.event === 'defense_triggered' && `${evt.data?.action} · ${evt.data?.samples_affected} samples`}
                                    {evt.event === 'human_review_required' && `score: ${(evt.data?.suspicion_score * 100).toFixed(1)}%`}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
