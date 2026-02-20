import { useState, useRef, useCallback } from 'react';
import { api } from '../services/api';

const ATTACK_COLORS = {
    label_flip: '#f59e0b',
    backdoor: '#ef4444',
    clean_label: '#a855f7',
    gradient_poisoning: '#06b6d4',
    boiling_frog: '#22c55e',
};

const VERDICT_CONFIG = {
    CONFIRMED_POISONED: { color: '#ef4444', bg: 'rgba(239,68,68,0.12)', icon: '☠️', label: 'CONFIRMED POISONED' },
    SUSPICIOUS: { color: '#f59e0b', bg: 'rgba(245,158,11,0.12)', icon: '⚠️', label: 'SUSPICIOUS' },
    LOW_RISK: { color: '#3b82f6', bg: 'rgba(59,130,246,0.10)', icon: '🔵', label: 'LOW RISK' },
    CLEAN: { color: '#22c55e', bg: 'rgba(34,197,94,0.12)', icon: '✅', label: 'CLEAN' },
};

const LAYER_LABELS = {
    statistical: 'L1 Statistical Shift',
    spectral: 'L2 Spectral Analysis',
    ensemble: 'L3 Ensemble Anomaly',
    causal: 'L4 Causal Proof',
    federated: 'L5 Federated Trust',
    shap_drift: 'SHAP Drift',
};

function ScoreBar({ label, score, color = '#6366f1' }) {
    const pct = Math.round((score || 0) * 100);
    return (
        <div style={{ marginBottom: 10 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4, fontSize: 12, color: '#94a3b8' }}>
                <span>{label}</span>
                <span style={{ color, fontWeight: 700 }}>{pct}%</span>
            </div>
            <div style={{ height: 6, background: 'rgba(255,255,255,0.06)', borderRadius: 4, overflow: 'hidden' }}>
                <div style={{
                    height: '100%', width: `${pct}%`, background: `linear-gradient(90deg, ${color}88, ${color})`,
                    borderRadius: 4, transition: 'width 1s ease'
                }} />
            </div>
        </div>
    );
}

function StatCard({ label, value, sub, color = '#6366f1', icon }) {
    return (
        <div style={{
            background: 'rgba(255,255,255,0.04)', border: `1px solid ${color}33`,
            borderRadius: 12, padding: '16px 20px', flex: 1, minWidth: 140
        }}>
            <div style={{ fontSize: 22, marginBottom: 4 }}>{icon}</div>
            <div style={{ fontSize: 22, fontWeight: 800, color, fontFamily: 'monospace' }}>{value}</div>
            <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{label}</div>
            {sub && <div style={{ fontSize: 10, color: '#475569', marginTop: 2 }}>{sub}</div>}
        </div>
    );
}

export default function UploadPage() {
    const [dragging, setDragging] = useState(false);
    const [file, setFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [progress, setProgress] = useState(0);
    const fileRef = useRef();

    const handleFile = (f) => {
        if (!f) return;
        if (!f.name.toLowerCase().endsWith('.csv')) {
            setError('Only CSV files are accepted.');
            return;
        }
        if (f.size > 10 * 1024 * 1024) {
            setError('File too large. Maximum size is 10MB.');
            return;
        }
        setFile(f);
        setError(null);
        setResult(null);
    };

    const onDrop = useCallback((e) => {
        e.preventDefault();
        setDragging(false);
        const f = e.dataTransfer.files[0];
        handleFile(f);
    }, []);

    const onDragOver = (e) => { e.preventDefault(); setDragging(true); };
    const onDragLeave = () => setDragging(false);

    const runAnalysis = async () => {
        if (!file) return;
        setLoading(true);
        setError(null);
        setProgress(0);

        // Fake progress animation
        const interval = setInterval(() => {
            setProgress(p => p < 85 ? p + Math.random() * 12 : p);
        }, 300);

        try {
            const data = await api.uploadCSV(file);
            clearInterval(interval);
            setProgress(100);
            setTimeout(() => setResult(data), 300);
        } catch (e) {
            clearInterval(interval);
            setError(e.message || 'Analysis failed.');
        } finally {
            setLoading(false);
        }
    };

    const verdict = result ? (VERDICT_CONFIG[result.verdict] || VERDICT_CONFIG.SUSPICIOUS) : null;
    const attackColor = result ? (ATTACK_COLORS[result.attack_classification?.attack_type] || '#6366f1') : '#6366f1';

    return (
        <div style={{ padding: '32px 40px', maxWidth: 1100, margin: '0 auto' }}>
            {/* Header */}
            <div style={{ marginBottom: 32 }}>
                <h1 style={{ fontSize: 28, fontWeight: 800, color: '#f1f5f9', margin: 0 }}>
                    📂 Upload Dataset for Analysis
                </h1>
                <p style={{ color: '#64748b', marginTop: 8, fontSize: 14 }}>
                    Upload any CSV file — the platform auto-detects schema, splits 70/30 for baseline, and runs all 5 detection layers.
                    Supports up to 50,000 rows · 10MB · supervised &amp; unsupervised modes.
                </p>
            </div>

            {/* Upload Zone */}
            <div
                onDrop={onDrop}
                onDragOver={onDragOver}
                onDragLeave={onDragLeave}
                onClick={() => fileRef.current?.click()}
                style={{
                    border: `2px dashed ${dragging ? '#6366f1' : file ? '#22c55e' : 'rgba(255,255,255,0.12)'}`,
                    borderRadius: 16, padding: '48px 32px', textAlign: 'center', cursor: 'pointer',
                    background: dragging ? 'rgba(99,102,241,0.08)' : file ? 'rgba(34,197,94,0.05)' : 'rgba(255,255,255,0.02)',
                    transition: 'all 0.3s ease', marginBottom: 24,
                }}
            >
                <input ref={fileRef} type="file" accept=".csv" style={{ display: 'none' }}
                    onChange={e => handleFile(e.target.files[0])} />
                <div style={{ fontSize: 48, marginBottom: 12 }}>
                    {file ? '📄' : dragging ? '📥' : '☁️'}
                </div>
                {file ? (
                    <>
                        <div style={{ color: '#22c55e', fontWeight: 700, fontSize: 16 }}>{file.name}</div>
                        <div style={{ color: '#64748b', fontSize: 13, marginTop: 4 }}>
                            {(file.size / 1024).toFixed(1)} KB · Click to change
                        </div>
                    </>
                ) : (
                    <>
                        <div style={{ color: '#94a3b8', fontWeight: 600, fontSize: 16 }}>
                            Drop your CSV here or click to browse
                        </div>
                        <div style={{ color: '#475569', fontSize: 13, marginTop: 6 }}>
                            Accepts .csv files up to 10MB (≤ 50,000 rows)
                        </div>
                    </>
                )}
            </div>

            {/* Error */}
            {error && (
                <div style={{
                    background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
                    borderRadius: 10, padding: '12px 16px', color: '#fca5a5', marginBottom: 20, fontSize: 14
                }}>
                    ⚠️ {error}
                </div>
            )}

            {/* Analyze Button */}
            {file && !result && (
                <button
                    onClick={runAnalysis}
                    disabled={loading}
                    style={{
                        width: '100%', padding: '14px 0', borderRadius: 12, border: 'none',
                        background: loading ? 'rgba(99,102,241,0.4)' : 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                        color: '#fff', fontWeight: 700, fontSize: 16, cursor: loading ? 'not-allowed' : 'pointer',
                        transition: 'all 0.3s', marginBottom: 24,
                        boxShadow: loading ? 'none' : '0 4px 24px rgba(99,102,241,0.4)',
                    }}
                >
                    {loading ? '🔬 Analyzing...' : '🚀 Run Poisoning Detection'}
                </button>
            )}

            {/* Progress Bar */}
            {loading && (
                <div style={{ marginBottom: 24 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#64748b', marginBottom: 6 }}>
                        <span>Running 5-layer detection pipeline...</span>
                        <span>{Math.round(progress)}%</span>
                    </div>
                    <div style={{ height: 8, background: 'rgba(255,255,255,0.06)', borderRadius: 4, overflow: 'hidden' }}>
                        <div style={{
                            height: '100%', width: `${progress}%`,
                            background: 'linear-gradient(90deg, #6366f1, #8b5cf6, #a855f7)',
                            borderRadius: 4, transition: 'width 0.4s ease',
                            boxShadow: '0 0 12px rgba(99,102,241,0.6)'
                        }} />
                    </div>
                    <div style={{ display: 'flex', gap: 8, marginTop: 10, flexWrap: 'wrap' }}>
                        {['Schema Detection', 'L1 Statistical', 'L2 Spectral', 'L3 Ensemble', 'L4 Causal', 'L5 Federated'].map((s, i) => (
                            <span key={s} style={{
                                fontSize: 11, padding: '3px 8px', borderRadius: 20,
                                background: progress > i * 14 ? 'rgba(99,102,241,0.3)' : 'rgba(255,255,255,0.04)',
                                color: progress > i * 14 ? '#a5b4fc' : '#475569',
                                border: `1px solid ${progress > i * 14 ? 'rgba(99,102,241,0.4)' : 'rgba(255,255,255,0.06)'}`,
                                transition: 'all 0.5s'
                            }}>{progress > i * 14 ? '✓ ' : ''}{s}</span>
                        ))}
                    </div>
                </div>
            )}

            {/* Results */}
            {result && (
                <div style={{ animation: 'fadeIn 0.5s ease' }}>
                    {/* Verdict Banner */}
                    <div style={{
                        background: verdict.bg, border: `1px solid ${verdict.color}44`,
                        borderRadius: 16, padding: '20px 28px', marginBottom: 24,
                        display: 'flex', alignItems: 'center', gap: 20
                    }}>
                        <div style={{ fontSize: 48 }}>{verdict.icon}</div>
                        <div style={{ flex: 1 }}>
                            <div style={{ fontSize: 22, fontWeight: 900, color: verdict.color, letterSpacing: 2 }}>
                                {verdict.label}
                            </div>
                            <div style={{ color: '#94a3b8', fontSize: 14, marginTop: 4 }}>
                                Suspicion Score: <strong style={{ color: verdict.color }}>
                                    {Math.round((result.overall_suspicion_score || 0) * 100)}%
                                </strong>
                                &nbsp;·&nbsp;
                                Poisoning Level: <strong style={{ color: verdict.color }}>
                                    {result.poisoning_level || 'N/A'}
                                </strong>
                                &nbsp;·&nbsp;
                                Mode: <strong style={{ color: '#94a3b8' }}>
                                    {result.detection_mode === 'supervised' ? '🏷️ Supervised' : '🔍 Unsupervised'}
                                </strong>
                            </div>
                        </div>
                        <button
                            onClick={() => { setResult(null); setFile(null); setProgress(0); }}
                            style={{
                                padding: '8px 16px', borderRadius: 8, border: '1px solid rgba(255,255,255,0.1)',
                                background: 'rgba(255,255,255,0.05)', color: '#94a3b8', cursor: 'pointer', fontSize: 13
                            }}
                        >
                            ↩ New Upload
                        </button>
                    </div>

                    {/* Dataset Info + Attack Classification */}
                    <div style={{ display: 'flex', gap: 20, marginBottom: 24, flexWrap: 'wrap' }}>
                        {/* Dataset Info */}
                        <div style={{
                            flex: 1, minWidth: 280, background: 'rgba(255,255,255,0.03)',
                            border: '1px solid rgba(255,255,255,0.08)', borderRadius: 14, padding: 20
                        }}>
                            <div style={{ fontWeight: 700, color: '#f1f5f9', marginBottom: 14, fontSize: 14 }}>
                                📊 Dataset Schema
                            </div>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                                {[
                                    { k: 'Filename', v: result.dataset_info?.filename || '—' },
                                    { k: 'Total Rows', v: result.dataset_info?.n_rows?.toLocaleString() || '—' },
                                    { k: 'Features', v: result.dataset_info?.n_features || '—' },
                                    { k: 'Label Column', v: result.dataset_info?.label_column || 'None (unsupervised)' },
                                    { k: 'Reference Split', v: `${result.dataset_info?.reference_split || 0} rows (70%)` },
                                    { k: 'Analyzed', v: `${result.n_samples || 0} rows (30%)` },
                                ].map(({ k, v }) => (
                                    <div key={k}>
                                        <div style={{ fontSize: 10, color: '#475569', textTransform: 'uppercase', letterSpacing: 1 }}>{k}</div>
                                        <div style={{ fontSize: 13, color: '#cbd5e1', fontWeight: 600, marginTop: 2, wordBreak: 'break-all' }}>{v}</div>
                                    </div>
                                ))}
                            </div>
                            {result.dataset_info?.feature_names?.length > 0 && (
                                <div style={{ marginTop: 14 }}>
                                    <div style={{ fontSize: 10, color: '#475569', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 6 }}>
                                        Feature Columns
                                    </div>
                                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                                        {result.dataset_info.feature_names.map(f => (
                                            <span key={f} style={{
                                                fontSize: 11, padding: '3px 8px', borderRadius: 20,
                                                background: 'rgba(99,102,241,0.15)', color: '#a5b4fc',
                                                border: '1px solid rgba(99,102,241,0.25)'
                                            }}>{f}</span>
                                        ))}
                                    </div>
                                </div>
                            )}
                            {result.dataset_info?.warnings?.length > 0 && (
                                <div style={{ marginTop: 12, padding: '8px 12px', background: 'rgba(245,158,11,0.08)', borderRadius: 8, border: '1px solid rgba(245,158,11,0.2)' }}>
                                    {result.dataset_info.warnings.map((w, i) => (
                                        <div key={i} style={{ fontSize: 12, color: '#fbbf24' }}>⚠️ {w}</div>
                                    ))}
                                </div>
                            )}
                        </div>

                        {/* Attack Classification */}
                        <div style={{
                            flex: 1, minWidth: 280, background: 'rgba(255,255,255,0.03)',
                            border: `1px solid ${attackColor}33`, borderRadius: 14, padding: 20
                        }}>
                            <div style={{ fontWeight: 700, color: '#f1f5f9', marginBottom: 14, fontSize: 14 }}>
                                🎯 Attack Classification
                            </div>
                            <div style={{ textAlign: 'center', marginBottom: 16 }}>
                                <div style={{ fontSize: 36, marginBottom: 8 }}>
                                    {result.attack_classification?.attack_type === 'backdoor' ? '🚪' :
                                        result.attack_classification?.attack_type === 'label_flip' ? '🔄' :
                                            result.attack_classification?.attack_type === 'clean_label' ? '🎭' :
                                                result.attack_classification?.attack_type === 'gradient_poisoning' ? '⚡' : '🐸'}
                                </div>
                                <div style={{ fontSize: 20, fontWeight: 800, color: attackColor, textTransform: 'capitalize' }}>
                                    {(result.attack_classification?.attack_type || 'unknown').replace(/_/g, ' ')}
                                </div>
                                <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>
                                    Subtype: {result.attack_classification?.attack_subtype?.replace(/_/g, ' ') || '—'}
                                </div>
                            </div>
                            <div style={{ marginBottom: 12 }}>
                                <ScoreBar
                                    label="Classification Confidence"
                                    score={result.attack_classification?.confidence || 0}
                                    color={attackColor}
                                />
                            </div>
                            <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.6 }}>
                                {result.attack_classification?.description}
                            </div>
                            {/* Probability breakdown */}
                            {result.attack_classification?.probabilities && (
                                <div style={{ marginTop: 14 }}>
                                    <div style={{ fontSize: 10, color: '#475569', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>
                                        Attack Probabilities
                                    </div>
                                    {Object.entries(result.attack_classification.probabilities)
                                        .sort(([, a], [, b]) => b - a)
                                        .map(([type, prob]) => (
                                            <ScoreBar
                                                key={type}
                                                label={type.replace(/_/g, ' ')}
                                                score={prob}
                                                color={ATTACK_COLORS[type] || '#6366f1'}
                                            />
                                        ))
                                    }
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Layer Scores */}
                    <div style={{
                        background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)',
                        borderRadius: 14, padding: 20, marginBottom: 24
                    }}>
                        <div style={{ fontWeight: 700, color: '#f1f5f9', marginBottom: 16, fontSize: 14 }}>
                            🔬 5-Layer Detection Scores
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
                            {Object.entries(result.layer_scores || {}).map(([layer, score]) => {
                                const pct = Math.round((score || 0) * 100);
                                const color = pct > 70 ? '#ef4444' : pct > 40 ? '#f59e0b' : '#22c55e';
                                return (
                                    <div key={layer} style={{
                                        background: 'rgba(255,255,255,0.03)', borderRadius: 10, padding: '14px 16px',
                                        border: `1px solid ${color}22`
                                    }}>
                                        <div style={{ fontSize: 11, color: '#64748b', marginBottom: 8 }}>
                                            {LAYER_LABELS[layer] || layer}
                                        </div>
                                        <div style={{ fontSize: 24, fontWeight: 800, color, fontFamily: 'monospace' }}>
                                            {pct}%
                                        </div>
                                        <div style={{ height: 4, background: 'rgba(255,255,255,0.06)', borderRadius: 2, marginTop: 8, overflow: 'hidden' }}>
                                            <div style={{ height: '100%', width: `${pct}%`, background: color, borderRadius: 2, transition: 'width 1s ease' }} />
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    {/* Stats Row */}
                    <div style={{ display: 'flex', gap: 16, marginBottom: 24, flexWrap: 'wrap' }}>
                        <StatCard
                            icon="🎯"
                            label="Sophistication Score"
                            value={`${result.sophistication?.sophistication_score || 0}/10`}
                            sub={result.sophistication?.level}
                            color="#a855f7"
                        />
                        <StatCard
                            icon="💥"
                            label="Batches Affected"
                            value={result.blast_radius?.n_batches_affected || 0}
                            sub={`${result.blast_radius?.n_models_affected || 0} models`}
                            color="#ef4444"
                        />
                        <StatCard
                            icon="📉"
                            label="Prediction Impact"
                            value={`${result.blast_radius?.prediction_impact_pct || 0}%`}
                            sub="accuracy degradation"
                            color="#f59e0b"
                        />
                        <StatCard
                            icon="🛡️"
                            label="Defense Action"
                            value={result.defense_action?.action?.replace(/_/g, ' ') || 'monitor'}
                            sub={result.defense_action?.reason?.replace(/_/g, ' ')}
                            color="#22c55e"
                        />
                    </div>

                    {/* Narrative */}
                    {result.injection_pattern?.narrative && (
                        <div style={{
                            background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)',
                            borderRadius: 14, padding: 20, marginBottom: 24
                        }}>
                            <div style={{ fontWeight: 700, color: '#f1f5f9', marginBottom: 12, fontSize: 14 }}>
                                📋 Attack Reconstruction Narrative
                            </div>
                            <pre style={{
                                fontFamily: 'monospace', fontSize: 12, color: '#94a3b8', lineHeight: 1.8,
                                whiteSpace: 'pre-wrap', margin: 0,
                                background: 'rgba(0,0,0,0.2)', padding: 16, borderRadius: 8
                            }}>
                                {result.injection_pattern.narrative}
                            </pre>
                        </div>
                    )}

                    {/* Download Report */}
                    <button
                        onClick={() => {
                            const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = `forensics_${result.dataset_info?.filename?.replace('.csv', '') || 'report'}_${Date.now()}.json`;
                            a.click();
                            URL.revokeObjectURL(url);
                        }}
                        style={{
                            width: '100%', padding: '12px 0', borderRadius: 12, border: '1px solid rgba(99,102,241,0.3)',
                            background: 'rgba(99,102,241,0.1)', color: '#a5b4fc', fontWeight: 700, fontSize: 14,
                            cursor: 'pointer', transition: 'all 0.3s'
                        }}
                    >
                        ⬇️ Download Forensic Report (JSON)
                    </button>
                </div>
            )}

            <style>{`
        @keyframes fadeIn { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
      `}</style>
        </div>
    );
}
