import { useState, useRef, useCallback } from 'react';
import { api } from '../services/api';

const VERDICT_CONFIG = {
    CONFIRMED_POISONED: { color: '#ef4444', bg: 'rgba(239,68,68,0.12)', icon: '☠️', label: 'CONFIRMED POISONED' },
    SUSPICIOUS: { color: '#f59e0b', bg: 'rgba(245,158,11,0.12)', icon: '⚠️', label: 'SUSPICIOUS' },
    LOW_RISK: { color: '#3b82f6', bg: 'rgba(59,130,246,0.10)', icon: '🔵', label: 'LOW RISK' },
    CLEAN: { color: '#22c55e', bg: 'rgba(34,197,94,0.12)', icon: '✅', label: 'CLEAN' },
};

const ATTACK_COLORS = {
    label_flip: '#f59e0b', backdoor: '#ef4444',
    clean_label: '#a855f7', gradient_poisoning: '#06b6d4', boiling_frog: '#22c55e',
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
                <div style={{ height: '100%', width: `${pct}%`, background: `linear-gradient(90deg, ${color}88, ${color})`, borderRadius: 4, transition: 'width 1s ease' }} />
            </div>
        </div>
    );
}

const LAYER_LABELS = {
    statistical: 'L1 Statistical Shift', spectral: 'L2 Spectral Analysis',
    ensemble: 'L3 Ensemble Anomaly', causal: 'L4 Causal Proof',
    federated: 'L5 Federated Trust', shap_drift: 'SHAP Drift',
};

function ResultPanel({ result, onReset }) {
    const verdict = VERDICT_CONFIG[result.verdict] || VERDICT_CONFIG.SUSPICIOUS;
    const attackColor = ATTACK_COLORS[result.attack_classification?.attack_type] || '#6366f1';

    return (
        <div style={{ animation: 'fadeIn 0.5s ease' }}>
            {/* Verdict */}
            <div style={{ background: verdict.bg, border: `1px solid ${verdict.color}44`, borderRadius: 16, padding: '20px 28px', marginBottom: 20, display: 'flex', alignItems: 'center', gap: 20 }}>
                <div style={{ fontSize: 44 }}>{verdict.icon}</div>
                <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 20, fontWeight: 900, color: verdict.color, letterSpacing: 2 }}>{verdict.label}</div>
                    <div style={{ color: '#94a3b8', fontSize: 13, marginTop: 4 }}>
                        Suspicion: <strong style={{ color: verdict.color }}>{Math.round((result.overall_suspicion_score || 0) * 100)}%</strong>
                        &nbsp;·&nbsp;Attack: <strong style={{ color: attackColor, textTransform: 'capitalize' }}>
                            {(result.attack_classification?.attack_type || 'unknown').replace(/_/g, ' ')}
                        </strong>
                        &nbsp;·&nbsp;Confidence: <strong style={{ color: attackColor }}>
                            {Math.round((result.attack_classification?.confidence || 0) * 100)}%
                        </strong>
                    </div>
                </div>
                <button onClick={onReset} style={{ padding: '8px 14px', borderRadius: 8, border: '1px solid rgba(255,255,255,0.1)', background: 'rgba(255,255,255,0.05)', color: '#94a3b8', cursor: 'pointer', fontSize: 13 }}>
                    ↩ New Scan
                </button>
            </div>

            {/* Layer Scores */}
            <div style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 14, padding: 20, marginBottom: 20 }}>
                <div style={{ fontWeight: 700, color: '#f1f5f9', marginBottom: 14, fontSize: 14 }}>🔬 5-Layer Detection Scores</div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 10 }}>
                    {Object.entries(result.layer_scores || {}).map(([layer, score]) => {
                        const pct = Math.round((score || 0) * 100);
                        const color = pct > 70 ? '#ef4444' : pct > 40 ? '#f59e0b' : '#22c55e';
                        return (
                            <div key={layer} style={{ background: 'rgba(255,255,255,0.03)', borderRadius: 10, padding: '12px 14px', border: `1px solid ${color}22` }}>
                                <div style={{ fontSize: 10, color: '#64748b', marginBottom: 6 }}>{LAYER_LABELS[layer] || layer}</div>
                                <div style={{ fontSize: 22, fontWeight: 800, color, fontFamily: 'monospace' }}>{pct}%</div>
                                <div style={{ height: 3, background: 'rgba(255,255,255,0.06)', borderRadius: 2, marginTop: 6, overflow: 'hidden' }}>
                                    <div style={{ height: '100%', width: `${pct}%`, background: color, borderRadius: 2, transition: 'width 1s ease' }} />
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* Attack probs */}
            {result.attack_classification?.probabilities && (
                <div style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 14, padding: 20, marginBottom: 20 }}>
                    <div style={{ fontWeight: 700, color: '#f1f5f9', marginBottom: 14, fontSize: 14 }}>🎯 Attack Probability Breakdown</div>
                    {Object.entries(result.attack_classification.probabilities).sort(([, a], [, b]) => b - a).map(([type, prob]) => (
                        <ScoreBar key={type} label={type.replace(/_/g, ' ')} score={prob} color={ATTACK_COLORS[type] || '#6366f1'} />
                    ))}
                </div>
            )}

            {/* Narrative */}
            {result.injection_pattern?.narrative && (
                <div style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 14, padding: 20, marginBottom: 20 }}>
                    <div style={{ fontWeight: 700, color: '#f1f5f9', marginBottom: 10, fontSize: 14 }}>📋 Attack Narrative</div>
                    <pre style={{ fontFamily: 'monospace', fontSize: 12, color: '#94a3b8', lineHeight: 1.8, whiteSpace: 'pre-wrap', margin: 0, background: 'rgba(0,0,0,0.2)', padding: 14, borderRadius: 8 }}>
                        {result.injection_pattern.narrative}
                    </pre>
                </div>
            )}

            {/* Download */}
            <button onClick={() => {
                const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a'); a.href = url;
                a.download = `model_scan_${result.scan_id || Date.now()}.json`; a.click();
                URL.revokeObjectURL(url);
            }} style={{ width: '100%', padding: '12px 0', borderRadius: 12, border: '1px solid rgba(99,102,241,0.3)', background: 'rgba(99,102,241,0.1)', color: '#a5b4fc', fontWeight: 700, fontSize: 14, cursor: 'pointer' }}>
                ⬇️ Download Full Scan Report (JSON)
            </button>
        </div>
    );
}

export default function ModelScanPage() {
    const [modelFile, setModelFile] = useState(null);
    const [datasetFile, setDatasetFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [progress, setProgress] = useState(0);
    const [dragging, setDragging] = useState(false);
    const modelRef = useRef();
    const datasetRef = useRef();

    const handleModelFile = (f) => {
        if (!f) return;
        if (!f.name.toLowerCase().endsWith('.pkl')) { setError('Only .pkl (pickle) model files are accepted.'); return; }
        if (f.size > 50 * 1024 * 1024) { setError('Model file too large. Maximum is 50MB.'); return; }
        setModelFile(f); setError(null); setResult(null);
    };

    const handleDatasetFile = (f) => {
        if (!f) return;
        if (!f.name.toLowerCase().endsWith('.csv')) { setError('Dataset must be a .csv file.'); return; }
        setDatasetFile(f); setError(null);
    };

    const onDrop = useCallback((e) => {
        e.preventDefault(); setDragging(false);
        const files = Array.from(e.dataTransfer.files);
        files.forEach(f => {
            if (f.name.endsWith('.pkl')) handleModelFile(f);
            else if (f.name.endsWith('.csv')) handleDatasetFile(f);
        });
    }, []);

    const runScan = async () => {
        if (!modelFile) return;
        setLoading(true); setError(null); setProgress(0);
        const interval = setInterval(() => setProgress(p => p < 85 ? p + Math.random() * 10 : p), 400);
        try {
            const formData = new FormData();
            formData.append('model_file', modelFile);
            if (datasetFile) formData.append('dataset_file', datasetFile);
            const data = await api.scanModel(formData);
            clearInterval(interval); setProgress(100);
            setTimeout(() => setResult(data), 300);
        } catch (e) {
            clearInterval(interval);
            setError(e.message || 'Model scan failed.');
        } finally { setLoading(false); }
    };

    const reset = () => { setResult(null); setModelFile(null); setDatasetFile(null); setProgress(0); setError(null); };

    return (
        <div style={{ padding: '32px 40px', maxWidth: 1000, margin: '0 auto' }}>
            <div style={{ marginBottom: 28 }}>
                <h1 style={{ fontSize: 26, fontWeight: 800, color: '#f1f5f9', margin: 0 }}>🤖 Model Poisoning Scanner</h1>
                <p style={{ color: '#64748b', marginTop: 8, fontSize: 14 }}>
                    Upload a trained sklearn <code style={{ background: 'rgba(255,255,255,0.08)', padding: '1px 6px', borderRadius: 4 }}>.pkl</code> model
                    — we extract its learned parameters and run all 5 detection layers to check if it was trained on poisoned data.
                    Optionally attach the training CSV for deeper analysis.
                </p>
                <div style={{ display: 'flex', gap: 8, marginTop: 10, flexWrap: 'wrap' }}>
                    {['LogisticRegression', 'RandomForest', 'GradientBoosting', 'SVC', 'MLP', 'DecisionTree', 'SGD', 'KNeighbors', 'Ridge/Lasso', 'NaiveBayes'].map(m => (
                        <span key={m} style={{ fontSize: 11, padding: '2px 8px', borderRadius: 20, background: 'rgba(99,102,241,0.12)', color: '#a5b4fc', border: '1px solid rgba(99,102,241,0.2)' }}>{m}</span>
                    ))}
                </div>
            </div>

            {!result && (
                <>
                    {/* Drop zone */}
                    <div onDrop={onDrop} onDragOver={e => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)}
                        style={{ border: `2px dashed ${dragging ? '#6366f1' : modelFile ? '#22c55e' : 'rgba(255,255,255,0.12)'}`, borderRadius: 16, padding: '40px 28px', textAlign: 'center', cursor: 'pointer', background: dragging ? 'rgba(99,102,241,0.08)' : 'rgba(255,255,255,0.02)', transition: 'all 0.3s', marginBottom: 16 }}
                        onClick={() => modelRef.current?.click()}>
                        <input ref={modelRef} type="file" accept=".pkl" style={{ display: 'none' }} onChange={e => handleModelFile(e.target.files[0])} />
                        <div style={{ fontSize: 44, marginBottom: 10 }}>{modelFile ? '🤖' : dragging ? '📥' : '🧠'}</div>
                        {modelFile ? (
                            <><div style={{ color: '#22c55e', fontWeight: 700 }}>{modelFile.name}</div>
                                <div style={{ color: '#64748b', fontSize: 12, marginTop: 4 }}>{(modelFile.size / 1024).toFixed(1)} KB · Click to change</div></>
                        ) : (
                            <><div style={{ color: '#94a3b8', fontWeight: 600 }}>Drop your .pkl model here or click to browse</div>
                                <div style={{ color: '#475569', fontSize: 13, marginTop: 4 }}>Accepts sklearn .pkl files up to 50MB</div></>
                        )}
                    </div>

                    {/* Optional CSV */}
                    <div onClick={() => datasetRef.current?.click()} style={{ border: `1px dashed ${datasetFile ? '#22c55e' : 'rgba(255,255,255,0.08)'}`, borderRadius: 12, padding: '14px 20px', cursor: 'pointer', background: 'rgba(255,255,255,0.02)', display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20 }}>
                        <input ref={datasetRef} type="file" accept=".csv" style={{ display: 'none' }} onChange={e => handleDatasetFile(e.target.files[0])} />
                        <span style={{ fontSize: 22 }}>{datasetFile ? '📄' : '➕'}</span>
                        <div>
                            <div style={{ color: datasetFile ? '#22c55e' : '#64748b', fontSize: 13, fontWeight: 600 }}>
                                {datasetFile ? datasetFile.name : 'Attach training dataset (optional CSV)'}
                            </div>
                            <div style={{ color: '#475569', fontSize: 11 }}>Enables deeper dataset-level poisoning analysis alongside model scan</div>
                        </div>
                        {datasetFile && <button onClick={e => { e.stopPropagation(); setDatasetFile(null); }} style={{ marginLeft: 'auto', background: 'none', border: 'none', color: '#64748b', cursor: 'pointer', fontSize: 18 }}>✕</button>}
                    </div>

                    {error && <div style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: 10, padding: '12px 16px', color: '#fca5a5', marginBottom: 16, fontSize: 14 }}>⚠️ {error}</div>}

                    {modelFile && (
                        <button onClick={runScan} disabled={loading} style={{ width: '100%', padding: '14px 0', borderRadius: 12, border: 'none', background: loading ? 'rgba(99,102,241,0.4)' : 'linear-gradient(135deg, #6366f1, #8b5cf6)', color: '#fff', fontWeight: 700, fontSize: 16, cursor: loading ? 'not-allowed' : 'pointer', boxShadow: loading ? 'none' : '0 4px 24px rgba(99,102,241,0.4)', marginBottom: 20 }}>
                            {loading ? '🔬 Scanning Model Parameters...' : '🚀 Scan for Poisoning'}
                        </button>
                    )}

                    {loading && (
                        <div style={{ marginBottom: 20 }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#64748b', marginBottom: 6 }}>
                                <span>Extracting parameters → running 5-layer detection...</span>
                                <span>{Math.round(progress)}%</span>
                            </div>
                            <div style={{ height: 8, background: 'rgba(255,255,255,0.06)', borderRadius: 4, overflow: 'hidden' }}>
                                <div style={{ height: '100%', width: `${progress}%`, background: 'linear-gradient(90deg, #6366f1, #8b5cf6, #a855f7)', borderRadius: 4, transition: 'width 0.4s ease', boxShadow: '0 0 12px rgba(99,102,241,0.6)' }} />
                            </div>
                        </div>
                    )}
                </>
            )}

            {result && <ResultPanel result={result} onReset={reset} />}

            <style>{`@keyframes fadeIn { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }`}</style>
        </div>
    );
}
