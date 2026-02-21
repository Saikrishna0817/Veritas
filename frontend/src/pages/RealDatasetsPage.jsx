import { useState, useEffect } from 'react';
import { api } from '../services/api';

const DATASET_META = {
    iris: { icon: '🌸', domain: 'Biology', color: '#a855f7' },
    wine: { icon: '🍷', domain: 'Chemistry', color: '#ef4444' },
    breast_cancer: { icon: '🏥', domain: 'Healthcare', color: '#06b6d4' },
    digits: { icon: '🔢', domain: 'Computer Vision', color: '#f59e0b' },
    diabetes: { icon: '💉', domain: 'Healthcare', color: '#10b981' },
    wine_quality: { icon: '🍇', domain: 'Food Science', color: '#ec4899' },
    covertype: { icon: '🌲', domain: 'Environmental', color: '#84cc16' },
};

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

const LAYER_LABELS = {
    statistical: 'L1 Statistical', spectral: 'L2 Spectral',
    ensemble: 'L3 Ensemble', causal: 'L4 Causal',
    federated: 'L5 Federated', shap_drift: 'SHAP Drift',
};

function ScoreBar({ label, score, color = '#6366f1' }) {
    const pct = Math.round((score || 0) * 100);
    return (
        <div style={{ marginBottom: 8 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3, fontSize: 11, color: '#94a3b8' }}>
                <span>{label}</span><span style={{ color, fontWeight: 700 }}>{pct}%</span>
            </div>
            <div style={{ height: 5, background: 'rgba(255,255,255,0.06)', borderRadius: 3, overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${pct}%`, background: `linear-gradient(90deg, ${color}88, ${color})`, borderRadius: 3, transition: 'width 1s ease' }} />
            </div>
        </div>
    );
}

function DatasetCard({ dataset, onAnalyze, onDownload, loading }) {
    const meta = DATASET_META[dataset.id] || { icon: '📊', domain: 'General', color: '#6366f1' };
    return (
        <div style={{ background: 'rgba(255,255,255,0.03)', border: `1px solid ${meta.color}22`, borderRadius: 14, padding: 20, display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <span style={{ fontSize: 32 }}>{meta.icon}</span>
                <div style={{ flex: 1 }}>
                    <div style={{ fontWeight: 700, color: '#f1f5f9', fontSize: 14 }}>{dataset.name}</div>
                    <div style={{ fontSize: 11, color: meta.color, marginTop: 2 }}>{meta.domain}</div>
                </div>
            </div>
            <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.6 }}>{dataset.description}</div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                {dataset.attack_types?.map(t => (
                    <span key={t} style={{ fontSize: 10, padding: '2px 7px', borderRadius: 20, background: `${ATTACK_COLORS[t] || '#6366f1'}18`, color: ATTACK_COLORS[t] || '#6366f1', border: `1px solid ${ATTACK_COLORS[t] || '#6366f1'}33` }}>
                        {t.replace(/_/g, ' ')}
                    </span>
                ))}
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
                <button onClick={() => onAnalyze(dataset.id)} disabled={loading === dataset.id}
                    style={{ flex: 1, padding: '9px 0', borderRadius: 8, border: 'none', background: loading === dataset.id ? 'rgba(99,102,241,0.3)' : `linear-gradient(135deg, ${meta.color}cc, ${meta.color})`, color: '#fff', fontWeight: 700, fontSize: 13, cursor: loading === dataset.id ? 'not-allowed' : 'pointer' }}>
                    {loading === dataset.id ? '🔬 Analyzing...' : '🚀 Analyze'}
                </button>
                <button onClick={() => onDownload(dataset.id)}
                    style={{ padding: '9px 14px', borderRadius: 8, border: `1px solid ${meta.color}33`, background: `${meta.color}10`, color: meta.color, fontWeight: 600, fontSize: 13, cursor: 'pointer' }}>
                    ⬇️ CSV
                </button>
            </div>
        </div>
    );
}

function ResultCard({ result, onClose }) {
    const verdict = VERDICT_CONFIG[result.verdict] || VERDICT_CONFIG.SUSPICIOUS;
    const attackColor = ATTACK_COLORS[result.attack_classification?.attack_type] || '#6366f1';
    return (
        <div style={{ background: 'rgba(255,255,255,0.03)', border: `1px solid ${verdict.color}33`, borderRadius: 16, padding: 24, animation: 'fadeIn 0.5s ease' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 20 }}>
                <span style={{ fontSize: 40 }}>{verdict.icon}</span>
                <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 18, fontWeight: 900, color: verdict.color }}>{verdict.label}</div>
                    <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 3 }}>
                        {result.real_dataset_description} · <span style={{ color: '#f59e0b' }}>{result.poison_note}</span>
                    </div>
                </div>
                <button onClick={onClose} style={{ padding: '6px 12px', borderRadius: 8, border: '1px solid rgba(255,255,255,0.1)', background: 'rgba(255,255,255,0.05)', color: '#94a3b8', cursor: 'pointer', fontSize: 12 }}>✕ Close</button>
            </div>

            <div style={{ display: 'flex', gap: 16, marginBottom: 16, flexWrap: 'wrap' }}>
                <div style={{ flex: 1, minWidth: 200 }}>
                    <div style={{ fontSize: 11, color: '#475569', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>Layer Scores</div>
                    {Object.entries(result.layer_scores || {}).map(([k, v]) => (
                        <ScoreBar key={k} label={LAYER_LABELS[k] || k} score={v}
                            color={Math.round(v * 100) > 70 ? '#ef4444' : Math.round(v * 100) > 40 ? '#f59e0b' : '#22c55e'} />
                    ))}
                </div>
                <div style={{ flex: 1, minWidth: 200 }}>
                    <div style={{ fontSize: 11, color: '#475569', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>Attack Probabilities</div>
                    {Object.entries(result.attack_classification?.probabilities || {}).sort(([, a], [, b]) => b - a).map(([type, prob]) => (
                        <ScoreBar key={type} label={type.replace(/_/g, ' ')} score={prob} color={ATTACK_COLORS[type] || '#6366f1'} />
                    ))}
                </div>
            </div>

            {result.injection_pattern?.narrative && (
                <pre style={{ fontFamily: 'monospace', fontSize: 11, color: '#94a3b8', lineHeight: 1.7, whiteSpace: 'pre-wrap', margin: 0, background: 'rgba(0,0,0,0.2)', padding: 14, borderRadius: 8 }}>
                    {result.injection_pattern.narrative}
                </pre>
            )}
        </div>
    );
}

export default function RealDatasetsPage() {
    const [catalog, setCatalog] = useState([]);
    const [loadingDataset, setLoadingDataset] = useState(null);
    const [results, setResults] = useState({});
    const [error, setError] = useState(null);

    useEffect(() => {
        api.getRealDatasets().then(d => setCatalog(d.datasets || [])).catch(() => { });
    }, []);

    const handleAnalyze = async (name) => {
        setLoadingDataset(name); setError(null);
        try {
            const data = await api.analyzeRealDataset(name);
            setResults(r => ({ ...r, [name]: data }));
        } catch (e) {
            setError(`Failed to analyze ${name}: ${e.message}`);
        } finally { setLoadingDataset(null); }
    };

    const handleDownload = async (name) => {
        try {
            const url = `${api.BASE_URL}/datasets/real/${name}/download`;
            const a = document.createElement('a');
            a.href = url; a.download = `${name}_poisoned.csv`; a.click();
        } catch (e) { setError(`Download failed: ${e.message}`); }
    };

    return (
        <div style={{ padding: '32px 40px', maxWidth: 1100, margin: '0 auto' }}>
            <div style={{ marginBottom: 28 }}>
                <h1 style={{ fontSize: 26, fontWeight: 800, color: '#f1f5f9', margin: 0 }}>📚 Real World Dataset Library</h1>
                <p style={{ color: '#64748b', marginTop: 8, fontSize: 14 }}>
                    {catalog.length} real public datasets (UCI, Wisconsin, MNIST, and more) with controlled poison injection — known ground truth,
                    real features. Click <strong style={{ color: '#f1f5f9' }}>Analyze</strong> to run the full 5-layer detection pipeline,
                    or <strong style={{ color: '#f1f5f9' }}>CSV</strong> to download and upload your own way.
                </p>
            </div>

            {error && (
                <div style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: 10, padding: '12px 16px', color: '#fca5a5', marginBottom: 20, fontSize: 14 }}>
                    ⚠️ {error}
                </div>
            )}

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 16, marginBottom: 28 }}>
                {catalog.map(ds => (
                    <DatasetCard key={ds.id} dataset={ds} onAnalyze={handleAnalyze} onDownload={handleDownload} loading={loadingDataset} />
                ))}
                {catalog.length === 0 && (
                    <div style={{ color: '#475569', fontSize: 14, padding: 20 }}>Loading dataset catalog...</div>
                )}
            </div>

            {/* Results */}
            {Object.entries(results).map(([name, result]) => (
                <div key={name} style={{ marginBottom: 20 }}>
                    <div style={{ fontSize: 12, color: '#475569', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>
                        {DATASET_META[name]?.icon} {name.replace(/_/g, ' ')} — Analysis Result
                    </div>
                    <ResultCard result={result} onClose={() => setResults(r => { const n = { ...r }; delete n[name]; return n; })} />
                </div>
            ))}

            <style>{`@keyframes fadeIn { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }`}</style>
        </div>
    );
}
