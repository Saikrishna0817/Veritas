import { useState, useRef, useCallback } from 'react';
import { api } from '../services/api';
import { ShieldAlert, Activity, Check, Microscope, Target, ClipboardList, Download, Bot, BrainCircuit, FolderUp, FileText, Plus, X } from 'lucide-react';

const VERDICT_CONFIG = {
    CONFIRMED_POISONED: { color: 'var(--status-critical)', bg: 'rgba(228,36,43,0.1)', icon: <ShieldAlert className="w-12 h-12 text-[var(--status-critical)]" />, label: 'CONFIRMED POISONED' },
    SUSPICIOUS: { color: 'var(--status-warn)', bg: 'rgba(242,184,75,0.1)', icon: <Activity className="w-12 h-12 text-[var(--status-warn)]" />, label: 'SUSPICIOUS' },
    LOW_RISK: { color: '#3b82f6', bg: 'rgba(59,130,246,0.1)', icon: <Check className="w-12 h-12 text-blue-500" />, label: 'LOW RISK' },
    CLEAN: { color: 'var(--status-safe)', bg: 'rgba(61,220,132,0.1)', icon: <Check className="w-12 h-12 text-[var(--status-safe)]" />, label: 'CLEAN' },
};

const ATTACK_COLORS = {
    label_flip: '#f59e0b', backdoor: 'var(--red-primary)',
    clean_label: '#a855f7', gradient_poisoning: '#06b6d4', boiling_frog: 'var(--status-safe)',
};

function ScoreBar({ label, score, color = 'var(--red-primary)' }) {
    const pct = Math.round((score || 0) * 100);
    return (
        <div className="mb-3">
            <div className="flex justify-between mb-1.5 text-xs text-textMuted font-bold tracking-wider uppercase">
                <span>{label}</span>
                <span style={{ color }}>{pct}%</span>
            </div>
            <div className="h-2 bg-bgVoid rounded-full overflow-hidden">
                <div className="h-full rounded-full transition-all duration-1000 ease-out" style={{ width: `${pct}%`, background: `linear-gradient(90deg, ${color}88, ${color})` }} />
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
    const attackColor = ATTACK_COLORS[result.attack_classification?.attack_type] || 'var(--red-primary)';

    return (
        <div className="animate-fadeInUp space-y-6">
            {/* Verdict */}
            <div className="rounded-[24px] p-6 md:p-8 flex flex-col md:flex-row items-center gap-6 border" style={{ background: verdict.bg, borderColor: `${verdict.color}40` }}>
                <div>{verdict.icon}</div>
                <div className="flex-1 text-center md:text-left">
                    <div className="text-2xl md:text-3xl font-display font-bold tracking-wide" style={{ color: verdict.color }}>{verdict.label}</div>
                    <div className="text-textPrimary text-sm mt-3 font-medium flex items-center justify-center md:justify-start gap-3 flex-wrap">
                        <span className="text-textMuted uppercase text-[10px] tracking-widest font-bold">Suspicion</span>
                        <strong style={{ color: verdict.color }} className="font-mono text-base">{Math.round((result.overall_suspicion_score || 0) * 100)}%</strong>
                        <span className="text-borderHairline">|</span>
                        <span className="text-textMuted uppercase text-[10px] tracking-widest font-bold">Attack</span>
                        <strong style={{ color: attackColor }} className="capitalize text-base">{(result.attack_classification?.attack_type || 'unknown').replace(/_/g, ' ')}</strong>
                        <span className="text-borderHairline">|</span>
                        <span className="text-textMuted uppercase text-[10px] tracking-widest font-bold">Confidence</span>
                        <strong style={{ color: attackColor }} className="font-mono text-base">{Math.round((result.attack_classification?.confidence || 0) * 100)}%</strong>
                    </div>
                </div>
                <button onClick={onReset} className="px-6 py-3 rounded-xl border border-borderHairline bg-bgPanel hover:bg-bgPanelRaised text-textPrimary font-bold transition-colors whitespace-nowrap">
                    ↩ New Scan
                </button>
            </div>

            {/* Layer Scores */}
            <div className="bg-bgPanel border border-borderHairline rounded-[20px] p-6">
                <div className="font-bold text-textPrimary mb-4 flex items-center gap-2 text-sm uppercase tracking-widest">
                    <Microscope className="w-5 h-5 text-redPrimary" /> 5-Layer Detection Scores
                </div>
                <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
                    {Object.entries(result.layer_scores || {}).map(([layer, score]) => {
                        const pct = Math.round((score || 0) * 100);
                        const color = pct > 70 ? 'var(--status-critical)' : pct > 40 ? 'var(--status-warn)' : 'var(--status-safe)';
                        return (
                            <div key={layer} className="bg-bgVoid/50 rounded-xl p-4 border" style={{ borderColor: `${color}30` }}>
                                <div className="text-[11px] text-textMuted mb-2 font-bold tracking-wide uppercase">{LAYER_LABELS[layer] || layer}</div>
                                <div className="text-3xl font-mono font-extrabold" style={{ color }}>{pct}%</div>
                                <div className="h-1.5 bg-bgVoid rounded-full mt-3 overflow-hidden">
                                    <div className="h-full rounded-full transition-all duration-1000 ease-out" style={{ width: `${pct}%`, background: color }} />
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Attack probs */}
                {result.attack_classification?.probabilities && (
                    <div className="bg-bgPanel border border-borderHairline rounded-[20px] p-6">
                        <div className="font-bold text-textPrimary mb-6 flex items-center gap-2 text-sm uppercase tracking-widest">
                            <Target className="w-5 h-5 text-redPrimary" /> Probability Breakdown
                        </div>
                        {Object.entries(result.attack_classification.probabilities).sort(([, a], [, b]) => b - a).map(([type, prob]) => (
                            <ScoreBar key={type} label={type.replace(/_/g, ' ')} score={prob} color={ATTACK_COLORS[type] || 'var(--red-primary)'} />
                        ))}
                    </div>
                )}

                {/* Narrative */}
                {result.injection_pattern?.narrative && (
                    <div className="bg-bgPanel border border-borderHairline rounded-[20px] p-6">
                        <div className="font-bold text-textPrimary mb-6 flex items-center gap-2 text-sm uppercase tracking-widest">
                            <ClipboardList className="w-5 h-5 text-redPrimary" /> Attack Narrative
                        </div>
                        <pre className="font-mono text-[13px] text-textSecondary leading-relaxed whitespace-pre-wrap m-0 bg-bgVoid p-5 rounded-xl border border-borderHairline">
                            {result.injection_pattern.narrative}
                        </pre>
                    </div>
                )}
            </div>

            {/* Download */}
            <button onClick={() => {
                const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a'); a.href = url;
                a.download = `model_scan_${result.scan_id || Date.now()}.json`; a.click();
                URL.revokeObjectURL(url);
            }} className="w-full py-5 rounded-2xl border border-redPrimary/40 bg-redPrimary/10 text-redPrimary font-bold text-xl hover:bg-redPrimary/20 transition-colors flex justify-center items-center gap-3 mt-4">
                <Download className="w-6 h-6" /> Download Full Scan Report (JSON)
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
        <div className="relative w-full min-h-[calc(100vh-80px)] bg-transparent overflow-hidden">
            <div className="relative z-10 px-6 md:px-12 py-12 max-w-4xl mx-auto flex flex-col gap-8">
                
                <div className="mb-4">
                    <div className="flex items-center gap-3 mb-4">
                        <div className="w-10 h-10 rounded-xl bg-redPrimary/10 border border-redPrimary/30 flex items-center justify-center">
                            <Bot className="w-5 h-5 text-redPrimary" />
                        </div>
                        <h1 className="text-[48px] font-display font-bold text-textPrimary m-0 tracking-tight leading-none">
                            Model Poisoning Scanner
                        </h1>
                    </div>
                    <p className="text-textSecondary text-[18px] leading-relaxed">
                        Upload a trained sklearn <code className="bg-bgPanel px-2 py-0.5 rounded border border-borderHairline font-mono text-redPrimary">.pkl</code> model
                        — we extract its learned parameters and run all 5 detection layers to check if it was trained on poisoned data.
                        Optionally attach the training CSV for deeper analysis.
                    </p>
                    <div className="flex gap-2 mt-6 flex-wrap">
                        {['LogisticRegression', 'RandomForest', 'GradientBoosting', 'SVC', 'MLP', 'DecisionTree', 'SGD', 'KNeighbors', 'Ridge/Lasso', 'NaiveBayes'].map(m => (
                            <span key={m} className="text-[11px] font-mono px-3 py-1.5 rounded-full bg-redDim/20 border border-redPrimary/30 text-redPrimary tracking-wide">
                                {m}
                            </span>
                        ))}
                    </div>
                </div>

                {!result && (
                    <>
                        {/* Drop zone */}
                        <div onDrop={onDrop} onDragOver={e => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)}
                            className={`border-[3px] border-dashed rounded-[28px] p-16 text-center cursor-pointer backdrop-blur-xl transition-all duration-300 w-full flex flex-col items-center justify-center min-h-[300px]
                            ${dragging ? 'border-redPrimary bg-redPrimary/5 shadow-red-glow' : modelFile ? 'border-statusSafe bg-statusSafe/5' : 'border-borderHairline bg-bgVoid/40 hover:bg-bgPanel'}`}
                            onClick={() => modelRef.current?.click()}>
                            <input ref={modelRef} type="file" accept=".pkl" className="hidden" onChange={e => handleModelFile(e.target.files[0])} />
                            
                            <div className="mb-6">
                                {modelFile ? <Bot className="w-16 h-16 text-statusSafe" /> : dragging ? <FolderUp className="w-16 h-16 text-redPrimary animate-bounce" /> : <BrainCircuit className="w-16 h-16 text-textMuted" />}
                            </div>

                            {modelFile ? (
                                <>
                                    <div className="text-statusSafe font-bold text-2xl tracking-wide">{modelFile.name}</div>
                                    <div className="text-textMuted text-sm font-mono mt-3">{(modelFile.size / 1024).toFixed(1)} KB · Click to change</div>
                                </>
                            ) : (
                                <>
                                    <div className="text-textPrimary font-bold text-2xl tracking-wide">Drop your .pkl model here or click to browse</div>
                                    <div className="text-textMuted font-mono text-sm mt-3">Accepts sklearn .pkl files up to 50MB</div>
                                </>
                            )}
                        </div>

                        {/* Optional CSV */}
                        <div onClick={() => datasetRef.current?.click()} className={`border border-dashed rounded-[16px] p-5 cursor-pointer backdrop-blur-xl transition-all flex items-center gap-4 ${datasetFile ? 'border-statusSafe bg-statusSafe/5' : 'border-borderHairline bg-bgVoid/40 hover:bg-bgPanel'}`}>
                            <input ref={datasetRef} type="file" accept=".csv" className="hidden" onChange={e => handleDatasetFile(e.target.files[0])} />
                            <div className={`p-3 rounded-xl flex-shrink-0 ${datasetFile ? 'bg-statusSafe/20 text-statusSafe' : 'bg-bgPanel text-textMuted'}`}>
                                {datasetFile ? <FileText className="w-6 h-6" /> : <Plus className="w-6 h-6" />}
                            </div>
                            <div className="flex-1">
                                <div className={`text-sm font-bold ${datasetFile ? 'text-statusSafe' : 'text-textPrimary'}`}>
                                    {datasetFile ? datasetFile.name : 'Attach training dataset (optional CSV)'}
                                </div>
                                <div className="text-xs text-textMuted mt-1">Enables deeper dataset-level poisoning analysis alongside model scan</div>
                            </div>
                            {datasetFile && <button onClick={e => { e.stopPropagation(); setDatasetFile(null); }} className="p-2 text-textMuted hover:text-redPrimary transition-colors"><X className="w-5 h-5" /></button>}
                        </div>

                        {error && <div className="bg-statusCritical/10 border border-statusCritical/30 rounded-xl p-4 text-statusCritical font-bold text-sm">⚠️ {error}</div>}

                        {modelFile && (
                            <button onClick={runScan} disabled={loading} className="w-full py-5 rounded-2xl bg-redPrimary text-white font-bold text-xl hover:bg-redBright hover:shadow-red-glow transition-all disabled:opacity-50 disabled:cursor-not-allowed border border-redPrimary/50">
                                {loading ? '🔬 Scanning Model Parameters...' : '🚀 Scan for Poisoning'}
                            </button>
                        )}

                        {loading && (
                            <div className="p-6 bg-bgPanel border border-borderHairline rounded-[20px]">
                                <div className="flex justify-between text-sm text-textPrimary font-bold mb-3">
                                    <span>Extracting parameters → running 5-layer detection...</span>
                                    <span className="font-mono">{Math.round(progress)}%</span>
                                </div>
                                <div className="h-3 bg-bgVoid rounded-full overflow-hidden">
                                    <div className="h-full bg-redPrimary transition-all duration-300 ease-out shadow-red-glow" style={{ width: `${progress}%` }} />
                                </div>
                            </div>
                        )}
                    </>
                )}

                {result && <ResultPanel result={result} onReset={reset} />}
            </div>
            
            <style>{`@keyframes fadeInUp { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:translateY(0); } }`}</style>
        </div>
    );
}
