import { useState, useRef, useCallback } from 'react';
import { api } from '../services/api';
import { ShieldAlert, Check, Download, Bot, FolderUp, BrainCircuit, AlertTriangle } from 'lucide-react';

const EXTENSIONS = ['.pkl', '.pickle', '.joblib', '.pt', '.pth', '.bin', '.ckpt', '.safetensors', '.onnx', '.npy', '.npz', '.gguf', '.h5', '.keras', '.zip'];
const MAX_MB = 200;

const VERDICT_CONFIG = {
    DANGEROUS: { color: 'var(--status-critical)', bg: 'rgba(228,36,43,0.1)', icon: <ShieldAlert className="w-12 h-12 text-[var(--status-critical)]" />, label: 'DANGEROUS', blurb: 'Loading this file can run code or access the system. Do not load it.' },
    REVIEW: { color: 'var(--status-warn)', bg: 'rgba(242,184,75,0.1)', icon: <AlertTriangle className="w-12 h-12 text-[var(--status-warn)]" />, label: 'NEEDS REVIEW', blurb: 'Something could not be verified. A person should look before this file is loaded.' },
    NO_CODE_EXECUTION_FOUND: { color: '#3b82f6', bg: 'rgba(59,130,246,0.1)', icon: <Check className="w-12 h-12 text-blue-500" />, label: 'NO CODE EXECUTION FOUND', blurb: 'Every import is a known data container. This is not a guarantee of safety.' },
    SAFE_FORMAT: { color: 'var(--status-safe)', bg: 'rgba(61,220,132,0.1)', icon: <Check className="w-12 h-12 text-[var(--status-safe)]" />, label: 'SAFE FORMAT', blurb: 'This format holds weights only and cannot contain code.' },
};

const SEVERITY_COLOR = { critical: 'var(--status-critical)', high: 'var(--status-critical)', medium: 'var(--status-warn)', low: '#3b82f6', info: 'var(--text-muted)' };

function ResultPanel({ result, onReset }) {
    const scan = result.scan || {};
    const verdict = VERDICT_CONFIG[result.verdict] || VERDICT_CONFIG.REVIEW;
    return (
        <div className="animate-fadeInUp space-y-6">
            <div className="rounded-[24px] p-6 md:p-8 flex flex-col md:flex-row items-center gap-6 border" style={{ background: verdict.bg, borderColor: `${verdict.color}40` }}>
                <div>{verdict.icon}</div>
                <div className="flex-1 text-center md:text-left">
                    <div className="text-2xl md:text-3xl font-display font-bold tracking-wide" style={{ color: verdict.color }}>{verdict.label}</div>
                    <div className="text-textPrimary text-sm mt-3 font-medium">{verdict.blurb}</div>
                    <div className="text-textMuted text-xs font-mono mt-3 break-all">
                        {result.model_filename} · format: {scan.format} · {(scan.size_bytes / 1024).toFixed(1)} KB · sha256 {scan.sha256?.slice(0, 16)}…
                    </div>
                </div>
                <button onClick={onReset} className="px-6 py-3 rounded-xl border border-borderHairline bg-bgPanel hover:bg-bgPanelRaised text-textPrimary font-bold transition-colors whitespace-nowrap">
                    ↩ New Scan
                </button>
            </div>

            <div className="bg-bgPanel border border-borderHairline rounded-[20px] p-6">
                <div className="font-bold text-textPrimary mb-4 text-sm uppercase tracking-widest">Findings ({(scan.findings || []).length})</div>
                {(scan.findings || []).length === 0 ? (
                    <div className="text-textMuted text-sm">Nothing flagged.</div>
                ) : (
                    <ul className="space-y-3 m-0 p-0 list-none">
                        {scan.findings.map((f, i) => (
                            <li key={i} className="flex gap-3 items-start text-sm">
                                <span className="font-mono text-[10px] font-bold uppercase px-2 py-1 rounded border whitespace-nowrap" style={{ color: SEVERITY_COLOR[f.severity], borderColor: SEVERITY_COLOR[f.severity] }}>{f.severity}</span>
                                <span className="text-textSecondary break-words">{f.detail}</span>
                            </li>
                        ))}
                    </ul>
                )}
            </div>

            {(scan.imports || []).length > 0 && (
                <div className="bg-bgPanel border border-borderHairline rounded-[20px] p-6">
                    <div className="font-bold text-textPrimary mb-4 text-sm uppercase tracking-widest">Imports the file would perform ({scan.imports.length})</div>
                    <div className="flex flex-wrap gap-2">
                        {scan.imports.map(([m, n]) => (
                            <span key={`${m}.${n}`} className="font-mono text-xs px-2 py-1 rounded bg-bgVoid border border-borderHairline text-textSecondary">{m}.{n}</span>
                        ))}
                    </div>
                </div>
            )}

            {result.model_metadata && (
                <div className="bg-bgPanel border border-borderHairline rounded-[20px] p-6 text-sm text-textSecondary font-mono">
                    scikit-learn model: {result.model_metadata.model_type}
                </div>
            )}

            <div className="text-textMuted text-xs leading-relaxed">{result.interpretation}</div>

            <button onClick={() => {
                const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a'); a.href = url;
                a.download = `model_scan_${result.scan_id || Date.now()}.json`; a.click();
                URL.revokeObjectURL(url);
            }} className="w-full py-5 rounded-2xl border border-redPrimary/40 bg-redPrimary/10 text-redPrimary font-bold text-xl hover:bg-redPrimary/20 transition-colors flex justify-center items-center gap-3 mt-4">
                <Download className="w-6 h-6" /> Download Scan Report (JSON)
            </button>
        </div>
    );
}

export default function ModelScanPage() {
    const [modelFile, setModelFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [dragging, setDragging] = useState(false);
    const modelRef = useRef();

    const handleModelFile = (f) => {
        if (!f) return;
        if (!EXTENSIONS.some((ext) => f.name.toLowerCase().endsWith(ext))) { setError(`Supported files: ${EXTENSIONS.join(' ')}`); return; }
        if (f.size > MAX_MB * 1024 * 1024) { setError(`File too large. Maximum is ${MAX_MB}MB.`); return; }
        setModelFile(f); setError(null); setResult(null);
    };

    const onDrop = useCallback((e) => {
        e.preventDefault(); setDragging(false);
        handleModelFile(e.dataTransfer.files[0]);
    }, []);

    const runScan = async () => {
        if (!modelFile) return;
        setLoading(true); setError(null);
        try {
            const formData = new FormData();
            formData.append('model_file', modelFile);
            setResult(await api.scanModel(formData));
        } catch (e) {
            setError(e.message || 'Model scan failed.');
        } finally { setLoading(false); }
    };

    const reset = () => { setResult(null); setModelFile(null); setError(null); };

    return (
        <div className="relative w-full min-h-[calc(100vh-80px)] bg-transparent overflow-hidden">
            <div className="relative z-10 px-6 md:px-12 py-12 max-w-4xl mx-auto flex flex-col gap-8">
                <div className="mb-4">
                    <div className="flex items-center gap-3 mb-4">
                        <div className="w-10 h-10 rounded-xl bg-redPrimary/10 border border-redPrimary/30 flex items-center justify-center">
                            <Bot className="w-5 h-5 text-redPrimary" />
                        </div>
                        <h1 className="text-[48px] font-display font-bold text-textPrimary m-0 tracking-tight leading-none">Model File Scanner</h1>
                    </div>
                    <p className="text-textSecondary text-[18px] leading-relaxed">
                        Check whether a model file can run code when it is loaded. The file is analysed statically in an isolated process and is never loaded or executed.
                    </p>
                    <div className="flex gap-2 mt-6 flex-wrap">
                        {['pickle / joblib', 'PyTorch .pt/.pth/.bin', 'safetensors', 'ONNX', 'NumPy .npy/.npz', 'GGUF'].map((m) => (
                            <span key={m} className="text-[11px] font-mono px-3 py-1.5 rounded-full bg-redDim/20 border border-redPrimary/30 text-redPrimary tracking-wide">{m}</span>
                        ))}
                    </div>
                </div>

                {!result && (
                    <>
                        <div onDrop={onDrop} onDragOver={(e) => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)}
                            className={`border-[3px] border-dashed rounded-[28px] p-16 text-center cursor-pointer backdrop-blur-xl transition-all duration-300 w-full flex flex-col items-center justify-center min-h-[300px]
                            ${dragging ? 'border-redPrimary bg-redPrimary/5 shadow-red-glow' : modelFile ? 'border-statusSafe bg-statusSafe/5' : 'border-borderHairline bg-bgVoid/40 hover:bg-bgPanel'}`}
                            onClick={() => modelRef.current?.click()}>
                            <input ref={modelRef} type="file" className="hidden" onChange={(e) => handleModelFile(e.target.files[0])} />
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
                                    <div className="text-textPrimary font-bold text-2xl tracking-wide">Drop a model file here or click to browse</div>
                                    <div className="text-textMuted font-mono text-sm mt-3">Up to {MAX_MB}MB</div>
                                </>
                            )}
                        </div>

                        {error && <div className="bg-statusCritical/10 border border-statusCritical/30 rounded-xl p-4 text-statusCritical font-bold text-sm">⚠️ {error}</div>}

                        {modelFile && (
                            <button onClick={runScan} disabled={loading} className="w-full py-5 rounded-2xl bg-redPrimary text-white font-bold text-xl hover:bg-redBright hover:shadow-red-glow transition-all disabled:opacity-50 disabled:cursor-not-allowed border border-redPrimary/50">
                                {loading ? '🔬 Scanning...' : '🚀 Scan File'}
                            </button>
                        )}
                    </>
                )}

                {result && <ResultPanel result={result} onReset={reset} />}
            </div>
            <style>{`@keyframes fadeInUp { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:translateY(0); } }`}</style>
        </div>
    );
}
