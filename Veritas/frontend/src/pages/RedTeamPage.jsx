import { useState } from 'react';
import { api } from '../services/api';
import { Target, Zap, Shield, CheckCircle, AlertTriangle } from 'lucide-react';

const ATTACK_TYPES = [
    { id: 'label_flip', label: 'Label Flip', desc: 'Flips labels of training samples', severity: 'medium', color: 'yellow' },
    { id: 'backdoor', label: 'Backdoor Attack', desc: 'Hidden trigger pattern injection', severity: 'critical', color: 'danger' },
    { id: 'clean_label', label: 'Clean Label', desc: 'Feature-space perturbation, correct labels', severity: 'critical', color: 'purple' },
    { id: 'gradient_poisoning', label: 'Gradient Poisoning', desc: 'Inverted gradient signals disrupt training', severity: 'high', color: 'accent' },
    { id: 'boiling_frog', label: 'Boiling Frog', desc: 'Gradual drift injection', severity: 'high', color: 'orange' },
];

function ResilienceGauge({ score }) {
    const pct = (score / 10) * 100;
    const color = score >= 7 ? '#00ffc8' : score >= 4 ? '#ffd166' : '#ff4d6a';
    return (
        <div className="flex flex-col items-center">
            <div className="relative w-32 h-32">
                <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
                    <circle cx="50" cy="50" r="40" fill="none" stroke="#1e3a52" strokeWidth="8" />
                    <circle cx="50" cy="50" r="40" fill="none" strokeWidth="8"
                        stroke={color}
                        strokeDasharray={`${2 * Math.PI * 40 * pct / 100} ${2 * Math.PI * 40}`}
                        strokeLinecap="round"
                        style={{ filter: `drop-shadow(0 0 6px ${color})` }} />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <span className="text-3xl font-bold" style={{ color }}>{score}</span>
                    <span className="font-mono text-xs text-text3">/10</span>
                </div>
            </div>
            <div className="font-mono text-xs text-text3 mt-2 uppercase tracking-widest">Resilience Score</div>
        </div>
    );
}

export default function RedTeamPage() {
    const [selectedAttack, setSelectedAttack] = useState('label_flip');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [history, setHistory] = useState([]);
    const [error, setError] = useState(null);

    const runSimulation = async () => {
        setLoading(true);
        setError(null);
        try {
            const res = await api.runRedTeam(selectedAttack);
            setResult(res);
            setHistory(prev => [res, ...prev].slice(0, 10));
        } catch (e) {
            setError(e.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="p-8 space-y-8 animate-in">
            <div>
                <div className="font-mono text-xs text-danger tracking-widest uppercase mb-2 flex items-center gap-2">
                    <Target className="w-3 h-3" /> Red-Team Mode
                </div>
                <h1 className="text-4xl font-bold tracking-tight">
                    Red-Team <span className="text-danger">Simulator</span>
                </h1>
                <p className="font-mono text-sm text-text2 mt-2">
                    Inject synthetic attacks and measure platform resilience
                </p>
            </div>

            {error && (
                <div className="bg-danger/10 border border-danger/30 rounded-lg p-4 font-mono text-sm text-danger">
                    ⚠ {error} — Make sure the backend is running.
                </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Attack Selector */}
                <div className="bg-surface border border-border rounded-lg p-6">
                    <div className="font-mono text-xs text-accent uppercase tracking-widest mb-4">
                        Select Attack Type
                    </div>
                    <div className="space-y-3">
                        {ATTACK_TYPES.map(attack => (
                            <button
                                key={attack.id}
                                onClick={() => setSelectedAttack(attack.id)}
                                className={`w-full text-left p-4 rounded-lg border transition-all duration-200 ${selectedAttack === attack.id
                                    ? `border-${attack.color}/60 bg-${attack.color}/10`
                                    : 'border-border bg-bg3 hover:border-border2'
                                    }`}
                            >
                                <div className="flex items-center justify-between mb-1">
                                    <span className={`font-bold text-sm ${selectedAttack === attack.id ? `text-${attack.color}` : 'text-text1'}`}>
                                        {attack.label}
                                    </span>
                                    <span className={`font-mono text-xs px-2 py-0.5 rounded border border-${attack.color}/30 text-${attack.color} bg-${attack.color}/10 uppercase`}>
                                        {attack.severity}
                                    </span>
                                </div>
                                <div className="font-mono text-xs text-text3">{attack.desc}</div>
                            </button>
                        ))}
                    </div>

                    <button
                        onClick={runSimulation}
                        disabled={loading}
                        className="w-full mt-6 py-3 bg-danger/10 border border-danger/40 rounded-lg
                       font-mono text-sm text-danger hover:bg-danger/20 transition-all
                       flex items-center justify-center gap-2 disabled:opacity-50"
                    >
                        {loading ? (
                            <><Zap className="w-4 h-4 animate-pulse" /> Injecting Attack...</>
                        ) : (
                            <><Zap className="w-4 h-4" /> Inject Attack & Test Detection</>
                        )}
                    </button>
                </div>

                {/* Result */}
                <div className="bg-surface border border-border rounded-lg p-6">
                    <div className="font-mono text-xs text-accent uppercase tracking-widest mb-4">
                        Simulation Result
                    </div>
                    {!result ? (
                        <div className="flex flex-col items-center justify-center h-48 text-text3 font-mono text-sm">
                            <Target className="w-8 h-8 mb-3 opacity-30" />
                            Select an attack type and click Inject
                        </div>
                    ) : (
                        <div className="space-y-6">
                            <div className="flex items-center justify-between">
                                <ResilienceGauge score={result.resilience_score} />
                                <div className="flex-1 ml-6 space-y-3">
                                    <div className="flex items-center gap-2">
                                        {result.detected
                                            ? <CheckCircle className="w-5 h-5 text-accent3" />
                                            : <AlertTriangle className="w-5 h-5 text-danger" />}
                                        <span className={`font-mono text-sm font-bold ${result.detected ? 'text-accent3' : 'text-danger'}`}>
                                            {result.detected ? 'ATTACK DETECTED' : 'EVADED DETECTION'}
                                        </span>
                                    </div>
                                    {[
                                        { label: 'Attack Type', value: result.attack_type?.replace(/_/g, ' ') },
                                        { label: 'Samples Injected', value: result.n_injected },
                                        { label: 'Suspicion Score', value: `${(result.suspicion_score * 100).toFixed(1)}%` },
                                        { label: 'Detection Speed', value: `${result.detection_speed_ms}ms` },
                                        { label: 'False Positive Rate', value: `${(result.false_positive_rate * 100).toFixed(1)}%` },
                                    ].map(({ label, value }) => (
                                        <div key={label} className="flex justify-between font-mono text-xs">
                                            <span className="text-text3">{label}</span>
                                            <span className="text-text2 font-medium">{value}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* History */}
            {history.length > 0 && (
                <div className="bg-surface border border-border rounded-lg p-6">
                    <div className="font-mono text-xs text-accent uppercase tracking-widest mb-4">
                        Simulation History
                    </div>
                    <div className="overflow-x-auto">
                        <table className="w-full font-mono text-xs">
                            <thead>
                                <tr className="border-b border-border text-text3">
                                    <th className="text-left py-2 pr-4">Attack</th>
                                    <th className="text-left py-2 pr-4">Detected</th>
                                    <th className="text-left py-2 pr-4">Suspicion</th>
                                    <th className="text-left py-2 pr-4">Speed</th>
                                    <th className="text-left py-2">Resilience</th>
                                </tr>
                            </thead>
                            <tbody>
                                {history.map((h, i) => (
                                    <tr key={i} className="border-b border-border/40 hover:bg-bg3">
                                        <td className="py-2 pr-4 text-text2 capitalize">{h.attack_type?.replace(/_/g, ' ')}</td>
                                        <td className="py-2 pr-4">
                                            <span className={h.detected ? 'text-accent3' : 'text-danger'}>
                                                {h.detected ? '✓ Yes' : '✗ No'}
                                            </span>
                                        </td>
                                        <td className="py-2 pr-4 text-text2">{(h.suspicion_score * 100).toFixed(1)}%</td>
                                        <td className="py-2 pr-4 text-text2">{h.detection_speed_ms}ms</td>
                                        <td className="py-2">
                                            <span className={
                                                h.resilience_score >= 7 ? 'text-accent3' :
                                                    h.resilience_score >= 4 ? 'text-yellow' : 'text-danger'
                                            }>{h.resilience_score}/10</span>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    );
}
