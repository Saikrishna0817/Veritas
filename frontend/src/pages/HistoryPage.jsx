import { useState, useEffect } from 'react';
import { api } from '../services/api';

const SOURCE_CONFIG = {
    demo: { color: '#6366f1', icon: '🎯', label: 'Demo' },
    upload: { color: '#22c55e', icon: '📂', label: 'Upload' },
    real_dataset: { color: '#06b6d4', icon: '📚', label: 'Real Dataset' },
    model_scan: { color: '#a855f7', icon: '🤖', label: 'Model Scan' },
};

const VERDICT_COLORS = {
    CONFIRMED_POISONED: '#ef4444',
    SUSPICIOUS: '#f59e0b',
    LOW_RISK: '#3b82f6',
    CLEAN: '#22c55e',
};

function StatCard({ label, value, color = '#6366f1', icon }) {
    return (
        <div style={{ background: 'rgba(0,0,0,0.03)', border: `1px solid ${color}22`, borderRadius: 12, padding: '16px 20px', flex: 1, minWidth: 120, textAlign: 'center' }}>
            <div style={{ fontSize: 24, marginBottom: 4 }}>{icon}</div>
            <div style={{ fontSize: 48, fontWeight: 900, color, fontFamily: 'monospace' }}>{value}</div>
            <div style={{ fontSize: 11, color: '#475569', marginTop: 2 }}>{label}</div>
        </div>
    );
}

export default function HistoryPage() {
    const [data, setData] = useState(null);
    const [source, setSource] = useState('');
    const [loading, setLoading] = useState(false);
    const [selected, setSelected] = useState(null);
    const [fullResult, setFullResult] = useState(null);
    const [loadingFull, setLoadingFull] = useState(false);

    const load = async (src) => {
        setLoading(true);
        try {
            const res = await api.getHistory(src || null);
            setData(res);
        } catch (e) { console.error(e); }
        finally { setLoading(false); }
    };

    useEffect(() => { load(source); }, [source]);

    const loadFull = async (id) => {
        setSelected(id); setLoadingFull(true); setFullResult(null);
        try {
            const res = await api.getHistoricalResult(id);
            setFullResult(res);
        } catch (e) { console.error(e); }
        finally { setLoadingFull(false); }
    };

    const stats = data?.stats || {};
    const results = data?.results || [];

    return (
        <div style={{ padding: '48px 64px', width: '100%', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '32px' }}>
            <div style={{ marginBottom: 28 }}>
                <h1 style={{ fontSize: 48, fontWeight: 900, color: '#141414', margin: 0 }}>🗄️ Analysis History</h1>
                <p style={{ color: '#475569', marginTop: 12, fontSize: 18, maxWidth: '80%', lineHeight: 1.6 }}>
                    All past analyses persisted in SQLite — survives server restarts. Click any row to load the full result.
                </p>
            </div>

            {/* Stats */}
            {stats.total_analyses !== undefined && (
                <div style={{ display: 'flex', gap: 12, marginBottom: 24, flexWrap: 'wrap' }}>
                    <StatCard icon="📊" label="Total Analyses" value={stats.total_analyses} color="#6366f1" />
                    <StatCard icon="🤖" label="Model Scans" value={stats.model_scans || 0} color="#a855f7" />
                    <StatCard icon="☠️" label="Poisoned Found" value={stats.by_verdict?.CONFIRMED_POISONED || 0} color="#ef4444" />
                    <StatCard icon="⚠️" label="Suspicious" value={stats.by_verdict?.SUSPICIOUS || 0} color="#f59e0b" />
                    <StatCard icon="✅" label="Clean" value={stats.by_verdict?.CLEAN || 0} color="#22c55e" />
                </div>
            )}

            {/* Filter */}
            <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
                {['', 'demo', 'upload', 'real_dataset', 'model_scan'].map(s => {
                    const cfg = SOURCE_CONFIG[s] || { color: '#6366f1', icon: '🔍', label: 'All' };
                    const active = source === s;
                    return (
                        <button key={s} onClick={() => setSource(s)} style={{ padding: '7px 14px', borderRadius: 20, border: `1px solid ${active ? cfg.color : 'rgba(255,255,255,0.1)'}`, background: active ? `${cfg.color}20` : 'rgba(255,255,255,0.03)', color: active ? cfg.color : '#64748b', fontWeight: active ? 700 : 400, fontSize: 13, cursor: 'pointer', transition: 'all 0.2s' }}>
                            {cfg.icon} {cfg.label || 'All'}
                        </button>
                    );
                })}
                <button onClick={() => load(source)} style={{ marginLeft: 'auto', padding: '7px 14px', borderRadius: 20, border: '1px solid rgba(0,0,0,0.15)', background: 'rgba(255,255,255,0.5)', color: '#475569', fontSize: 13, cursor: 'pointer' }}>
                    {loading ? '⏳' : '🔄'} Refresh
                </button>
            </div>

            {/* Table */}
            <div style={{ background: 'rgba(255,255,255,0.4)', border: '1px solid rgba(0,0,0,0.1)', borderRadius: 14, overflow: 'hidden', marginBottom: 24 }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 80px 100px 120px 100px 140px', gap: 0, padding: '10px 16px', background: 'rgba(0,0,0,0.03)', fontSize: 10, color: '#475569', textTransform: 'uppercase', letterSpacing: 1 }}>
                    <span>Filename / ID</span><span>Source</span><span>Verdict</span><span>Attack Type</span><span>Samples</span><span>Date</span>
                </div>
                {results.length === 0 && (
                    <div style={{ padding: '32px 16px', textAlign: 'center', color: '#475569', fontSize: 14 }}>
                        {loading ? 'Loading...' : 'No results yet. Run a demo, upload a CSV, or scan a model.'}
                    </div>
                )}
                {results.map((row, i) => {
                    const src = SOURCE_CONFIG[row.source] || { color: '#6366f1', icon: '?', label: row.source };
                    const vColor = VERDICT_COLORS[row.verdict] || '#64748b';
                    const isSelected = selected === row.id;
                    return (
                        <div key={row.id} onClick={() => loadFull(row.id)} style={{ display: 'grid', gridTemplateColumns: '1fr 80px 100px 120px 100px 140px', gap: 0, padding: '12px 16px', borderTop: '1px solid rgba(255,255,255,0.05)', cursor: 'pointer', background: isSelected ? 'rgba(232,98,44,0.05)' : i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.01)', transition: 'background 0.2s' }}>
                            <div>
                                <div style={{ fontSize: 13, color: '#cbd5e1', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{row.filename || '—'}</div>
                                <div style={{ fontSize: 10, color: '#475569', fontFamily: 'monospace', marginTop: 2 }}>{row.id?.slice(0, 16)}…</div>
                            </div>
                            <span style={{ fontSize: 12, color: src.color }}>{src.icon} {src.label}</span>
                            <span style={{ fontSize: 12, fontWeight: 700, color: vColor }}>{row.verdict || '—'}</span>
                            <span style={{ fontSize: 12, color: '#334155', textTransform: 'capitalize' }}>{(row.attack_type || '—').replace(/_/g, ' ')}</span>
                            <span style={{ fontSize: 12, color: '#475569', fontFamily: 'monospace' }}>{row.n_samples || '—'}</span>
                            <span style={{ fontSize: 11, color: '#475569' }}>{row.created_at ? new Date(row.created_at.endsWith('Z') ? row.created_at : row.created_at + 'Z').toLocaleString() : '—'}</span>
                        </div>
                    );
                })}
            </div>

            {/* Full result panel */}
            {selected && (
                <div style={{ background: 'rgba(255,255,255,0.5)', border: '1px solid rgba(0,0,0,0.1)', borderRadius: 14, padding: 20 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
                        <div style={{ fontWeight: 700, color: '#141414', fontSize: 14 }}>📄 Full Result</div>
                        <div style={{ display: 'flex', gap: 8 }}>
                            {fullResult && (
                                <button onClick={() => {
                                    const blob = new Blob([JSON.stringify(fullResult, null, 2)], { type: 'application/json' });
                                    const url = URL.createObjectURL(blob);
                                    const a = document.createElement('a'); a.href = url;
                                    a.download = `result_${selected.slice(0, 8)}.json`; a.click();
                                    URL.revokeObjectURL(url);
                                }} style={{ padding: '6px 12px', borderRadius: 8, border: '2px solid rgba(232,98,44,0.4)', background: 'rgba(232,98,44,0.1)', color: '#E8622C', cursor: 'pointer', fontSize: 12 }}>
                                    ⬇️ Download JSON
                                </button>
                            )}
                            <button onClick={() => { setSelected(null); setFullResult(null); }} style={{ padding: '6px 12px', borderRadius: 8, border: '1px solid rgba(0,0,0,0.15)', background: 'rgba(0,0,0,0.05)', color: '#334155', cursor: 'pointer', fontSize: 12 }}>✕ Close</button>
                        </div>
                    </div>
                    {loadingFull && <div style={{ color: '#475569', fontSize: 13 }}>Loading full result...</div>}
                    {fullResult && (
                        <pre style={{ fontFamily: 'monospace', fontSize: 11, color: '#334155', lineHeight: 1.6, whiteSpace: 'pre-wrap', margin: 0, maxHeight: 400, overflow: 'auto', background: 'rgba(0,0,0,0.2)', padding: 14, borderRadius: 8 }}>
                            {JSON.stringify(fullResult, null, 2)}
                        </pre>
                    )}
                </div>
            )}
        </div>
    );
}
