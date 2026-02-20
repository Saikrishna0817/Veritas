import { useState, useEffect } from 'react';
import { api } from '../services/api';
import { Users, Shield, AlertTriangle, RefreshCw } from 'lucide-react';
import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer, Tooltip } from 'recharts';

function TrustMeter({ score, clientId, quarantined }) {
    const pct = score * 100;
    const color = quarantined ? '#ff4d6a' : pct >= 70 ? '#00ffc8' : pct >= 40 ? '#ffd166' : '#ff4d6a';
    const status = quarantined ? 'QUARANTINED' : pct >= 70 ? 'TRUSTED' : pct >= 40 ? 'SUSPICIOUS' : 'MALICIOUS';

    return (
        <div className={`bg-bg3 border rounded-lg p-4 transition-all ${quarantined ? 'border-danger/40 bg-danger/5' :
                pct >= 70 ? 'border-accent3/20' : 'border-yellow/30'
            }`}>
            <div className="flex items-center justify-between mb-3">
                <span className="font-mono text-xs text-text2">{clientId}</span>
                <span className={`font-mono text-xs px-2 py-0.5 rounded uppercase`}
                    style={{ color, background: `${color}15`, border: `1px solid ${color}40` }}>
                    {status}
                </span>
            </div>
            <div className="h-2 bg-border rounded-full overflow-hidden mb-2">
                <div className="h-full rounded-full transition-all duration-1000"
                    style={{ width: `${pct}%`, background: color, boxShadow: `0 0 8px ${color}` }} />
            </div>
            <div className="flex justify-between font-mono text-xs">
                <span className="text-text3">Trust Score</span>
                <span style={{ color }} className="font-bold">{pct.toFixed(1)}%</span>
            </div>
        </div>
    );
}

export default function FederatedPage() {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);

    const load = async () => {
        setLoading(true);
        try {
            const res = await api.getFederatedClients();
            setData(res);
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => { load(); }, []);

    const radarData = data?.clients?.slice(0, 6).map(c => ({
        client: c.client_id.slice(-6),
        trust: Math.round(c.trust_score * 100),
        similarity: Math.round((c.cosine_similarity + 1) / 2 * 100),
    })) || [];

    return (
        <div className="p-8 space-y-8 animate-in">
            <div className="flex items-start justify-between">
                <div>
                    <div className="font-mono text-xs text-purple tracking-widest uppercase mb-2 flex items-center gap-2">
                        <Users className="w-3 h-3" /> Federated Learning
                    </div>
                    <h1 className="text-4xl font-bold tracking-tight">
                        Federated <span className="text-purple">Trust</span>
                    </h1>
                    <p className="font-mono text-sm text-text2 mt-2">
                        Sybil-resistant trust scoring · Behavioral fingerprinting · Auto-quarantine
                    </p>
                </div>
                <button onClick={load} disabled={loading}
                    className="flex items-center gap-2 px-4 py-2 border border-purple/30 bg-purple/10 text-purple font-mono text-xs rounded-lg hover:bg-purple/20 transition-all">
                    <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} />
                    Refresh
                </button>
            </div>

            {data && (
                <>
                    {/* Summary */}
                    <div className="grid grid-cols-3 gap-4">
                        {[
                            { label: 'Total Clients', value: data.clients?.length || 0, color: 'accent' },
                            { label: 'Quarantined', value: data.n_quarantined || 0, color: 'danger' },
                            { label: 'Avg Trust', value: `${((data.avg_trust || 0) * 100).toFixed(1)}%`, color: data.avg_trust > 0.7 ? 'accent3' : 'yellow' },
                        ].map(({ label, value, color }) => (
                            <div key={label} className="bg-surface border border-border rounded-lg p-5 text-center">
                                <div className={`text-3xl font-bold text-${color}`}>{value}</div>
                                <div className="font-mono text-xs text-text3 mt-1">{label}</div>
                            </div>
                        ))}
                    </div>

                    {/* Quarantine Alert */}
                    {data.n_quarantined > 0 && (
                        <div className="bg-danger/5 border border-danger/30 rounded-lg p-4 flex items-center gap-3">
                            <AlertTriangle className="w-5 h-5 text-danger flex-shrink-0" />
                            <div className="font-mono text-sm text-danger">
                                {data.n_quarantined} client(s) quarantined: {data.quarantined_clients?.join(', ')}
                            </div>
                        </div>
                    )}

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {/* Client Trust Meters */}
                        <div className="bg-surface border border-border rounded-lg p-6">
                            <div className="font-mono text-xs text-purple uppercase tracking-widest mb-4">
                                Client Trust Scores
                            </div>
                            <div className="space-y-3">
                                {data.clients?.map(client => (
                                    <TrustMeter
                                        key={client.client_id}
                                        score={client.trust_score}
                                        clientId={client.client_id}
                                        quarantined={client.quarantined}
                                    />
                                ))}
                            </div>
                        </div>

                        {/* Radar Chart */}
                        <div className="bg-surface border border-border rounded-lg p-6">
                            <div className="font-mono text-xs text-purple uppercase tracking-widest mb-4">
                                Behavioral Fingerprint Radar
                            </div>
                            {radarData.length > 0 ? (
                                <ResponsiveContainer width="100%" height={300}>
                                    <RadarChart data={radarData}>
                                        <PolarGrid stroke="#1e3a52" />
                                        <PolarAngleAxis dataKey="client"
                                            tick={{ fill: '#4a7a9b', fontSize: 10, fontFamily: 'JetBrains Mono' }} />
                                        <Radar name="Trust" dataKey="trust" stroke="#bd93f9" fill="#bd93f9" fillOpacity={0.2} />
                                        <Radar name="Similarity" dataKey="similarity" stroke="#00e5ff" fill="#00e5ff" fillOpacity={0.1} />
                                        <Tooltip contentStyle={{ background: '#080f18', border: '1px solid #1e3a52', fontFamily: 'JetBrains Mono', fontSize: 11 }} />
                                    </RadarChart>
                                </ResponsiveContainer>
                            ) : (
                                <div className="flex items-center justify-center h-48 text-text3 font-mono text-sm">
                                    No client data
                                </div>
                            )}
                            <div className="flex gap-4 mt-2 font-mono text-xs text-text3">
                                <span className="flex items-center gap-2"><span className="w-3 h-0.5 bg-purple inline-block" /> Trust</span>
                                <span className="flex items-center gap-2"><span className="w-3 h-0.5 bg-accent inline-block" /> Similarity</span>
                            </div>
                        </div>
                    </div>
                </>
            )}

            {!data && !loading && (
                <div className="text-center py-20 text-text3 font-mono text-sm">
                    Loading federated client data...
                </div>
            )}
        </div>
    );
}
