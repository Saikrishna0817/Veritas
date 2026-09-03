import { useState, useRef, useCallback } from 'react';
import { api } from '../services/api';
import UploadSchemaCard from '../components/upload/UploadSchemaCard';
import AttackClassificationCard from '../components/upload/AttackClassificationCard';
import LayerScoresGrid from '../components/upload/LayerScoresGrid';
import Tactile3DHero from '../components/Tactile3DHero';

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

function StatCard({ label, value, sub, color = '#E8622C', icon }) {
  return (
    <div
      style={{
        background: 'rgba(0,0,0,0.03)',
        border: `1px solid ${color}40`,
        borderRadius: 16,
        padding: '24px',
        flex: 1,
        minWidth: 180,
      }}
    >
      <div style={{ fontSize: 28, marginBottom: 8 }}>{icon}</div>
      <div style={{ fontSize: 28, fontWeight: 900, color, fontFamily: 'monospace' }}>{value}</div>
      <div style={{ fontSize: 13, color: '#334155', marginTop: 4, fontWeight: 600 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{sub}</div>}
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
    if (f.size > 200 * 1024 * 1024) {
      setError('File too large. Maximum size is 200MB.');
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

  const onDragOver = (e) => {
    e.preventDefault();
    setDragging(true);
  };
  const onDragLeave = () => setDragging(false);

  const runAnalysis = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setProgress(0);

    const interval = setInterval(() => {
      setProgress((p) => (p < 85 ? p + Math.random() * 12 : p));
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

  const verdict = result ? VERDICT_CONFIG[result.verdict] || VERDICT_CONFIG.SUSPICIOUS : null;
  const attackColor = result
    ? ATTACK_COLORS[result.attack_classification?.attack_type] || '#E8622C'
    : '#E8622C';

  return (
    <div style={{ position: 'relative', width: '100%', minHeight: 'calc(100vh - 80px)', overflow: 'hidden', background: 'transparent' }}>
      
      {/* Main Content Area */}
      <div style={{ position: 'relative', zIndex: 1, padding: '48px 64px', width: '100%', margin: '0 auto', display: 'flex', flexDirection: 'col', gap: '32px' }}>
        
        {/* Header */}
        <div style={{ marginBottom: 16 }}>
          <h1 style={{ fontSize: 48, fontWeight: 900, color: '#141414', margin: 0, letterSpacing: '-0.02em' }}>
            📂 Upload Dataset for Analysis
          </h1>
          <p style={{ color: '#334155', marginTop: 12, fontSize: 18, maxWidth: '80%', lineHeight: 1.6 }}>
            Upload any CSV file — the platform auto-detects schema, splits 70/30 for baseline, and runs all 5 detection layers.
            Supports up to 200,000 rows · 200MB · supervised &amp; unsupervised modes.
          </p>
        </div>

        {/* Upload Zone */}
        <div
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onClick={() => fileRef.current?.click()}
          style={{
            border: `3px dashed ${dragging ? '#E8622C' : file ? '#22c55e' : 'rgba(0,0,0,0.15)'}`,
            borderRadius: 24,
            padding: '80px 48px',
            textAlign: 'center',
            cursor: 'pointer',
            background: dragging
              ? 'rgba(232, 98, 44, 0.05)'
              : file
              ? 'rgba(34,197,94,0.05)'
              : 'rgba(255,255,255,0.4)',
            backdropFilter: 'blur(10px)',
            transition: 'all 0.3s ease',
            width: '100%',
            marginBottom: 24,
          }}
        >
          <input
            ref={fileRef}
            type="file"
            accept=".csv"
            style={{ display: 'none' }}
            onChange={(e) => handleFile(e.target.files[0])}
          />
          <div style={{ fontSize: 72, marginBottom: 20 }}>
            {file ? '📄' : dragging ? '📥' : '☁️'}
          </div>
          {file ? (
            <>
              <div style={{ color: '#22c55e', fontWeight: 800, fontSize: 24 }}>{file.name}</div>
              <div style={{ color: '#475569', fontSize: 16, marginTop: 8 }}>
                {(file.size / 1024 / 1024).toFixed(2)} MB · Click to change
              </div>
            </>
          ) : (
            <>
              <div style={{ color: '#141414', fontWeight: 800, fontSize: 24 }}>
                Drop your CSV here or click to browse
              </div>
              <div style={{ color: '#475569', fontSize: 16, marginTop: 12 }}>
                Accepts .csv files up to 200MB (≤ 200,000 rows)
              </div>
            </>
          )}
        </div>

        {/* Error */}
        {error && (
          <div
            style={{
              background: 'rgba(239,68,68,0.1)',
              border: '2px solid rgba(239,68,68,0.3)',
              borderRadius: 16,
              padding: '16px 24px',
              color: '#dc2626',
              marginBottom: 24,
              fontSize: 16,
              fontWeight: 600
            }}
          >
            ⚠️ {error}
          </div>
        )}

        {/* Analyze Button */}
        {file && !result && (
          <button
            onClick={runAnalysis}
            disabled={loading}
            style={{
              width: '100%',
              padding: '20px 0',
              borderRadius: 16,
              border: 'none',
              background: loading ? '#ccc' : '#E8622C',
              color: '#fff',
              fontWeight: 800,
              fontSize: 20,
              cursor: loading ? 'not-allowed' : 'pointer',
              transition: 'all 0.3s',
              marginBottom: 24,
              boxShadow: loading ? 'none' : '0 8px 24px rgba(232, 98, 44, 0.3)',
            }}
          >
            {loading ? '🔬 Analyzing...' : '🚀 Run Poisoning Detection'}
          </button>
        )}

        {/* Progress Bar */}
        {loading && (
          <div style={{ marginBottom: 32, padding: '24px', background: 'rgba(255,255,255,0.5)', borderRadius: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 14, color: '#334155', marginBottom: 12, fontWeight: 700 }}>
              <span>Running 5-layer detection pipeline...</span>
              <span>{Math.round(progress)}%</span>
            </div>
            <div style={{ height: 12, background: 'rgba(0,0,0,0.06)', borderRadius: 6, overflow: 'hidden' }}>
              <div
                style={{
                  height: '100%',
                  width: `${progress}%`,
                  background: '#E8622C',
                  borderRadius: 6,
                  transition: 'width 0.4s ease',
                  boxShadow: '0 0 12px rgba(232, 98, 44, 0.6)',
                }}
              />
            </div>
            <div style={{ display: 'flex', gap: 12, marginTop: 16, flexWrap: 'wrap' }}>
              {['Schema Detection', 'L1 Statistical', 'L2 Spectral', 'L3 Ensemble', 'L4 Causal', 'L5 Federated'].map((s, i) => (
                <span
                  key={s}
                  style={{
                    fontSize: 13,
                    padding: '6px 12px',
                    borderRadius: 20,
                    fontWeight: 600,
                    background: progress > i * 14 ? 'rgba(232, 98, 44, 0.1)' : 'rgba(0,0,0,0.04)',
                    color: progress > i * 14 ? '#E8622C' : '#64748b',
                    border: `1px solid ${progress > i * 14 ? 'rgba(232, 98, 44, 0.3)' : 'rgba(0,0,0,0.06)'}`,
                    transition: 'all 0.5s',
                  }}
                >
                  {progress > i * 14 ? '✓ ' : ''}
                  {s}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Results */}
        {result && (
          <div style={{ animation: 'fadeIn 0.5s ease', background: 'rgba(255,255,255,0.7)', borderRadius: 24, padding: 32, backdropFilter: 'blur(10px)', boxShadow: '0 10px 40px rgba(0,0,0,0.05)' }}>
            {/* Verdict Banner */}
            <div
              style={{
                background: verdict.bg,
                border: `2px solid ${verdict.color}44`,
                borderRadius: 20,
                padding: '24px 32px',
                marginBottom: 32,
                display: 'flex',
                alignItems: 'center',
                gap: 24,
              }}
            >
              <div style={{ fontSize: 56 }}>{verdict.icon}</div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 28, fontWeight: 900, color: verdict.color, letterSpacing: 1 }}>
                  {verdict.label}
                </div>
                <div style={{ color: '#475569', fontSize: 16, marginTop: 6, fontWeight: 500 }}>
                  Suspicion Score:{' '}
                  <strong style={{ color: verdict.color, fontSize: 18 }}>
                    {Math.round((result.overall_suspicion_score || 0) * 100)}%
                  </strong>
                  &nbsp;·&nbsp; Poisoning Level:{' '}
                  <strong style={{ color: verdict.color, fontSize: 18 }}>{result.poisoning_level || 'N/A'}</strong>
                  &nbsp;·&nbsp; Mode:{' '}
                  <strong style={{ color: '#334155', fontSize: 18 }}>
                    {result.detection_mode === 'supervised' ? '🏷️ Supervised' : '🔍 Unsupervised'}
                  </strong>
                </div>
              </div>
              <button
                onClick={() => {
                  setResult(null);
                  setFile(null);
                  setProgress(0);
                }}
                style={{
                  padding: '12px 24px',
                  borderRadius: 12,
                  border: '2px solid rgba(0,0,0,0.1)',
                  background: 'rgba(0,0,0,0.05)',
                  color: '#141414',
                  fontWeight: 700,
                  cursor: 'pointer',
                  fontSize: 16,
                  transition: 'background 0.2s'
                }}
                onMouseOver={(e) => e.target.style.background = 'rgba(0,0,0,0.1)'}
                onMouseOut={(e) => e.target.style.background = 'rgba(0,0,0,0.05)'}
              >
                ↩ New Upload
              </button>
            </div>

            {/* Schema & Classification Row */}
            <div style={{ display: 'flex', gap: 24, marginBottom: 32, flexWrap: 'wrap' }}>
              <div style={{ flex: 1, minWidth: '400px' }}>
                <UploadSchemaCard datasetInfo={result.dataset_info} nSamples={result.n_samples} />
              </div>
              <div style={{ flex: 1, minWidth: '400px' }}>
                <AttackClassificationCard
                  classification={result.attack_classification}
                  attackColor={attackColor}
                />
              </div>
            </div>

            {/* Layer Scores */}
            <div style={{ marginBottom: 32 }}>
              <LayerScoresGrid layerScores={result.layer_scores} />
            </div>

            {/* Stats Row */}
            <div style={{ display: 'flex', gap: 24, marginBottom: 32, flexWrap: 'wrap' }}>
              <StatCard
                icon="🎯"
                label="Sophistication Score"
                value={`${result.sophistication?.sophistication_score || 0}/10`}
                sub={result.sophistication?.level}
                color="#F2E85C"
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
              <div
                style={{
                  background: 'rgba(0,0,0,0.02)',
                  border: '1px solid rgba(0,0,0,0.1)',
                  borderRadius: 16,
                  padding: 24,
                  marginBottom: 32,
                }}
              >
                <div style={{ fontWeight: 800, color: '#141414', marginBottom: 16, fontSize: 18 }}>
                  📋 Attack Reconstruction Narrative
                </div>
                <pre
                  style={{
                    fontFamily: 'monospace',
                    fontSize: 14,
                    color: '#334155',
                    lineHeight: 1.8,
                    whiteSpace: 'pre-wrap',
                    margin: 0,
                    background: 'rgba(255,255,255,0.8)',
                    padding: 24,
                    borderRadius: 12,
                    border: '1px solid rgba(0,0,0,0.05)'
                  }}
                >
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
                width: '100%',
                padding: '16px 0',
                borderRadius: 16,
                border: '2px solid rgba(232, 98, 44, 0.4)',
                background: 'rgba(232, 98, 44, 0.1)',
                color: '#E8622C',
                fontWeight: 800,
                fontSize: 18,
                cursor: 'pointer',
                transition: 'all 0.3s',
              }}
              onMouseOver={(e) => e.target.style.background = 'rgba(232, 98, 44, 0.2)'}
              onMouseOut={(e) => e.target.style.background = 'rgba(232, 98, 44, 0.1)'}
            >
              ⬇️ Download Forensic Report (JSON)
            </button>
          </div>
        )}

      </div>
      
      <style>{`
        @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
      `}</style>
    </div>
  );
}
