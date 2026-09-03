const LAYER_LABELS = {
  statistical: 'L1 Statistical Shift',
  spectral: 'L2 Spectral Analysis',
  ensemble: 'L3 Ensemble Anomaly',
  causal: 'L4 Causal Proof',
  federated: 'L5 Federated Trust',
  shap_drift: 'SHAP Drift',
};

export default function LayerScoresGrid({ layerScores }) {
  if (!layerScores) return null;

  return (
    <div
      style={{
        background: 'rgba(255,255,255,0.03)',
        border: '1px solid rgba(255,255,255,0.08)',
        borderRadius: 14,
        padding: 20,
        marginBottom: 24,
      }}
    >
      <div style={{ fontWeight: 700, color: '#f1f5f9', marginBottom: 16, fontSize: 14 }}>
        🔬 5-Layer Detection Scores
      </div>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
          gap: 12,
        }}
      >
        {Object.entries(layerScores).map(([layer, score]) => {
          const pct = Math.round((score || 0) * 100);
          const color = pct > 70 ? '#ef4444' : pct > 40 ? '#f59e0b' : '#22c55e';
          return (
            <div
              key={layer}
              style={{
                background: 'rgba(255,255,255,0.03)',
                borderRadius: 10,
                padding: '14px 16px',
                border: `1px solid ${color}22`,
              }}
            >
              <div style={{ fontSize: 11, color: '#64748b', marginBottom: 8 }}>
                {LAYER_LABELS[layer] || layer}
              </div>
              <div style={{ fontSize: 24, fontWeight: 800, color, fontFamily: 'monospace' }}>{pct}%</div>
              <div
                style={{
                  height: 4,
                  background: 'rgba(255,255,255,0.06)',
                  borderRadius: 2,
                  marginTop: 8,
                  overflow: 'hidden',
                }}
              >
                <div
                  style={{
                    height: '100%',
                    width: `${pct}%`,
                    background: color,
                    borderRadius: 2,
                    transition: 'width 1s ease',
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
