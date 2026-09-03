const ATTACK_COLORS = {
  label_flip: '#f59e0b',
  backdoor: '#ef4444',
  clean_label: '#a855f7',
  gradient_poisoning: '#06b6d4',
  boiling_frog: '#22c55e',
};

function ScoreBar({ label, score, color = '#6366f1' }) {
  const pct = Math.round((score || 0) * 100);
  return (
    <div style={{ marginBottom: 10 }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          marginBottom: 4,
          fontSize: 12,
          color: '#94a3b8',
        }}
      >
        <span>{label}</span>
        <span style={{ color, fontWeight: 700 }}>{pct}%</span>
      </div>
      <div style={{ height: 6, background: 'rgba(255,255,255,0.06)', borderRadius: 4, overflow: 'hidden' }}>
        <div
          style={{
            height: '100%',
            width: `${pct}%`,
            background: `linear-gradient(90deg, ${color}88, ${color})`,
            borderRadius: 4,
            transition: 'width 1s ease',
          }}
        />
      </div>
    </div>
  );
}

export default function AttackClassificationCard({ classification, attackColor }) {
  if (!classification) return null;

  const attackType = classification.attack_type || 'unknown';

  return (
    <div
      style={{
        flex: 1,
        minWidth: 280,
        background: 'rgba(255,255,255,0.03)',
        border: `1px solid ${attackColor}33`,
        borderRadius: 14,
        padding: 20,
      }}
    >
      <div style={{ fontWeight: 700, color: '#f1f5f9', marginBottom: 14, fontSize: 14 }}>
        🎯 Attack Classification
      </div>
      <div style={{ textAlign: 'center', marginBottom: 16 }}>
        <div style={{ fontSize: 36, marginBottom: 8 }}>
          {attackType === 'backdoor'
            ? '🚪'
            : attackType === 'label_flip'
            ? '🔄'
            : attackType === 'clean_label'
            ? '🎭'
            : attackType === 'gradient_poisoning'
            ? '⚡'
            : '🐸'}
        </div>
        <div style={{ fontSize: 20, fontWeight: 800, color: attackColor, textTransform: 'capitalize' }}>
          {attackType.replace(/_/g, ' ')}
        </div>
        <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>
          Subtype: {classification.attack_subtype?.replace(/_/g, ' ') || '—'}
        </div>
      </div>

      <div style={{ marginBottom: 12 }}>
        <ScoreBar
          label="Classification Confidence"
          score={classification.confidence || 0}
          color={attackColor}
        />
      </div>

      <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.6 }}>{classification.description}</div>

      {classification.probabilities && (
        <div style={{ marginTop: 14 }}>
          <div
            style={{
              fontSize: 10,
              color: '#475569',
              textTransform: 'uppercase',
              letterSpacing: 1,
              marginBottom: 8,
            }}
          >
            Attack Probabilities
          </div>
          {Object.entries(classification.probabilities)
            .sort(([, a], [, b]) => b - a)
            .map(([type, prob]) => (
              <ScoreBar
                key={type}
                label={type.replace(/_/g, ' ')}
                score={prob}
                color={ATTACK_COLORS[type] || '#6366f1'}
              />
            ))}
        </div>
      )}
    </div>
  );
}
