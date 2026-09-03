export default function UploadSchemaCard({ datasetInfo, nSamples }) {
  if (!datasetInfo) return null;

  return (
    <div
      style={{
        flex: 1,
        minWidth: 280,
        background: 'rgba(255,255,255,0.03)',
        border: '1px solid rgba(255,255,255,0.08)',
        borderRadius: 14,
        padding: 20,
      }}
    >
      <div style={{ fontWeight: 700, color: '#f1f5f9', marginBottom: 14, fontSize: 14 }}>
        📊 Dataset Schema
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
        {[
          { k: 'Filename', v: datasetInfo.filename || '—' },
          { k: 'Total Rows', v: datasetInfo.n_rows?.toLocaleString() || '—' },
          { k: 'Features', v: datasetInfo.n_features || '—' },
          { k: 'Label Column', v: datasetInfo.label_column || 'None (unsupervised)' },
          { k: 'Reference Split', v: `${datasetInfo.reference_split || 0} rows (70%)` },
          { k: 'Analyzed', v: `${nSamples || 0} rows (30%)` },
        ].map(({ k, v }) => (
          <div key={k}>
            <div style={{ fontSize: 10, color: '#475569', textTransform: 'uppercase', letterSpacing: 1 }}>
              {k}
            </div>
            <div
              style={{
                fontSize: 13,
                color: '#cbd5e1',
                fontWeight: 600,
                marginTop: 2,
                wordBreak: 'break-all',
              }}
            >
              {v}
            </div>
          </div>
        ))}
      </div>

      {datasetInfo.feature_names?.length > 0 && (
        <div style={{ marginTop: 14 }}>
          <div
            style={{
              fontSize: 10,
              color: '#475569',
              textTransform: 'uppercase',
              letterSpacing: 1,
              marginBottom: 6,
            }}
          >
            Feature Columns
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {datasetInfo.feature_names.map((f) => (
              <span
                key={f}
                style={{
                  fontSize: 11,
                  padding: '3px 8px',
                  borderRadius: 20,
                  background: 'rgba(99,102,241,0.15)',
                  color: '#a5b4fc',
                  border: '1px solid rgba(99,102,241,0.25)',
                }}
              >
                {f}
              </span>
            ))}
          </div>
        </div>
      )}

      {datasetInfo.warnings?.length > 0 && (
        <div
          style={{
            marginTop: 12,
            padding: '8px 12px',
            background: 'rgba(245,158,11,0.08)',
            borderRadius: 8,
            border: '1px solid rgba(245,158,11,0.2)',
          }}
        >
          {datasetInfo.warnings.map((w, i) => (
            <div key={i} style={{ fontSize: 12, color: '#fbbf24' }}>
              ⚠️ {w}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
