import { useState, useRef, useCallback } from 'react';
import { api } from '../services/api';
import UploadSchemaCard from '../components/upload/UploadSchemaCard';
import AttackClassificationCard from '../components/upload/AttackClassificationCard';
import LayerScoresGrid from '../components/upload/LayerScoresGrid';
import { FolderUp, CloudUpload, FileText, Target, Zap, Activity, ShieldAlert, ClipboardList, Download, Check } from 'lucide-react';

const ATTACK_COLORS = {
  label_flip: '#f59e0b',
  backdoor: 'var(--red-primary)',
  clean_label: '#a855f7',
  gradient_poisoning: '#06b6d4',
  boiling_frog: 'var(--status-safe)',
};

const VERDICT_CONFIG = {
  CONFIRMED_POISONED: { color: 'var(--status-critical)', bg: 'rgba(228,36,43,0.1)', icon: <ShieldAlert className="w-12 h-12 text-[var(--status-critical)]" />, label: 'CONFIRMED POISONED' },
  SUSPICIOUS: { color: 'var(--status-warn)', bg: 'rgba(242,184,75,0.1)', icon: <Activity className="w-12 h-12 text-[var(--status-warn)]" />, label: 'SUSPICIOUS' },
  LOW_RISK: { color: '#3b82f6', bg: 'rgba(59,130,246,0.1)', icon: <Check className="w-12 h-12 text-blue-500" />, label: 'LOW RISK' },
  CLEAN: { color: 'var(--status-safe)', bg: 'rgba(61,220,132,0.1)', icon: <Check className="w-12 h-12 text-[var(--status-safe)]" />, label: 'CLEAN' },
};

function StatCard({ label, value, sub, color = 'var(--red-primary)', icon }) {
  return (
    <div className="bg-bgPanel border border-borderHairline rounded-[20px] p-6 flex-1 min-w-[180px] hover:bg-bgPanelRaised transition-colors">
      <div className="mb-3 w-10 h-10 rounded-lg flex items-center justify-center bg-black/20 border border-borderHairline text-white">
        {icon}
      </div>
      <div className="text-3xl font-mono font-extrabold" style={{ color }}>{value}</div>
      <div className="text-[13px] text-textPrimary mt-1 font-bold tracking-wide uppercase">{label}</div>
      {sub && <div className="text-[11px] text-textMuted mt-1">{sub}</div>}
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
    ? ATTACK_COLORS[result.attack_classification?.attack_type] || 'var(--red-primary)'
    : 'var(--red-primary)';

  return (
    <div className="relative w-full min-h-[calc(100vh-80px)] bg-transparent overflow-hidden">
      <div className="relative z-10 px-6 md:px-12 py-12 max-w-7xl mx-auto flex flex-col gap-12">
        
        {!result && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-start">
            {/* Left Column: Context / How it works */}
            <div className="space-y-8">
              <div>
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 rounded-xl bg-redPrimary/10 border border-redPrimary/30 flex items-center justify-center">
                    <FolderUp className="w-5 h-5 text-redPrimary" />
                  </div>
                  <h1 className="text-[56px] font-display font-bold text-textPrimary m-0 tracking-tight leading-none">
                    Upload Dataset
                  </h1>
                </div>
                <p className="text-textSecondary text-[18px] leading-relaxed max-w-md">
                  Upload any CSV file — the platform auto-detects schema, splits 70/30 for baseline, and runs all 5 detection layers.
                </p>
              </div>

              <div className="bg-bgPanel border border-borderHairline rounded-[24px] p-8 space-y-6">
                <h3 className="font-display font-bold text-textPrimary text-xl">How it works</h3>
                <ol className="space-y-6 relative before:absolute before:inset-y-0 before:left-4 before:w-[2px] before:bg-borderHairline">
                  <li className="relative pl-12">
                    <div className="absolute left-[9px] top-1 w-3.5 h-3.5 bg-redPrimary rounded-full ring-4 ring-bgPanel" />
                    <h4 className="font-bold text-textPrimary mb-1">1. Secure Upload</h4>
                    <p className="text-textMuted text-sm">Accepts .csv files up to 200MB (≤ 200,000 rows). Processed in memory without storage.</p>
                  </li>
                  <li className="relative pl-12">
                    <div className="absolute left-[9px] top-1 w-3.5 h-3.5 bg-redPrimary/30 border border-redPrimary rounded-full ring-4 ring-bgPanel" />
                    <h4 className="font-bold text-textPrimary mb-1">2. Auto-split & Baseline</h4>
                    <p className="text-textMuted text-sm">Automatically detects schema and splits 70/30 to establish a nominal statistical baseline.</p>
                  </li>
                  <li className="relative pl-12">
                    <div className="absolute left-[9px] top-1 w-3.5 h-3.5 bg-borderHairline rounded-full ring-4 ring-bgPanel" />
                    <h4 className="font-bold text-textPrimary mb-1">3. Run 5 Detection Layers</h4>
                    <p className="text-textMuted text-sm">Executes Statistical, Spectral, Ensemble, Causal, and Federated validation layers.</p>
                  </li>
                </ol>
              </div>
            </div>

            {/* Right Column: Upload Zone */}
            <div className="flex flex-col gap-6">
              <div
                onDrop={onDrop}
                onDragOver={onDragOver}
                onDragLeave={onDragLeave}
                onClick={() => fileRef.current?.click()}
                className={`border-[3px] border-dashed rounded-[28px] p-16 text-center cursor-pointer backdrop-blur-xl transition-all duration-300 w-full flex flex-col items-center justify-center min-h-[400px]
                  ${dragging ? 'border-redPrimary bg-redPrimary/5 shadow-red-glow' : file ? 'border-statusSafe bg-statusSafe/5' : 'border-borderHairline bg-bgVoid/40 hover:bg-bgPanel'}`}
              >
                <input
                  ref={fileRef}
                  type="file"
                  accept=".csv"
                  className="hidden"
                  onChange={(e) => handleFile(e.target.files[0])}
                />
                <div className="mb-6">
                  {file ? <FileText className="w-16 h-16 text-statusSafe" /> : dragging ? <FolderUp className="w-16 h-16 text-redPrimary animate-bounce" /> : <CloudUpload className="w-16 h-16 text-textMuted" />}
                </div>
                {file ? (
                  <>
                    <div className="text-statusSafe font-bold text-2xl tracking-wide">{file.name}</div>
                    <div className="text-textMuted text-sm font-mono mt-3">
                      {(file.size / 1024 / 1024).toFixed(2)} MB · Click to change
                    </div>
                  </>
                ) : (
                  <>
                    <div className="text-textPrimary font-bold text-2xl tracking-wide">
                      Drop your CSV here or click to browse
                    </div>
                    <div className="text-textMuted font-mono text-sm mt-3">
                      Accepts .csv files up to 200MB (≤ 200,000 rows)
                    </div>
                  </>
                )}
              </div>

              {error && (
                <div className="bg-statusCritical/10 border border-statusCritical/30 rounded-xl p-4 text-statusCritical font-bold text-sm">
                  ⚠️ {error}
                </div>
              )}

              {file && !result && (
                <button
                  onClick={runAnalysis}
                  disabled={loading}
                  className="w-full py-5 rounded-2xl bg-redPrimary text-white font-bold text-xl hover:bg-redBright hover:shadow-red-glow transition-all disabled:opacity-50 disabled:cursor-not-allowed border border-redPrimary/50"
                >
                  {loading ? '🔬 Analyzing...' : '🚀 Run Poisoning Detection'}
                </button>
              )}

              {loading && (
                <div className="p-6 bg-bgPanel border border-borderHairline rounded-[20px]">
                  <div className="flex justify-between text-sm text-textPrimary font-bold mb-3">
                    <span>Running 5-layer detection pipeline...</span>
                    <span className="font-mono">{Math.round(progress)}%</span>
                  </div>
                  <div className="h-3 bg-bgVoid rounded-full overflow-hidden">
                    <div
                      className="h-full bg-redPrimary transition-all duration-300 ease-out shadow-red-glow"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                  <div className="flex gap-2 mt-4 flex-wrap">
                    {['Schema Detection', 'L1 Statistical', 'L2 Spectral', 'L3 Ensemble', 'L4 Causal', 'L5 Federated'].map((s, i) => (
                      <span
                        key={s}
                        className={`text-[11px] font-mono px-3 py-1.5 rounded-full border transition-all duration-500
                          ${progress > i * 14 ? 'bg-redPrimary/10 border-redPrimary/30 text-redPrimary' : 'bg-bgVoid/50 border-borderHairline text-textMuted'}`}
                      >
                        {progress > i * 14 ? '✓ ' : ''}{s}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="animate-fadeInUp bg-bgVoid/80 backdrop-blur-2xl rounded-[32px] p-8 md:p-12 border border-borderHairline shadow-2xl">
            {/* Verdict Banner */}
            <div
              className="rounded-[24px] p-6 md:p-8 mb-10 flex items-center gap-8 border"
              style={{ background: verdict.bg, borderColor: `${verdict.color}40` }}
            >
              <div>{verdict.icon}</div>
              <div className="flex-1">
                <div className="text-3xl md:text-4xl font-display font-bold tracking-wide" style={{ color: verdict.color }}>
                  {verdict.label}
                </div>
                <div className="text-textPrimary text-lg mt-3 font-medium flex items-center gap-3 flex-wrap">
                  <span className="text-textMuted uppercase text-xs tracking-wider font-bold">Suspicion Score</span>
                  <strong style={{ color: verdict.color }} className="font-mono">{Math.round((result.overall_suspicion_score || 0) * 100)}%</strong>
                  <span className="text-borderHairline">|</span>
                  <span className="text-textMuted uppercase text-xs tracking-wider font-bold">Poisoning Level</span>
                  <strong style={{ color: verdict.color }}>{result.poisoning_level || 'N/A'}</strong>
                  <span className="text-borderHairline">|</span>
                  <span className="text-textMuted uppercase text-xs tracking-wider font-bold">Mode</span>
                  <strong className="text-textPrimary">
                    {result.detection_mode === 'supervised' ? 'Supervised' : 'Unsupervised'}
                  </strong>
                </div>
              </div>
              <button
                onClick={() => {
                  setResult(null);
                  setFile(null);
                  setProgress(0);
                }}
                className="px-6 py-3 rounded-xl border border-borderHairline bg-bgPanel hover:bg-bgPanelRaised text-textPrimary font-bold transition-colors"
              >
                ↩ New Upload
              </button>
            </div>

            {/* Schema & Classification Row */}
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-8 mb-10">
              <UploadSchemaCard datasetInfo={result.dataset_info} nSamples={result.n_samples} />
              <AttackClassificationCard classification={result.attack_classification} attackColor={attackColor} />
            </div>

            {/* Layer Scores */}
            <div className="mb-10">
              <LayerScoresGrid layerScores={result.layer_scores} />
            </div>

            {/* Stats Row */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
              <StatCard
                icon={<Target className="w-6 h-6" />}
                label="Sophistication Score"
                value={`${result.sophistication?.sophistication_score || 0}/10`}
                sub={result.sophistication?.level}
                color="var(--status-warn)"
              />
              <StatCard
                icon={<Zap className="w-6 h-6" />}
                label="Batches Affected"
                value={result.blast_radius?.n_batches_affected || 0}
                sub={`${result.blast_radius?.n_models_affected || 0} models`}
                color="var(--status-critical)"
              />
              <StatCard
                icon={<Activity className="w-6 h-6" />}
                label="Prediction Impact"
                value={`${result.blast_radius?.prediction_impact_pct || 0}%`}
                sub="accuracy degradation"
                color="var(--status-warn)"
              />
              <StatCard
                icon={<ShieldAlert className="w-6 h-6" />}
                label="Defense Action"
                value={result.defense_action?.action?.replace(/_/g, ' ') || 'monitor'}
                sub={result.defense_action?.reason?.replace(/_/g, ' ')}
                color="var(--status-safe)"
              />
            </div>

            {/* Narrative */}
            {result.injection_pattern?.narrative && (
              <div className="bg-bgPanel border border-borderHairline rounded-[24px] p-8 mb-10">
                <div className="font-bold text-textPrimary mb-4 flex items-center gap-2 text-xl">
                  <ClipboardList className="w-6 h-6 text-redPrimary" /> Attack Reconstruction Narrative
                </div>
                <pre className="font-mono text-sm text-textSecondary leading-relaxed whitespace-pre-wrap m-0 bg-bgVoid p-6 rounded-xl border border-borderHairline">
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
              className="w-full py-5 rounded-2xl border border-redPrimary/40 bg-redPrimary/10 text-redPrimary font-bold text-xl hover:bg-redPrimary/20 transition-colors flex justify-center items-center gap-3"
            >
              <Download className="w-6 h-6" /> Download Forensic Report (JSON)
            </button>
          </div>
        )}

      </div>
    </div>
  );
}
