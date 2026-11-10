import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { CloudUpload, Hourglass, Cpu } from 'lucide-react';

const STEPS = [
  { id: 'uploading', label: 'Uploading', icon: CloudUpload },
  { id: 'queued', label: 'Waiting for a worker', icon: Hourglass },
  { id: 'running', label: 'Analysing', icon: Cpu },
];

function clock(ms) {
  const s = Math.max(0, Math.floor(ms / 1000));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

/** Progress for a large file analysed as a background job. stage: 'uploading' | 'queued' | 'running'. */
export default function JobCard({ filename, stage, startedAt, jobId }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  const active = Math.max(0, STEPS.findIndex((s) => s.id === stage));

  return (
    <div className="glass-card" role="status" aria-live="polite">
      <div className="p-6">
        <div className="flex items-baseline justify-between gap-3">
          <div className="min-w-0">
            <div className="rule-label">Large file · running in the background</div>
            <div className="mt-1 truncate font-mono text-sm">{filename}</div>
          </div>
          <span className="font-mono text-sm text-textSecondary tabular" aria-label="Elapsed time">{clock(now - startedAt)}</span>
        </div>

        <ol className="mt-6 grid grid-cols-3 gap-2">
          {STEPS.map(({ id, label, icon }, i) => {
            const state = i < active ? 'done' : i === active ? 'now' : 'next';
            const Icon = icon;
            return (
              <li key={id} className="space-y-2">
                <div className="h-1.5 overflow-hidden rounded-full bg-white/[0.08]">
                  {state === 'done' && <div className="h-full w-full rounded-full bg-uv" />}
                  {state === 'now' && (
                    <motion.div
                      className="h-full w-1/3 rounded-full bg-uv"
                      animate={{ x: ['-100%', '300%'] }}
                      transition={{ duration: 1.6, repeat: Infinity, ease: 'easeInOut' }}
                    />
                  )}
                </div>
                <div className={`flex items-center gap-1.5 text-xs ${state === 'next' ? 'text-textMuted' : 'text-textPrimary'}`}>
                  <Icon className="h-3.5 w-3.5" aria-hidden /> {label}
                </div>
              </li>
            );
          })}
        </ol>

        <p className="mt-5 text-sm text-textSecondary">
          You can leave this page. Coming back here picks the job up again, and the result is saved to{' '}
          <Link to="/history" className="text-violet underline-offset-4 hover:underline">History</Link> when it finishes.
        </p>
        {jobId && <p className="mt-2 font-mono text-[11px] text-textMuted">job {jobId}</p>}
      </div>
    </div>
  );
}
