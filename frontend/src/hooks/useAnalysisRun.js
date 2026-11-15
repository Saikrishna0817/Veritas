import { useCallback, useEffect, useRef, useState } from 'react';
import { runAnalysis, waitForJob, pendingJob } from '../services/uploads';

/**
 * Runs a check or model scan and tracks it. Small files return directly; large ones and media zips become a
 * background job shown as `job` ({ filename, stage, startedAt, jobId, kind }). A job left running when the page
 * closed is picked up again. `kinds` is one kind or a list (the Check data page runs several); the first is the
 * default for run(), and `resumedKind` says which kind a picked-up job is.
 */
export function useAnalysisRun(kinds) {
  const list = Array.isArray(kinds) ? kinds : [kinds];
  const [defaultKind] = list;
  // A job left running when the page closed is shown straight away and picked up again below.
  const [resumed] = useState(() => list.map((k) => pendingJob.get(k)).find(Boolean) || null);
  const [loading, setLoading] = useState(Boolean(resumed));
  const [job, setJob] = useState(() => (resumed
    ? { filename: resumed.filename, stage: 'queued', startedAt: resumed.startedAt, jobId: resumed.jobId, kind: resumed.kind }
    : null));
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const abort = useRef(null);

  const onJob = useCallback((j) => {
    setJob((cur) => (cur ? { ...cur, jobId: j.job_id, stage: j.status === 'running' ? 'running' : 'queued' } : cur));
  }, []);

  // A run whose page was left (or remounted, as StrictMode does in development) must not touch the state.
  const settle = useCallback((promise, signal) => promise
    .then((r) => { if (!signal.aborted) { setResult(r); window.scrollTo({ top: 0, behavior: 'smooth' }); } })
    .catch((e) => { if (!signal.aborted) setError(e.message || 'The analysis failed. Check the file and try again.'); })
    .finally(() => { if (!signal.aborted) { setLoading(false); setJob(null); } }), []);

  useEffect(() => {
    if (!resumed) return undefined;
    const ctrl = new AbortController();
    abort.current = ctrl;
    settle(waitForJob(resumed.jobId, { onJob, signal: ctrl.signal }).finally(() => { if (!ctrl.signal.aborted) pendingJob.clear(); }), ctrl.signal);
    return () => ctrl.abort();
  }, [resumed, settle, onJob]);

  useEffect(() => () => abort.current?.abort(), []);

  const run = useCallback((file, opts = {}, kind = defaultKind) => {
    const ctrl = new AbortController();
    abort.current = ctrl;
    const startedAt = Date.now();
    setLoading(true);
    setError('');
    return settle(runAnalysis(kind, file, opts, {
      signal: ctrl.signal,
      onJob,
      onStage: (stage) => setJob({ filename: file.name, stage, startedAt, jobId: null, kind }),
    }), ctrl.signal);
  }, [defaultKind, settle, onJob]);

  const reset = useCallback(() => { setResult(null); setError(''); }, []);

  return { run, loading, job, result, error, setError, reset, resumedKind: resumed?.kind || null };
}
