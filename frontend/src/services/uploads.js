/**
 * How a file reaches the analysis, chosen by size:
 *
 *   small  -> one request that returns the result (POST /analyze/upload, /analyze/model)
 *   large  -> direct to S3/R2 when the server has it (POST /uploads/presign, PUT, POST /uploads/complete),
 *             otherwise through the API as a background job (?background=true)
 *
 * Both large paths return a job id that is polled at GET /jobs/{id}. The finished job's result is the same
 * object a small upload returns, and the backend also saves it to History.
 */
import { api } from './api';

export const BACKGROUND_FROM_BYTES = 10 * 1024 * 1024;
const POLL_MS = 2000;
const JOB_KEY = 'veritas.pendingJob';

/** The job a page should resume after a reload. Storage may be unavailable; then nothing is resumed. */
export const pendingJob = {
  get(kind) {
    try {
      const job = JSON.parse(localStorage.getItem(JOB_KEY) || 'null');
      return job && job.kind === kind ? job : null;
    } catch {
      return null;
    }
  },
  set(job) {
    try { localStorage.setItem(JOB_KEY, JSON.stringify(job)); } catch { /* resume is a convenience only */ }
  },
  clear() {
    try { localStorage.removeItem(JOB_KEY); } catch { /* nothing to clear */ }
  },
};

// The browser sets Content-Length itself and refuses scripts that try to.
const FORBIDDEN_HEADERS = new Set(['content-length', 'host']);

async function putToStorage(presigned, file) {
  const headers = Object.fromEntries(Object.entries(presigned.headers || {}).filter(([k]) => !FORBIDDEN_HEADERS.has(k.toLowerCase())));
  const res = await fetch(presigned.url, { method: presigned.method || 'PUT', headers, body: file, credentials: 'omit' });
  if (!res.ok) throw new Error(`The file could not be uploaded to storage (HTTP ${res.status}).`);
}

/** Direct upload; returns a job id, or null when the server has no S3-compatible storage (501). */
async function tryDirect(kind, file, opts, onStage) {
  let main;
  try {
    main = await api.presignUpload(kind === 'model' ? 'model' : 'csv', file.name, file.size);
  } catch (err) {
    if (err.status === 501) return null;
    throw err;
  }
  const body = { upload_id: main.upload_id };
  onStage?.('uploading');
  await putToStorage(main, file);
  if (kind === 'csv' && opts.referenceFile) {
    const ref = await api.presignUpload('csv_reference', opts.referenceFile.name, opts.referenceFile.size);
    await putToStorage(ref, opts.referenceFile);
    body.reference_upload_id = ref.upload_id;
  }
  if (kind === 'csv' && opts.splitColumn && opts.splitCutoff) {
    body.split_column = opts.splitColumn;
    body.split_cutoff = opts.splitCutoff;
  }
  const { job_id } = await api.completeUpload(body);
  return job_id;
}

async function startJob(kind, file, opts, onStage) {
  const direct = await tryDirect(kind, file, opts, onStage);
  if (direct) return direct;
  onStage?.('uploading');
  const res = kind === 'model'
    ? await api.uploadModel(file, { background: true })
    : await api.uploadCSV(file, opts, { background: true });
  return res.job_id;
}

/** Poll until the job finishes. `onJob` receives every status update. Resolves to the analysis result. */
export async function waitForJob(jobId, { onJob, signal } = {}) {
  for (;;) {
    if (signal?.aborted) throw new DOMException('Stopped watching the job.', 'AbortError');
    const job = await api.getJob(jobId);
    onJob?.(job);
    if (job.status === 'completed') return job.result;
    if (job.status === 'failed') throw new Error(job.error || 'The analysis failed on the server.');
    await new Promise((resolve) => setTimeout(resolve, POLL_MS));
  }
}

/**
 * Run a CSV check or a model scan. kind: 'csv' | 'model'.
 * opts (csv): { referenceFile, splitColumn, splitCutoff }.
 * hooks: { onStage(stage), onJob(job), signal } — stage is 'uploading' | 'queued' | 'running'.
 */
export async function runAnalysis(kind, file, opts = {}, hooks = {}) {
  const total = file.size + (opts.referenceFile?.size || 0);
  if (total < BACKGROUND_FROM_BYTES) {
    return kind === 'model' ? api.uploadModel(file) : api.uploadCSV(file, opts);
  }
  const jobId = await startJob(kind, file, opts, hooks.onStage);
  pendingJob.set({ kind, jobId, filename: file.name, startedAt: Date.now() });
  try {
    return await waitForJob(jobId, hooks);
  } finally {
    if (!hooks.signal?.aborted) pendingJob.clear();
  }
}
