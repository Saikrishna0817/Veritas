import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./api', () => ({
  api: {
    uploadCSV: vi.fn(),
    uploadModel: vi.fn(),
    presignUpload: vi.fn(),
    completeUpload: vi.fn(),
    getJob: vi.fn(),
  },
}));

const { api } = await import('./api');
const { runAnalysis, waitForJob, pendingJob, BACKGROUND_FROM_BYTES } = await import('./uploads');

const MB = 1024 * 1024;
const small = { name: 'small.csv', size: 1 * MB };
const large = { name: 'large.csv', size: BACKGROUND_FROM_BYTES + MB };
const notImplemented = () => Object.assign(new Error('Direct upload needs S3-compatible storage.'), { status: 501 });

function memoryStorage() {
  const m = new Map();
  return { getItem: (k) => (m.has(k) ? m.get(k) : null), setItem: (k, v) => m.set(k, String(v)), removeItem: (k) => m.delete(k) };
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.stubGlobal('localStorage', memoryStorage());
  vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, status: 200 })));
  Object.values(api).forEach((fn) => fn.mockReset());
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('runAnalysis', () => {
  it('sends a small CSV in one request and never touches jobs', async () => {
    api.uploadCSV.mockResolvedValue({ verdict: 'CLEAN' });
    const opts = { splitColumn: 'day', splitCutoff: '2026-01-01' };

    await expect(runAnalysis('csv', small, opts)).resolves.toEqual({ verdict: 'CLEAN' });
    expect(api.uploadCSV).toHaveBeenCalledWith(small, opts);
    expect(api.presignUpload).not.toHaveBeenCalled();
    expect(api.getJob).not.toHaveBeenCalled();
  });

  it('sends a small model file to the model endpoint', async () => {
    api.uploadModel.mockResolvedValue({ verdict: 'SAFE_FORMAT' });
    await runAnalysis('model', { name: 'm.safetensors', size: MB });
    expect(api.uploadModel).toHaveBeenCalledWith({ name: 'm.safetensors', size: MB });
  });

  it('counts the reference file towards the size limit', async () => {
    api.presignUpload.mockRejectedValue(notImplemented());
    api.uploadCSV.mockResolvedValue({ job_id: 'j1', status: 'queued' });
    api.getJob.mockResolvedValue({ job_id: 'j1', status: 'completed', result: { verdict: 'CLEAN' } });
    const reference = { name: 'ref.csv', size: BACKGROUND_FROM_BYTES };

    await runAnalysis('csv', small, { referenceFile: reference });
    expect(api.uploadCSV).toHaveBeenCalledWith(small, { referenceFile: reference }, { background: true });
  });

  it('falls back to a background job when the server has no S3 storage (501), and polls it', async () => {
    api.presignUpload.mockRejectedValue(notImplemented());
    api.uploadCSV.mockResolvedValue({ job_id: 'j2', status: 'queued' });
    api.getJob
      .mockResolvedValueOnce({ job_id: 'j2', status: 'queued' })
      .mockResolvedValueOnce({ job_id: 'j2', status: 'running' })
      .mockResolvedValueOnce({ job_id: 'j2', status: 'completed', result: { verdict: 'SUSPICIOUS' } });
    const stages = [];
    const statuses = [];

    const run = runAnalysis('csv', large, {}, { onStage: (s) => stages.push(s), onJob: (j) => statuses.push(j.status) });
    await vi.advanceTimersByTimeAsync(0);
    expect(pendingJob.get('csv')).toMatchObject({ jobId: 'j2', filename: 'large.csv' });   // resumable while running
    await vi.advanceTimersByTimeAsync(5000);

    await expect(run).resolves.toEqual({ verdict: 'SUSPICIOUS' });
    expect(stages).toEqual(['uploading']);
    expect(statuses).toEqual(['queued', 'running', 'completed']);
    expect(pendingJob.get('csv')).toBeNull();   // cleared once finished
  });

  it('uploads straight to storage when presign works, with the reference and split', async () => {
    api.presignUpload
      .mockResolvedValueOnce({ upload_id: 'u-main', url: 'https://bucket/main', method: 'PUT', headers: { 'Content-Length': '1' } })
      .mockResolvedValueOnce({ upload_id: 'u-ref', url: 'https://bucket/ref', method: 'PUT', headers: {} });
    api.completeUpload.mockResolvedValue({ job_id: 'j3', status: 'queued' });
    api.getJob.mockResolvedValue({ job_id: 'j3', status: 'completed', result: { verdict: 'CLEAN' } });
    const reference = { name: 'ref.csv', size: MB };

    await expect(runAnalysis('csv', large, { referenceFile: reference, splitColumn: 'day', splitCutoff: 'x' })).resolves.toEqual({ verdict: 'CLEAN' });
    expect(api.presignUpload).toHaveBeenNthCalledWith(1, 'csv', 'large.csv', large.size);
    expect(api.presignUpload).toHaveBeenNthCalledWith(2, 'csv_reference', 'ref.csv', MB);
    // The browser forbids scripts from setting Content-Length, so it must not be passed on.
    expect(fetch).toHaveBeenCalledWith('https://bucket/main', expect.objectContaining({ method: 'PUT', headers: {}, body: large, credentials: 'omit' }));
    expect(api.completeUpload).toHaveBeenCalledWith({ upload_id: 'u-main', reference_upload_id: 'u-ref', split_column: 'day', split_cutoff: 'x' });
    expect(api.uploadCSV).not.toHaveBeenCalled();
  });

  it('uses the model role for a large model file', async () => {
    api.presignUpload.mockResolvedValue({ upload_id: 'u', url: 'https://bucket/m', headers: {} });
    api.completeUpload.mockResolvedValue({ job_id: 'j4' });
    api.getJob.mockResolvedValue({ status: 'completed', result: { verdict: 'DANGEROUS' } });
    await runAnalysis('model', { name: 'big.pt', size: large.size });
    expect(api.presignUpload).toHaveBeenCalledWith('model', 'big.pt', large.size);
    expect(api.completeUpload).toHaveBeenCalledWith({ upload_id: 'u' });
  });

  it('reports a failed storage PUT instead of queueing anything', async () => {
    api.presignUpload.mockResolvedValue({ upload_id: 'u', url: 'https://bucket/x', headers: {} });
    fetch.mockResolvedValueOnce({ ok: false, status: 403 });
    await expect(runAnalysis('csv', large)).rejects.toThrow('HTTP 403');
    expect(api.completeUpload).not.toHaveBeenCalled();
  });

  it('passes on errors other than 501 from presign', async () => {
    api.presignUpload.mockRejectedValue(Object.assign(new Error('Daily analyses quota reached'), { status: 429 }));
    await expect(runAnalysis('csv', large)).rejects.toThrow('quota');
    expect(api.uploadCSV).not.toHaveBeenCalled();
  });
});

describe('waitForJob', () => {
  it('throws the server error when the job fails', async () => {
    api.getJob.mockResolvedValue({ status: 'failed', error: 'Only CSV files are accepted.' });
    await expect(waitForJob('j')).rejects.toThrow('Only CSV files are accepted.');
  });

  it('stops polling once aborted', async () => {
    api.getJob.mockResolvedValue({ status: 'running' });
    const ctrl = new AbortController();
    const run = waitForJob('j', { signal: ctrl.signal });
    const settled = expect(run).rejects.toMatchObject({ name: 'AbortError' });
    await vi.advanceTimersByTimeAsync(0);
    ctrl.abort();
    await vi.advanceTimersByTimeAsync(2000);
    await settled;
    expect(api.getJob).toHaveBeenCalledTimes(1);
  });
});

describe('pendingJob', () => {
  it('only returns a job of the asked kind, and survives storage being unavailable', () => {
    pendingJob.set({ kind: 'model', jobId: 'm1' });
    expect(pendingJob.get('csv')).toBeNull();
    expect(pendingJob.get('model')).toMatchObject({ jobId: 'm1' });

    vi.stubGlobal('localStorage', { getItem: () => { throw new Error('blocked'); }, setItem: () => { throw new Error('blocked'); }, removeItem: () => { throw new Error('blocked'); } });
    expect(() => pendingJob.set({ kind: 'csv' })).not.toThrow();
    expect(pendingJob.get('csv')).toBeNull();
    expect(() => pendingJob.clear()).not.toThrow();
  });
});
