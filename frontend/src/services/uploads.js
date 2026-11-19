/**
 * Large files go straight to S3/R2 when the server has it: POST /uploads/presign, PUT to the returned URL,
 * POST /uploads/complete (which queues a job). With local storage presign answers 501 and `startDirect` returns null,
 * so the caller uses its existing path. `api` is passed in (the api.js object) so this file has no imports.
 */
export const DIRECT_FROM_BYTES = 10 * 1024 * 1024;

// The browser sets Content-Length itself and refuses scripts that try to.
const FORBIDDEN_HEADERS = new Set(['content-length', 'host']);

async function putToStorage(presigned, file) {
  const headers = Object.fromEntries(Object.entries(presigned.headers || {}).filter(([k]) => !FORBIDDEN_HEADERS.has(k.toLowerCase())));
  const res = await fetch(presigned.url, { method: presigned.method || 'PUT', headers, body: file, credentials: 'omit' });
  if (!res.ok) throw new Error(`The file could not be uploaded to storage (HTTP ${res.status}).`);
}

/**
 * kind: 'table' | 'model'. opts (table): { referenceFile, splitColumn, splitCutoff, noReference }.
 * Resolves to a job id, or null when direct upload does not apply (small file, zips, or no S3-compatible storage).
 */
export async function startDirect(api, kind, file, opts = {}, onStage) {
  const model = kind === 'model';
  if (!model && kind !== 'table') return null;                       // image, audio and video zips keep their endpoints
  if (file.size + (opts.referenceFile?.size || 0) < DIRECT_FROM_BYTES) return null;
  let main;
  try {
    main = await api.presignUpload(model ? 'model' : 'csv', file.name, file.size);
  } catch (err) {
    if (err.status === 501) return null;
    throw err;
  }
  onStage?.('uploading');
  await putToStorage(main, file);
  const body = { upload_id: main.upload_id };
  if (!model) {
    if (opts.referenceFile) {
      const ref = await api.presignUpload('csv_reference', opts.referenceFile.name, opts.referenceFile.size);
      await putToStorage(ref, opts.referenceFile);
      body.reference_upload_id = ref.upload_id;
    }
    if (opts.splitColumn && opts.splitCutoff) Object.assign(body, { split_column: opts.splitColumn, split_cutoff: opts.splitCutoff });
    body.self_check = Boolean(opts.noReference);                     // always said: the server's default is a screening
  }
  return (await api.completeUpload(body)).job_id;
}

/**
 * Follows a job until it ends and returns a function that stops following. The job runs on the server whatever happens
 * here, so a dropped connection, a server error or rate limiting only pauses the wait (retried with backoff); a 4xx
 * other than 429 is a final answer (signed out, not allowed, job gone) and ends it.
 * Handlers: onStatus(status, retrying), onDone(result), onFail(message).
 */
export function followJob(api, id, { onStatus, onDone, onFail }, stage = 'queued') {
  let timer;
  let misses = 0;
  const tick = async () => {
    try {
      const s = await api.getJob(id);
      misses = 0;
      if (s.status === 'completed') return onDone(s.result);
      if (s.status === 'failed') return onFail(s.error || 'The analysis failed.');
      stage = s.status;
      onStatus(stage, false);
      timer = setTimeout(tick, 1500);
    } catch (e) {
      if (e.status >= 400 && e.status < 500 && e.status !== 429) return onFail(e.message);
      misses += 1;
      onStatus(stage, true);
      timer = setTimeout(tick, Math.min(15000, 1500 * 2 ** misses));
    }
  };
  tick();
  return () => clearTimeout(timer);
}
