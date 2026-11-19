// Run with `npm test` (Node's built-in test runner; no extra dependency).
import assert from 'node:assert/strict';
import { afterEach, beforeEach, describe, it, mock } from 'node:test';
import { setImmediate } from 'node:timers/promises';
import { DIRECT_FROM_BYTES, followJob, startDirect } from './uploads.js';

const MB = 1024 * 1024;
const small = { name: 'small.csv', size: MB };
const large = { name: 'large.csv', size: DIRECT_FROM_BYTES + MB };
const notImplemented = () => Object.assign(new Error('Direct upload needs S3-compatible storage.'), { status: 501 });

let api;
beforeEach(() => {
  api = { presignUpload: mock.fn(), completeUpload: mock.fn(), getJob: mock.fn() };
  globalThis.fetch = mock.fn(async () => ({ ok: true, status: 200 }));
});
afterEach(() => mock.timers.reset());

describe('startDirect', () => {
  it('leaves small files, and zips of any size, to the normal upload', async () => {
    assert.equal(await startDirect(api, 'table', small), null);
    assert.equal(await startDirect(api, 'images', large), null);
    assert.equal(api.presignUpload.mock.callCount(), 0);
  });

  it('counts the reference file towards the size', async () => {
    api.presignUpload.mock.mockImplementation(async () => { throw notImplemented(); });
    assert.equal(await startDirect(api, 'table', small, { referenceFile: { name: 'r.csv', size: DIRECT_FROM_BYTES } }), null);
    assert.equal(api.presignUpload.mock.callCount(), 1);
  });

  it('answers null on 501 (local storage) so the caller falls back', async () => {
    api.presignUpload.mock.mockImplementation(async () => { throw notImplemented(); });
    assert.equal(await startDirect(api, 'table', large), null);
    assert.equal(api.completeUpload.mock.callCount(), 0);
  });

  it('passes on other presign errors', async () => {
    api.presignUpload.mock.mockImplementation(async () => { throw Object.assign(new Error('quota'), { status: 429 }); });
    await assert.rejects(startDirect(api, 'table', large), /quota/);
  });

  it('uploads to storage with the reference and split, never sending Content-Length', async () => {
    const urls = [{ upload_id: 'u-main', url: 'https://bucket/main', headers: { 'Content-Length': '1', 'x-amz-meta': 'a' } }, { upload_id: 'u-ref', url: 'https://bucket/ref', headers: {} }];
    api.presignUpload.mock.mockImplementation(async () => urls.shift());
    api.completeUpload.mock.mockImplementation(async () => ({ job_id: 'j3' }));
    const ref = { name: 'ref.csv', size: MB };
    const stages = [];

    const id = await startDirect(api, 'table', large, { referenceFile: ref, splitColumn: 'day', splitCutoff: 'x' }, (s) => stages.push(s));

    assert.equal(id, 'j3');
    assert.deepEqual(stages, ['uploading']);
    assert.deepEqual(api.presignUpload.mock.calls.map((c) => c.arguments), [['csv', 'large.csv', large.size], ['csv_reference', 'ref.csv', MB]]);
    const [url, init] = fetch.mock.calls[0].arguments;
    assert.equal(url, 'https://bucket/main');
    assert.deepEqual(init, { method: 'PUT', headers: { 'x-amz-meta': 'a' }, body: large, credentials: 'omit' });
    assert.deepEqual(api.completeUpload.mock.calls[0].arguments[0], { upload_id: 'u-main', reference_upload_id: 'u-ref', split_column: 'day', split_cutoff: 'x', self_check: false });
  });

  it('asks for a screening when there is no reference', async () => {
    api.presignUpload.mock.mockImplementation(async () => ({ upload_id: 'u1', url: 'https://bucket/put', headers: {} }));
    api.completeUpload.mock.mockImplementation(async () => ({ job_id: 'j1' }));
    await startDirect(api, 'table', large, { noReference: true });
    assert.deepEqual(api.completeUpload.mock.calls[0].arguments[0], { upload_id: 'u1', self_check: true });
  });

  it('uses the model role for a large model file', async () => {
    api.presignUpload.mock.mockImplementation(async () => ({ upload_id: 'u', url: 'https://bucket/m', headers: {} }));
    api.completeUpload.mock.mockImplementation(async () => ({ job_id: 'j4' }));
    assert.equal(await startDirect(api, 'model', { name: 'big.pt', size: large.size }), 'j4');
    assert.deepEqual(api.presignUpload.mock.calls[0].arguments, ['model', 'big.pt', large.size]);
    assert.deepEqual(api.completeUpload.mock.calls[0].arguments[0], { upload_id: 'u' });
  });

  it('reports a failed storage PUT instead of queueing anything', async () => {
    api.presignUpload.mock.mockImplementation(async () => ({ upload_id: 'u', url: 'https://bucket/x', headers: {} }));
    globalThis.fetch = mock.fn(async () => ({ ok: false, status: 403 }));
    await assert.rejects(startDirect(api, 'table', large), /HTTP 403/);
    assert.equal(api.completeUpload.mock.callCount(), 0);
  });
});

describe('followJob', () => {
  const offline = () => Object.assign(new Error("Can't reach the Veritas server."), { status: 0 });
  const settle = () => setImmediate();
  async function run(steps, ms = 1500) {
    mock.timers.reset();
    mock.timers.enable({ apis: ['setTimeout'] });
    const seen ={ status: [], done: null, fail: null };
    followJob(api, 'j', { onStatus: (s, r) => seen.status.push([s, r]), onDone: (x) => { seen.done = x; }, onFail: (m) => { seen.fail = m; } });
    for (let i = 0; i < steps; i++) { await settle(); mock.timers.tick(ms); }
    await settle();
    return seen;
  }

  it('follows queued, running, done', async () => {
    const answers = [{ status: 'queued' }, { status: 'running' }, { status: 'completed', result: { verdict: 'CLEAN' } }];
    api.getJob.mock.mockImplementation(async () => answers.shift());
    const seen = await run(3);
    assert.deepEqual(seen.status, [['queued', false], ['running', false]]);
    assert.deepEqual(seen.done, { verdict: 'CLEAN' });
  });

  it('keeps watching through a dropped connection and a 502, then finishes', async () => {
    const answers = [() => { throw offline(); }, () => { throw Object.assign(new Error('Bad Gateway'), { status: 502 }); }, () => ({ status: 'completed', result: { ok: 1 } })];
    api.getJob.mock.mockImplementation(async () => answers.shift()());
    const seen = await run(4, 15000);
    assert.deepEqual(seen.status.map(([, retrying]) => retrying), [true, true]);
    assert.deepEqual(seen.done, { ok: 1 });
  });

  it('ends on a final answer such as an unknown job, and on a failed job', async () => {
    api.getJob.mock.mockImplementation(async () => { throw Object.assign(new Error('Job not found.'), { status: 404 }); });
    assert.equal((await run(0)).fail, 'Job not found.');
    api.getJob.mock.mockImplementation(async () => ({ status: 'failed', error: 'Only CSV files are accepted.' }));
    assert.equal((await run(0)).fail, 'Only CSV files are accepted.');
  });

  it('stops when told to', async () => {
    api.getJob.mock.mockImplementation(async () => ({ status: 'running' }));
    mock.timers.enable({ apis: ['setTimeout'] });
    const stop = followJob(api, 'j', { onStatus() {}, onDone() {}, onFail() {} });
    await settle();
    stop();
    mock.timers.tick(5000);
    await settle();
    assert.equal(api.getJob.mock.callCount(), 1);
  });
});
